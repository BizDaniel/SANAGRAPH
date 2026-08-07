import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm.auto import tqdm
import re
import os

from torch import tensor, float32, int64
from torch.nn import Module, ModuleList, ModuleDict, Linear
from torch.nn.functional import l1_loss
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

from utils import get_case_ids, get_one_hot_encodings

MISSING_VALUE = "MISSING_VALUE"

# function for create the graph

def get_one_hot_encoder(dataset: pd.DataFrame, key: str):
    datas = np.append(dataset[key].unique(), MISSING_VALUE)
    datas = datas.astype(np.str_)
    datas = datas.reshape([len(datas), 1])
    onehot = sklearn.preprocessing.OneHotEncoder()
    onehot.fit(datas)
    return onehot

def build_one_hot_encoders(tab_all, cat_features):
    return {key: get_one_hot_encoder(tab_all, key) for key in cat_features}

def group_by_case(tab):
    return {
        caseid: group.reset_index(drop=True).drop(columns="CaseID")
        for caseid, group in tab.groupby("CaseID")
    }

def add_new_timestamp(trace: pd.DataFrame):
    times = list(trace["time:timestamp"].copy())
    for i in range(1, len(times)):
        times[i] = times[i] - times[0]
    times[0] = 0.
    trace2 = trace.copy()
    trace2["time:timestamp"] = times
    return trace2

def get_node_features(trace, cat_features, real_features, encoders) -> dict:
    res = {}
    for key in trace:
        values = trace[key].values
        if key in cat_features:
            values = values.astype(np.str_)
            try:
                res[key] = tensor(
                    get_one_hot_encodings(encoders[key], values),
                    dtype=float32,
                    requires_grad=True
                )
            except ValueError:
                print(key)
                print(values)
        if key in real_features:
            res[key] = tensor(values, dtype=float32)
            res[key] = res[key].reshape(res[key].shape[0], 1)
    return res

def compute_edges_indexs(node_features: dict, prefix_len):
    res = {}
    keys = node_features.keys()
    indexes = [[i, i + 1] for i in range(prefix_len - 1)]

    for k in keys:
        if len(node_features[k]) != 1:
            if k == "Activity":
                res[(k, "followed_by", k)] = indexes
                for k2 in keys:
                    if k2 != k:
                        if len(node_features[k2]) == 1:
                            res[(k, "related_to", k2)] = [
                                [i, 0] for i in range(prefix_len)
                            ]
                        else:
                            res[(k, "related_to", k2)] = [
                                [i, i] for i in range(prefix_len)
                            ]
            else:
                res[(k, "related_to", k)] = indexes

    return res

def get_masked_trace_dict_mask(grouped_masked_trace, cat_features, real_features, caseid, newtimestamp):
    trace = grouped_masked_trace[caseid].copy()  

    null = trace.isnull()
    masks = {k: null[k] for k in trace.columns}
    mask_indexes = {k: [i for i in range(len(masks[k])) if masks[k][i]] for k in masks}

    trace["time:timestamp"] = newtimestamp

    for k in cat_features:
        for i in mask_indexes[k]:
            trace.loc[i, k] = MISSING_VALUE

    for k in real_features:
        for i in mask_indexes[k]:
            trace.loc[i, k] = -1

    masks = {k: torch.tensor(masks[k], dtype=torch.bool) for k in masks}

    return trace, masks

def build_prefixes_graph_from_trace(
    trace, cat_features, real_features, grouped_masked_datasets, nan_methods, encoders,
    caseid=None, mask_method=None
):
    X = []
    trace = add_new_timestamp(trace)
    node_features = get_node_features(trace, cat_features, real_features, encoders)

    if mask_method is not None:
        masked_methods = [mask_method]
    else:
        masked_methods = list(nan_methods)

    for j in range(len(masked_methods)):
        G = HeteroData()

        masked_trace, masks = get_masked_trace_dict_mask(
            grouped_masked_datasets[masked_methods[j]], cat_features, real_features, caseid,
            trace["time:timestamp"].values
        )

        G.masks = masks
        node_features_masked_trace = get_node_features(masked_trace, cat_features, real_features, encoders)

        for k in node_features_masked_trace:
            G[k].x = node_features_masked_trace[k]

        edges_indexes = compute_edges_indexs(node_features_masked_trace, len(trace))
        for k in edges_indexes:
            ce = [[], []]
            for i in range(len(edges_indexes[k])):
                ce[0].append(edges_indexes[k][i][0])
                ce[1].append(edges_indexes[k][i][1])
            edges_indexes[k] = ce
        for k in edges_indexes:
            G[k].edge_index = tensor(edges_indexes[k], dtype=int64)

        G.y = {}
        for k in node_features:
            if k in cat_features:
                G.y[k] = torch.max(node_features[k], 1)[1]
            else:
                G.y[k] = node_features[k].reshape(1, -1)[0].detach().clone()

        X.append(G)

    return X
   

def build_split_graphs(grouped_tab_split, cat_features, real_features, grouped_masked_datasets, nan_methods, encoders, case_ids=None):
    if case_ids is None:
        case_ids = list(grouped_tab_split.keys())

    X = []
    for caseid in tqdm(case_ids):
        trace = grouped_tab_split.get(caseid)
        if trace is not None and len(trace) > 2:
            graphs = build_prefixes_graph_from_trace(
                trace=trace,
                cat_features=cat_features,
                real_features=real_features,
                grouped_masked_datasets=grouped_masked_datasets,
                nan_methods=nan_methods,
                encoders=encoders,
                caseid=caseid,
            )
            X.extend(graphs)
    return X

def build_test_graphs_for_type(grouped_tab_test, cat_features, real_features, grouped_masked_datasets, nan_methods, encoders, mask_type, case_ids=None):
    if case_ids is None:
        case_ids = list(grouped_tab_test.keys())

    X = []
    for caseid in tqdm(case_ids):
        trace = grouped_tab_test.get(caseid)
        if trace is not None and len(trace) > 2:
            graphs = build_prefixes_graph_from_trace(
                trace=trace,
                cat_features=cat_features,
                real_features=real_features,
                grouped_masked_datasets=grouped_masked_datasets,
                nan_methods=nan_methods,
                encoders=encoders,
                caseid=caseid,
                mask_method=mask_type,
            )
            X.extend(graphs)
    return X

# Functions to define the HGNN

class HGNN(Module):

    def __init__(self, output_cat, output_real, nodes_relations, parameters, device):
        super().__init__()

        hid = parameters["hid"]
        layers = parameters["layers"]
        aggregation = parameters["aggregation"]

        self.output_cat = output_cat
        self.output_real = output_real

        self.convs = ModuleList()
        for _ in range(layers):
            conv = HeteroConv(
                {
                    relation: (
                        SAGEConv((-1, -1), aggr=aggregation, out_channels=hid, normalize=False)
                    )
                    for relation in nodes_relations
                },
                aggr=aggregation,
            )

            self.convs.append(conv)

        self.FC = ModuleDict()

        for k in output_cat:
            self.FC[k] = Linear(hid, output_cat[k], device=device)
        for k in output_real:
            self.FC[k] = Linear(hid, 1, device=device)

    def forward(self, batch):

        for i in range(len(self.convs)):
            batch.x_dict = self.convs[i](
                batch.x_dict, batch.edge_index_dict
            )
            batch.x_dict = {key: x.relu() for key, x in batch.x_dict.items()}

        output = {}

        for k in self.output_cat:
            output[k] = self.FC[k](batch.x_dict[k][batch.masks[k]])
        for k in self.output_real:
            output[k] = self.FC[k](batch.x_dict[k][batch.masks[k]]).reshape(1, -1)[0]

        return output

def train_hgnn(config, output_cat, output_real, edge_types, X_train, X_valid, device, epochs=20):
    print(config)

    net = HGNN(
        parameters=config,
        output_cat=output_cat,
        output_real=output_real,
        nodes_relations=edge_types,
        device=device,
    )
    net = net.to(device)

    losses = {}

    for k in output_cat:
        losses[k] = nn.CrossEntropyLoss()
    for k in output_real:
        losses[k] = nn.L1Loss()

    train_loader = DataLoader(X_train, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(X_valid, batch_size=config["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_model = None
    best_loss = 0
    patience = 5
    pat_count = 0

    torch.cuda.empty_cache()

    for epoch in tqdm(range(0, epochs)):

        net.train()
        for _, x in enumerate(train_loader):
            x = x.to(device)

            all_labels = x.y
            labels = {k: all_labels[k][x.masks[k]] for k in all_labels}

            optimizer.zero_grad()
            outputs = net(x)

            losses_step = {k: losses[k](outputs[k], labels[k]) for k in losses}

            total_loss = 0
            for k in losses_step:
                total_loss += losses_step[k]

            total_loss.backward()
            optimizer.step()

        running_total_loss = []

        net.eval()
        with torch.no_grad():
            for i, x in enumerate(valid_loader):
                x = x.to(device)

                all_labels = x.y
                labels = {k: all_labels[k][x.masks[k]] for k in all_labels}

                outputs = net(x)

                losses_step = {k: losses[k](outputs[k], labels[k]) for k in losses}

                running_total_loss.append(sum(list(losses_step.values())))

        val_loss = sum(running_total_loss) / len(running_total_loss)

        if epoch == 0:
            best_model = deepcopy(net)
            best_loss = val_loss
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(net)
                pat_count = 0
            if pat_count == patience:
                return best_model
        pat_count += 1

    return best_model

def test_hgnn(net, output_cat, output_real, device, test_graphs):


    test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)

    losses = {}

    for k in output_cat:
        losses[k] = nn.CrossEntropyLoss()
    for k in output_real:
        losses[k] = nn.L1Loss()

    predictions_categorical = {k: [] for k in output_cat}
    target_categorical = {k: [] for k in output_cat}

    avg_MAE = {k: [] for k in output_real}
    prediction_numerical = {k: [] for k in output_real}
    target_numerical = {k: [] for k in output_real}

    total_loss = []

    net.eval()
    with torch.no_grad():
        for _, x in enumerate(test_loader):
            x = x.to(device)

            all_labels = x.y
            labels = {k: all_labels[k][x.masks[k]] for k in all_labels}

            outputs = net(x)

            losses_step = {k: losses[k](outputs[k], labels[k]).item() for k in losses}
            total_loss.append(sum(list(losses_step.values())))

            for k in output_cat:
                predictions_categorical[k].append(
                    torch.argmax(torch.softmax(outputs[k], dim=1), 1)
                )
                target_categorical[k].append(labels[k])

            for k in output_real:
                prediction_numerical[k].append(
                    outputs[k]
                )
                target_numerical[k].append(
                    labels[k]
                )
                avg_MAE[k].append(losses_step[k])

    for k in predictions_categorical:
        predictions_categorical[k] = torch.cat(predictions_categorical[k])
        target_categorical[k] = torch.cat(target_categorical[k])

    for k in prediction_numerical:
        prediction_numerical[k] = torch.cat(prediction_numerical[k])
        target_numerical[k] = torch.cat(target_numerical[k])

    MAE = {
        k: l1_loss(prediction_numerical[k], target_numerical[k]).item()
        for k in output_real
    }

    accuracy = {
        k: multiclass_accuracy(
            predictions_categorical[k],
            target_categorical[k],
            num_classes=output_cat[k],
        )
        for k in output_cat
    }

    avg_MAE = {k: sum(avg_MAE[k]) / len(avg_MAE[k]) for k in avg_MAE}

    Average_total_loss = sum(total_loss) / len(total_loss)

    res = (
        {f"{k}_acc": accuracy[k].item() for k in accuracy}
        | {f"{k}_mae": avg_MAE[k] for k in avg_MAE}
        | {"AVG_total_loss": Average_total_loss}
        | {f"MAE_{k}": MAE[k] for k in MAE}
    )

    print(res)

    return res

# Useful functions for ablation

def convert_feature_name(feature): 
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', feature)

def get_feature_variants(dataset_info):
    variants = []
    for f in dataset_info["categorical"] + dataset_info["numerical"]:
        cat = [c for c in dataset_info["categorical"] if c != f]
        num = [n for n in dataset_info["numerical"] if n != f]
        variants.append((f, cat, num))
    return variants




