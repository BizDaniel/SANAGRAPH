# SANAGRAPH

## Libraries

The code was developed using `Python 3.12`.


All libraries can be installed using the command 

`pip install -r req.txt`

## Datasets

The `data` folder contains the event logs already preprocessed into `.csv` files

In oder to create the corresponsing graph encoding, the `Create_graphs.ipynb` notebook can be used

## Model

The model can be trained and tested using the `main_notebook_version.ipynb` notebook.

## Main files

The work focuses on creating ablated graphs, training and evaluating each variant, analyzing the results against the full-feature model.

```text
SANAGRAPH/
├── data/
│   ├── Create_ablated_graphs.ipynb   # Generate ablated graphs
│   ├── pipeline.py                 # Graph, model, training and evaluation utilities
│   └── Create_graphs.ipynb          # Original graph generation
├── ablation_training_loop.ipynb     # Train and evaluate each variant
├── main_notebook_version.ipynb      # Original model workflow
├── results_CAISE/
│   └── ablation/                   # Ablation results
└── analysis/
    ├── 01_ablation_tables.ipynb     # Result comparisons and statistical analysis
    ├── 02_ablation_visualization.ipynb # Plot results
    ├── tables/                     # Analysis tables
    └── figures/                    # Generated figures
```
