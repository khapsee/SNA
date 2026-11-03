# SNA - GNN Citation Prediction

This project uses a Graph Convolutional Network (GCN) to predict the future citation count of academic authors by modeling co-authorship networks and individual features.

- Structural features from co-authorship graphs
- Individual features (h-index, publication count)
- Graph Neural Networks (GCN)
- Baseline comparison with Linear Regression

## Installation

1. ### Clone the Repository

```
git clone https://gitlab2.informatik.uni-wuerzburg.de/s469439/sna-gnn-citation-prediction.git
cd sna-gnn-citation-prediction
```


2. ### Install Required Packages
```
pip install pandas numpy scikit-learn torch torch-geometric matplotlib
```

3. ### Train the GCN Model

Run `python main.py`

Results saved to <b>prediction_vs_actual.csv</b>

4. ### Test the performance of the GCN model over multiple runs

Run `python main.py --multi`

Results saved to <b>gcn_metrics_over_runs.csv</b>

5. ### Compare GCN model against Linear Regression (No Graph)

Run `python baseline_linear.py`

Results saved to <b>baseline_prediction_vs_actual.csv</b>

Then run `python compare_linear_vs_gcn.py`

6. ### Example Result
```
Run 5
RMSE: 2.13, MAE: 1.73, R²: 0.3949

Paired t-test: Linear Regression vs GCN
t-statistic = -69.9161
p-value     = 0.0000
```