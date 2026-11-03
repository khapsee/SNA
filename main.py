import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_loader import load_data
from graph_builder import build_graph_from_csv
from train import train_model, get_model
from predict import predict_citations

MODEL_PATH = "trained_gcn.pt"
N_RUNS = 5

def run_single():
    print("Single-run")
    full_df     = load_data("combined_author_data.csv")
    all_authors = full_df['author_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_authors)

    split         = int(0.8 * len(all_authors))
    train_authors = set(all_authors[:split])
    test_authors  = set(all_authors[split:])

    data, id_map = build_graph_from_csv(full_df, test_author_ids=test_authors)

    if os.path.exists(MODEL_PATH):
        model = get_model(data.num_node_features)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    else:
        model = train_model(data, model_path=MODEL_PATH)

    predictions = predict_citations(model, data, test_authors, id_map)

    with open("predicted_citations.json", "w") as f:
        json.dump(predictions, f, indent=2)

    with torch.no_grad():
        model.eval()
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = model(data.x, data.edge_index).detach()[data.test_mask].cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)

        print("\nTest Evaluation Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")

    test_ids         = []
    predicted_values = []
    actual_values    = []

    for aid in test_authors:
        if aid in id_map:
            idx = id_map[aid]
            test_ids.append(aid)
            predicted_values.append(predictions.get(aid, np.nan))
            actual_values.append(data.y[idx].item())

    df_out = pd.DataFrame({
        'author_id': test_ids,
        'predicted_citations': predicted_values,
        'actual_citations': actual_values
    })
    df_out.to_csv("prediction_vs_actual.csv", index=False)
    print("Saved to prediction_vs_actual.csv")

def run_multi():
    print("Multi-run (5 runs)")
    metrics = []

    for run in range(N_RUNS):
        print(f"\nRun {run+1}")
        seed          = 42 + run
        np.random.seed(seed)
        torch.manual_seed(seed)

        full_df       = load_data("combined_author_data.csv")
        all_authors   = full_df['author_id'].unique()
        np.random.shuffle(all_authors)

        split         = int(0.8 * len(all_authors))
        train_authors = set(all_authors[:split])
        test_authors  = set(all_authors[split:])

        data, id_map  = build_graph_from_csv(full_df, test_author_ids=test_authors)
        model         = train_model(data)

        with torch.no_grad():
            model.eval()
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = model(data.x, data.edge_index).detach()[data.test_mask].cpu().numpy()

            y_true_exp = y_true
            y_pred_exp = np.clip(y_pred, 0, 1000)

            rmse = np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
            mae  = mean_absolute_error(y_true_exp, y_pred_exp)
            r2   = r2_score(y_true_exp, y_pred_exp)

            metrics.append([rmse, mae, r2])
            print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

            if run == N_RUNS - 1:
                author_ids = np.array([aid for aid, idx in id_map.items() if data.test_mask[idx]])
                pred_df = pd.DataFrame({
                    "author_id": author_ids,
                    "predicted_citations": y_pred_exp,
                    "actual_citations": y_true_exp
                })
                pred_df.to_csv("prediction_vs_actual.csv", index=False)
                print("Saved to prediction_vs_actual.csv (final run)")

    metrics_df = pd.DataFrame(metrics, columns=["RMSE", "MAE", "R2"])
    metrics_df.to_csv("gcn_metrics_over_runs.csv", index=False)
    print("\nSummary over 5 runs")
    print(metrics_df.describe())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_run', action='store_true', help="Run multi-seed repeated experiment (5 runs)")
    args = parser.parse_args()

    if args.multi_run:
        run_multi()
    else:
        run_single()
