import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from data_loader import load_data

def run_baseline_linear():
    df = load_data("combined_author_data.csv")

    all_authors                   = df['author_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_authors)
    split                         = int(0.8 * len(all_authors))
    train_ids                     = set(all_authors[:split])
    test_ids                      = set(all_authors[split:])
    author_df                     = df.sort_values('publication_year').drop_duplicates('author_id', keep='last')

    author_df['works_count']      = author_df['works_count'].fillna(0)
    author_df['h_index']          = author_df['h_index'].fillna(0)
    author_df['cited_by_count_x'] = author_df['cited_by_count_x'].fillna(0)

    train_df                      = author_df[author_df['author_id'].isin(train_ids)].copy()
    test_df                       = author_df[author_df['author_id'].isin(test_ids)].copy()

    features                      = ['works_count', 'h_index']
    X_train                       = train_df[features].values
    X_test                        = test_df[features].values
    y_train                       = train_df['cited_by_count_x'].values
    y_test                        = test_df['cited_by_count_x'].values

    scaler                        = StandardScaler()
    X_train                       = scaler.fit_transform(X_train)
    X_test                        = scaler.transform(X_test)

    model                         = LinearRegression()
    model.fit(X_train, y_train)
    y_pred                        = model.predict(X_test)
    y_pred                        = np.clip(y_pred, 0, None)

    rmse                          = np.sqrt(mean_squared_error(y_test, y_pred))
    mae                           = mean_absolute_error(y_test, y_pred)
    r2                            = r2_score(y_test, y_pred)
    print("\nLinear Regression Baseline:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R²:   {r2:.4f}")

    results = pd.DataFrame({
        'author_id': test_df['author_id'],
        'predicted_citations': y_pred,
        'actual_citations': y_test
    })
    results.to_csv("baseline_prediction_vs_actual.csv", index=False)

if __name__ == '__main__':
    run_baseline_linear()
