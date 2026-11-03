import torch
from torch_geometric.data import Data
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def build_graph_from_csv(df, test_author_ids=None):
    print("DataFrame columns:", df.columns.tolist())

    author_ids = df['author_id'].unique()
    id_map     = {author_id: idx for idx, author_id in enumerate(author_ids)}

    work_to_authors = defaultdict(set)
    for row in df.itertuples():
        if row.author_id in id_map:
            work_to_authors[row.work_id].add(id_map[row.author_id])

    edge_set    = set()
    for authors in work_to_authors.values():
        authors = list(authors)
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                edge_set.add((authors[i], authors[j]))
                edge_set.add((authors[j], authors[i]))

    edge_index            = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous() if edge_set else torch.empty((2, 0), dtype=torch.long)
    all_author_ids        = pd.DataFrame({'author_id': list(id_map.keys())})
    author_df             = all_author_ids.merge(df.drop_duplicates(subset='author_id'), on='author_id', how='left')
    author_df['node_idx'] = author_df['author_id'].map(id_map)

    for col in ['works_count', 'h_index', 'publication_year']:
        if col not in author_df.columns:
            author_df[col] = 0
        author_df[col]     = author_df[col].fillna(0)

    author_years             = df.groupby('author_id')['publication_year'].max().reset_index()
    author_years.columns     = ['author_id', 'publication_year']
    author_df.drop(columns=['publication_year'], errors='ignore', inplace=True)
    author_df                = author_df.merge(author_years, on='author_id', how='left')
    author_df                = author_df.sort_values(by='node_idx')

    feature_cols = ['works_count', 'h_index']
    for col in feature_cols:
        if col not in author_df.columns:
            author_df[col] = 0.0

    scaler                  = StandardScaler()
    author_df[feature_cols] = scaler.fit_transform(author_df[feature_cols])
    x                       = torch.tensor(author_df[feature_cols].values, dtype=torch.float)

    citation_col            = 'cited_by_count_y' if 'cited_by_count_y' in author_df.columns else 'cited_by_count_x'
    author_df[citation_col] = author_df[citation_col].fillna(0)

    citations = author_df[citation_col].copy()
    if test_author_ids:
        is_test             = author_df['author_id'].isin(test_author_ids)
        citations[~is_test] = citations[~is_test].clip(0, 1000)

    author_df['citations_log'] = np.log1p(citations)
    y = torch.tensor(author_df['citations_log'].values, dtype=torch.float).view(-1, 1)

    publication_year = torch.tensor(author_df['publication_year'].values, dtype=torch.long)

    author_df['is_test']  = author_df['author_id'].isin(test_author_ids if test_author_ids else [])
    author_df['is_train'] = ~author_df['is_test']

    train_mask  = torch.tensor(author_df['is_train'].values, dtype=torch.bool)
    test_mask   = torch.tensor(author_df['is_test'].values, dtype=torch.bool)

    data                    = Data(x=x, edge_index=edge_index, y=y)
    data.publication_year   = publication_year
    data.train_mask         = train_mask
    data.test_mask          = test_mask

    return data, id_map
