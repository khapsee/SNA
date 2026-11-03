import json
import torch
from train import GCN

def load_model(data, model_path="trained_gcn.pt"):
    model = GCN(data.num_node_features, 32, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_citations(model, data, author_ids, id_map):
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).detach().squeeze()
        predictions = predictions.clamp(min=0, max=15)
        predictions = torch.expm1(predictions)

        results = {}
        for author_id in author_ids:
            idx = id_map.get(author_id)
            if idx is not None:
                results[author_id] = predictions[idx].item()
    return results

def load_target_authors(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
