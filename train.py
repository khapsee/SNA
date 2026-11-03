import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def get_model(input_dim):
    return GCN(input_dim, 32, 1)

def train_model(data, model_path="trained_gcn.pt", epochs=200, lr=0.01):
    model     = get_model(data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index)
        loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred_test = torch.expm1(pred[data.test_mask])
            true_test = torch.expm1(data.y[data.test_mask])
            print("Sample test prediction:", pred_test[:3].squeeze())
            print("Sample test actual:", true_test[:3].squeeze())

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                val_loss = loss_fn(pred[data.test_mask], data.y[data.test_mask])
                print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    return model
