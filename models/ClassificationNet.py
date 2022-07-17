import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


class ClassificationNet(torch.nn.Module):
    def __init__(self, out_channels: int, point_dim: int = 3, k: int = 20, aggr: str = 'max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * point_dim, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5)

    def forward(self, data):
        pos, batch = data.pos, data.batch

        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)