import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


class TrackingNet(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max', point_dim=2):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv1 = DynamicEdgeConv(MLP([2 * point_dim, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5)

    def forward(self, data):
        f_size = data.shape[0]*data.shape[1]

        batch = [[i] * data.shape[1] for i in range(len(data))]
        batch = [e for sub_list in batch for e in sub_list]
        batch = torch.LongTensor(batch).to(self.device)

        data = torch.reshape(data, (f_size, data.shape[2]))

        x1 = self.conv1(data, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))

        out = global_max_pool(out, batch)
        out = self.mlp(out)

        return F.log_softmax(out, dim=1)