import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import BatchNorm, SAGEConv, HeteroConv


class GNN_Encoder(nn.Module):
    def __init__(self, metadata, layer_sizes):
        super().__init__()

        self.convs = nn.ModuleList()
        self.cell_batchnorms = nn.ModuleList()
        self.gene_batchnorms = nn.ModuleList()
        for hidden_channels in layer_sizes:
            conv = HeteroConv({
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in metadata[1]
                })    
            self.convs.append(conv)
            self.cell_batchnorms.append(BatchNorm(hidden_channels))
            self.gene_batchnorms.append(BatchNorm(hidden_channels))

    def forward(self, data):
        for i, conv in enumerate(self.convs):
            x_dict = conv(data.x_dict, data.edge_index_dict)
            x_dict['cell'] = self.cell_batchnorms[i](x_dict['cell'])
            x_dict['gene'] = self.gene_batchnorms[i](x_dict['gene'])

        return x_dict['cell']

    def reset_parameters(self):
        self.model.reset_parameters()


