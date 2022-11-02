import torch
import scanpy as sc
import numpy as np
from torch_geometric.data import HeteroData

def read_data(name):

    path = f'./data/{name}.h5'
    adata = sc.read(path)

    return adata

def normalize(adata, HVG=0.2, filter_min_counts=True, size_factors=True, logtrans_input=True, normalize_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    n = int(adata.X.shape[1] * HVG)
    hvg_gene_idx = np.argsort(adata.X.var(axis=0))[-n:]
    adata = adata[:,hvg_gene_idx]

    adata.raw = adata.copy()

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def construct_graph(Unnormalized_featureMatrix, featureMatrix, num_cell, num_gene):

    X = torch.tensor(Unnormalized_featureMatrix)

    cells, genes = torch.nonzero(X, as_tuple=True)

    data = HeteroData()
    data['cell'].x = torch.tensor(featureMatrix)

    data['cell', 'to', 'gene'].edge_index = torch.stack((cells,genes))
    data['gene', 'to', 'cell'].edge_index = torch.stack((genes,cells))

    data['cell'].num_nodes = num_cell
    data['gene'].num_nodes = num_gene  

    data['cell']['n_id'] = torch.arange(num_cell)
    data['gene']['n_id'] = torch.arange(num_gene)

    return data
