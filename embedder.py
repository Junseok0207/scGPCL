import os
import torch
import numpy as np
from src.argument import printConfig
from torch_geometric.loader import HGTLoader
from sklearn.metrics.cluster import adjusted_rand_score
from src.data import read_data, normalize, construct_graph

class embedder:
    def __init__(self, args):
        self.args = args
        printConfig(args)

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        
        self.model_path = os.path.join('./weights', f'{str(self.args.recon)}_{str(self.args.name)}.pt')
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def _init_dataset(self, seed):
        adata = read_data(self.args.name)
        self.adata = normalize(adata, HVG=self.args.HVG, size_factors=self.args.sf, logtrans_input=self.args.log, normalize_input=self.args.normal)
        self.c_g_graph = construct_graph(self.adata.raw.X, self.adata.X, self.adata.n_obs, self.adata.n_vars)

        input_idx = torch.arange(self.adata.n_obs)

        generator1 = torch.Generator()
        generator1.manual_seed(seed)
        sampler1 = torch.utils.data.RandomSampler(input_idx, generator=generator1)
        
        generator2 = torch.Generator()
        generator2.manual_seed(seed)
        sampler2 = torch.utils.data.RandomSampler(input_idx, generator=generator2)

        NS = eval(self.args.ns)
        cell_nodes = ('cell', torch.ones(self.adata.n_obs).bool())
        self.train_loader1 = HGTLoader(self.c_g_graph, num_samples=NS,  sampler=sampler1, input_nodes=cell_nodes, batch_size=self.args.batch_size)
        self.train_loader2 = HGTLoader(self.c_g_graph, num_samples=NS, sampler=sampler2, input_nodes=cell_nodes, batch_size=self.args.batch_size)
        self.eval_loader = HGTLoader(self.c_g_graph, num_samples=NS, shuffle=False, input_nodes=cell_nodes, batch_size=self.args.batch_size)

    def Pretrain_Evaluate_Convergence(self, epoch):
        
        flag=0
        self.model.eval()
        cell_rep = self.model.predict_full_cell_rep(self.eval_loader, self.gene_embedding)
        y_pred = self.model.predict_celltype(cell_rep)
        
        if epoch == self.args.warmup+1:
            self.old_celltype_result = y_pred
        else:
            ari = adjusted_rand_score(self.old_celltype_result, y_pred)
            self.old_celltype_result = y_pred
            if ari > self.args.r:
                flag=1
                print("Reach tolerance threshold. Stopping pre-training.")

        return flag

    def Fine_Evaluate_Convergence(self):
        
        flag = 0
        num = self.adata.X.shape[0]

        delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
        self.y_pred_last = self.y_pred
        if delta_label < self.args.tol:
            print('delta_label ', delta_label, '< tol ', self.args.tol)
            print("Reach tolerance threshold. Stopping fine-tuning.")
            flag = 1

        return flag