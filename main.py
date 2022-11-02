import torch
from src.utils import set_seed
from src.argument import parse_args

def main():
    set_seed()
    args, _ = parse_args()
    torch.set_num_threads(3)

    if args.recon == 'mse':
        from models import scGPCL_MSE_Trainer
        embedder = scGPCL_MSE_Trainer(args)

    elif args.recon == 'zinb':
        from models import scGPCL_Trainer
        embedder = scGPCL_Trainer(args)
    
    embedder.train()

if __name__ == "__main__":
    main()