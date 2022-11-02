import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Logical
    parser.add_argument('--save_model', action='store_true', default=False)

    # Experiments
    parser.add_argument('--name', type=str, default='Zeisel', help='Worm_neuron_cells, 10X_PBMC, Kolodziejczyk, Chung, Zeisel, Klein, 3000_3000_0.32')
    parser.add_argument('--n_clusters', type=int, default=9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=2)

    # Preprocessing
    parser.add_argument('--HVG', type=float, default=0.2)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--sf', action='store_false', default=True)
    parser.add_argument('--log', action='store_false', default=True)
    parser.add_argument('--normal', action='store_false', default=True)

    # Layers
    parser.add_argument("--layers", nargs='?', default='[256,64]', help='[256, 128, 64], [256,64]')
    
    # learning
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--fine_lr', default=1.0, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--fine_epochs', type=int, default=200)    
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument("--ns", nargs='?', default='[2048,1024]')
    parser.add_argument('--recon', type=str, default='zinb', help='mse,zinb')

    # Hyper-Parameters
    parser.add_argument('--tau', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument("--granularity", nargs='?', default='[1,1.5,2]')
    parser.add_argument('--lam1', default=1.0, type=float)
    parser.add_argument('--lam2', default=0.05, type=float)
    parser.add_argument('--lam3', default=1.0, type=float)

    parser.add_argument('--r', default=0.99, type=float, help='threshold to terminate pre-training stage')
    parser.add_argument('--tol', default=0.001, type=float, help='tolerance for delta clustering labels to terminate fine-tuning stage')

    # Augmentation
    parser.add_argument("--df_1", type=float, default=0.2)
    parser.add_argument("--df_2", type=float, default=0.2)
        
    return parser.parse_known_args()

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{} <- {} / ".format(name, val)
        st += st_

    return st[:-1]



