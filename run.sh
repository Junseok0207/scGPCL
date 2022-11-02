device=0

python main.py --recon zinb --name Mouse_ES --HVG 0.2 --n_clusters 4 --lr 0.001 --device $device
python main.py --recon zinb --name MCA --HVG 0.2 --n_clusters 16 --lr 0.00001 --device $device
python main.py --recon zinb --name Zeisel --HVG 0.2 --n_clusters 9 --lr 0.0001 --device $device
python main.py --recon zinb --name Worm_neuron_cells --HVG 0.2 --n_clusters 10 --lr 0.0001 --device $device
python main.py --recon zinb --name 10X_PBMC --HVG 0.2 --n_clusters 8 --lr 0.001 --device $device
python main.py --recon zinb --name Human_kidney_cells --HVG 0.2 --n_clusters 11 --lr 0.0001 --device $device
python main.py --recon zinb --name Shekhar --HVG 1.0 --n_clusters 19 --lr 0.001 --device $device

# MSE-based reconstruction loss
python main.py --recon mse --name Mouse_ES --HVG 0.2 --n_clusters 4 --lr 0.001 --device $device
python main.py --recon mse --name MCA --HVG 0.2 --n_clusters 16 --lr 0.00001 --device $device
python main.py --recon mse --name Zeisel --HVG 0.2 --n_clusters 9 --lr 0.0001 --device $device
python main.py --recon mse --name Worm_neuron_cells --HVG 0.2 --n_clusters 10 --lr 0.0001 --device $device
python main.py --recon mse --name 10X_PBMC --HVG 0.2 --n_clusters 8 --lr 0.001 --device $device
python main.py --recon mse --name Human_kidney_cells --HVG 0.2 --n_clusters 11 --lr 0.0001 --device $device
python main.py --recon mse --name Shekhar --HVG 1.0 --n_clusters 19 --lr 0.001 --device $device
