cd ..

device=0

python main.py --name Sig1 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name Sig2 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name Sig3 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name Sig4 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
