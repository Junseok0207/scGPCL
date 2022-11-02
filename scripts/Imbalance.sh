cd ..

device=0

python main.py --name Imb1 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name Imb2 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name Imb3 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name Imb4 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name Imb5 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device