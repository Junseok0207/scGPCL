cd ..

device=0

# 3 cell types
python main.py --name SimT3_1 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name SimT3_2 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name SimT3_3 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name SimT3_4 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device
python main.py --name SimT3_5 --HVG 1.0 --n_clusters 3 --lr 0.01 --device $device

# 6 cell types
python main.py --name SimT6_1 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name SimT6_2 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name SimT6_3 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name SimT6_4 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device
python main.py --name SimT6_5 --HVG 1.0 --n_clusters 6 --lr 0.01 --device $device