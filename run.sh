python3 train.py --epochs 100 --shadow_id 0 --debug
python3 train.py --epochs 100 --shadow_id 1 --debug
python3 train.py --epochs 100 --shadow_id 2 --debug
python3 train.py --epochs 100 --shadow_id 3 --debug
python3 train.py --epochs 100 --shadow_id 4 --debug
python3 train.py --epochs 100 --shadow_id 5 --debug
python3 train.py --epochs 100 --shadow_id 6 --debug
python3 train.py --epochs 100 --shadow_id 7 --debug
python3 train.py --epochs 100 --shadow_id 8 --debug
python3 train.py --epochs 100 --shadow_id 9 --debug
python3 train.py --epochs 100 --shadow_id 10 --debug
python3 train.py --epochs 100 --shadow_id 11 --debug
python3 train.py --epochs 100 --shadow_id 12 --debug
python3 train.py --epochs 100 --shadow_id 13 --debug
python3 train.py --epochs 100 --shadow_id 14 --debug
python3 train.py --epochs 100 --shadow_id 15 --debug

python3 inference.py --savedir exp/cifar10
python3 score.py --savedir exp/cifar10
python3 plot.py --savedir exp/cifar10

