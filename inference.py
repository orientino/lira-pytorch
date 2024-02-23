# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--savedir", default="exp/cifar10", type=str)
args = parser.parse_args()


def run():
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    # Dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )
    datadir = Path().home() / "opt/data/cifar"
    train_ds = CIFAR10(root=datadir, train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

    # Infer the logits
    for path in os.listdir(args.savedir):
        m = models.resnet18(weights=None, num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.load_state_dict(torch.load(os.path.join(args.savedir, path, "model.pt")))
        m.to(DEVICE)
        m.eval()

        logits = []
        with torch.no_grad():
            for x, _ in tqdm(train_dl):
                x = x.to(DEVICE)
                outputs = m(x)
                logits.append(outputs.cpu().numpy())
        logits = np.concatenate(logits)[:, None, :]  # TODO remove after adding augs
        print(logits.shape)

        np.save(os.path.join(args.savedir, path, "logits.npy"), logits)


if __name__ == "__main__":
    run()
