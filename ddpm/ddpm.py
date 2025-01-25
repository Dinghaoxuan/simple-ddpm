import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from model import UNet


def train(model_config):
    device = torch.device(model_config["device"])
    
    dataset = CIFAR10(
        root = "./CIFAR10",
        train = True,
        download = True,
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    )
    
    dataloader = DataLoader(dataset = dataset, 
                            batch_size = model_config["batch_size"],
                            shuffle = True,
                            num_workers = 4,
                            drop_last = True,
                            pin_memory = True)
    
    net_model = UNet()
    
    