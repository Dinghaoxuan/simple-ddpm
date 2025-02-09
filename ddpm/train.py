import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from model import UNet
from scheduler import GradualWarmupScheduler
from diffusion import GaussianDiffusionTrainer


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
    
    net_model = UNet(T=model_config["T"], 
                     dim=model_config["dim"], 
                     dim_scale=model_config["dim_scale"], 
                     attn=model_config["attn"],
                     num_res_blocks=model_config["num_res_blocks"],
                     dropout=model_config["dropout"]).to(device)
    
    if model_config["load_weight"] is not None:
        model_dict = torch.load(os.path.join(model_config["save_dir"], model_config["load_weight"]), map_location=device)
        net_model.load_state_dict(model_dict)

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=model_config["lr"], weight_decay=1e-4)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=model_config["epoch"],
                                                              eta_min=0,
                                                              last_epoch=-1)
    warmup_scheduler = GradualWarmupScheduler(optimizer=optimizer, 
                                              multiplier=model_config["multiplier"],
                                              warm_epoch=model_config["epoch"] // 10,
                                              after_scheduler=cosine_scheduler)
    
    trainer = GaussianDiffusionTrainer(net_model, model_config["beta_1"], model_config["beta_T"], model_config["T"]).to(device)
    
    
    