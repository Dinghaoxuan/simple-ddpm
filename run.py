from ddpm.train import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "test",
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "dim": 128,
        "dim_scale": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "clip_grad": 1.0,
        "device": "cuda:0",
        "load_weight": "/root/simple-ddpm/exp/ckpt_17.pt",
        "save_dir": "/root/autodl-tmp/simple-ddpm/exp/",
        "test_load_weight": "ckpt_64.pt",
        "sampled_dir": "/root/autodl-tmp/simple-ddpm/sampledimgs",
        "sampled_noise_name": "noise_image.png",
        "sampled_image_name": "sampled_image.png",
        "nrow": 8
    }
    
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)
        
if __name__=="__main__":
    main()