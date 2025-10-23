import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils_1d import setup_logging, get_data, plot_images, save_images  # Ensure you have these functions in utils.py
from modules import UNet_conditional, EMA  # Ensure these are in modules.py
import logging
from airfoil_dataset_1d import AirfoilDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import math
import matplotlib.pyplot as plt

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-3, beta_end=1e-3, num_airfoil_points=200, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        #self.beta = self._cosine_variance_schedule(noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.num_airfoil_points = num_airfoil_points*2
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 4e-5):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def noise_images(self, x, t, visualize=False):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, conditioning, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.num_airfoil_points)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, conditioning)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x
    
    def sample_with_repaint(self, model, n, conditioning, img_ref, mask, repaint_rounds=20, cfg_scale=3):
        """
        RePaint 采样：保留 mask 区域 (1=参考, 0=生成)
        """
        logging.info(f"Sampling {n} with RePaint (U={repaint_rounds})...")
        model.eval()
        device = self.device

        # 初始化噪声、mask、参考
        img = torch.randn((n, 1, self.num_airfoil_points), device=device)
        img_ref = img_ref.to(device)
        mask = mask.to(device).float()

        with torch.no_grad():
            for t in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps, desc="RePaint sampling"):
                t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
                alpha = self.alpha[t_tensor][:, None, None]
                alpha_hat = self.alpha_hat[t_tensor][:, None, None]
                beta = self.beta[t_tensor][:, None, None]
                for u in range(repaint_rounds):
                    # 1️⃣ 预测噪声
                    pred_noise = model(img, t_tensor, conditioning)
                    if cfg_scale > 0:
                        uncond_pred = model(img, t_tensor, None)
                        pred_noise = torch.lerp(uncond_pred, pred_noise, cfg_scale)

                    # 2️⃣ 反向一步去噪
                    if t > 1:
                        noise = torch.randn_like(img)
                    else:
                        noise = torch.zeros_like(img)
                    x_prev = (1 / torch.sqrt(alpha) *(img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise)+ torch.sqrt(beta) * noise)

                    # 3️⃣ 保留 mask 区域的参考噪声
                    if t < self.noise_steps - 1:
                        img_ref_t = self.noise_images(img_ref, t_tensor)
                    else:
                        img_ref_t = img_ref

                    # 合成（RePaint核心）
                    img = img_ref_t * mask + x_prev * (1 - mask)

                    # 4️⃣ 若未完成U轮，则重新加噪
                    if t > 1 and u < repaint_rounds - 1:
                        re_noise = torch.randn_like(img)
                        sqrt_one_minus_beta = torch.sqrt(1 - beta)
                        img = img * sqrt_one_minus_beta + re_noise * beta
        model.train()
        return img.clamp(-1, 1)
    
    def save_noised_airfoils(self, airfoils, airfoil_x, epoch, step, save_dir="noised_airfoils"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch}_step_{step}.jpg")
        self.plot_and_save_airfoils(airfoils, airfoil_x, save_path)
    
    def plot_and_save_airfoils(self, airfoils, airfoil_x, save_path):
        num_airfoils = airfoils.shape[0]
        fig, axs = plt.subplots(1, num_airfoils, figsize=(num_airfoils * 5, 5))
        
        if num_airfoils == 1:
            axs = [axs]
        
        for i in range(num_airfoils):
            ax = axs[i]
            airfoil = airfoils[i].cpu().numpy()
            ax.scatter(airfoil_x, airfoil[0, :], color='black')
            ax.set_title(f'Airfoil {i+1}')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
    
def chamfer_distance(pred_y, target_y, x_values):
    """
    Computes the Chamfer Distance between predicted and target y-values with consistent x-values.

    :param pred_y: Predicted y-values (batch_size, num_points)
    :param target_y: Target y-values (batch_size, num_points)
    :param x_values: The fixed x-values (num_points)
    :return: Chamfer distance
    """
    x_tensor = torch.tensor(x_values).unsqueeze(0).to(pred_y.device)
    x_values = x_tensor.repeat(pred_y.shape[0], 1)
    # Combine x and y values into full coordinate pairs
    pred = torch.stack([x_values, pred_y], dim=-1)
    target = torch.stack([x_values, target_y], dim=-1)
    
    # Calculate distances
    batch_size, num_points, _ = pred.shape
    pred = pred.unsqueeze(2).repeat(1, 1, num_points, 1)
    target = target.unsqueeze(1).repeat(1, num_points, 1, 1)
    dist = torch.norm(pred - target, dim=-1)
    
    # Get minimum distances
    min_dist_pred = torch.min(dist, dim=2)[0]
    min_dist_target = torch.min(dist, dim=1)[0]
    
    # Return the Chamfer distance
    return torch.mean(min_dist_pred) + torch.mean(min_dist_target)

def Design_gen(args):
    setup_logging(args.run_name)
    device = args.device
    dataset = AirfoilDataset(args.dataset_path, num_points_per_side=args.num_airfoil_points)
    model = UNet_conditional(c_in=1, c_out=1, cond_dim=args.cond_dim, time_dim=64, base_dim=16).to(device)
    model.load_state_dict(torch.load(args.diffusion_path, weights_only=True))
    diffusion = Diffusion(noise_steps=args.noise_step, num_airfoil_points=args.num_airfoil_points, device=device)
    cl = torch.linspace(-0.2, 1.5, 5).unsqueeze(1).to(device)
    sampled_images = diffusion.sample(model, n=5, conditioning=cl)
    airfoil_x = dataset.get_x()
    save_images(sampled_images, airfoil_x, os.path.join("results", args.run_name, f"{20001}_{args.noise_step}_steps.jpg"), cl)
    print("Done")
    
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_path', type=str, default="models/run_3/ckpt.pt")
    parser.add_argument('--run_name', type=str, default="test")
    parser.add_argument('--epochs', type=int, default=20001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_airfoil_points', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, default="coord_seligFmt/")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noise_step', type=int, default=1000) # 500
    args = parser.parse_args()
    Design_gen(args)

if __name__ == '__main__':
    launch()

    