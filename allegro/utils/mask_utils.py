from abc import ABC, abstractmethod
import torch
import random

    
class BaseNoiseAdder(ABC):
    
    @abstractmethod
    def add_noise(self, mask_pixel_values, mask):
        pass

    def __call__(self, mask_pixel_values, mask):
        return self.add_noise(mask_pixel_values, mask)
    
class GaussianNoiseAdder(BaseNoiseAdder):
    def __init__(self, mean=-3.0, std=0.5, clear_ratio=0.05):
        self.mean = mean
        self.std = std
        self.clear_ratio = clear_ratio

    def add_noise(self, masked_pixel_values, mask):
        if random.random() < self.clear_ratio:
            return masked_pixel_values
        noise_sigma = torch.normal(mean=self.mean, std=self.std, size=(masked_pixel_values.shape[0],), device=masked_pixel_values.device)
        noise_sigma = torch.exp(noise_sigma).to(dtype=masked_pixel_values.dtype)
        noise = torch.randn_like(masked_pixel_values) * noise_sigma[:, None, None, None, None]
        noise = torch.where(mask < 0.5, noise, torch.zeros_like(noise))
        return masked_pixel_values + noise