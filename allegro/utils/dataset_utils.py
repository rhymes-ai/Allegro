import torch
from PIL import Image
from einops import rearrange
import numpy as np
from torchvision import transforms
from torchvision.transforms import Lambda
import jsonlines
import pandas as pd
from tqdm import tqdm

from allegro.dataset.transform import ToTensorVideo, CenterCropResizeVideo

class Collate:
    def __init__(self, args):
        if args.num_frames == 1:
            self.latent_thw = (1, args.max_height // args.vae_stride_h, args.max_width // args.vae_stride_w)
        else:
            self.latent_thw = (args.num_frames // args.vae_stride_t, args.max_height // args.vae_stride_h, args.max_width // args.vae_stride_w)

    def __call__(self, batch):
        batch_tubes = [i['pixel_values'] for i in batch]  # b [c t h w]
        input_ids = [i['input_ids'] for i in batch]  # b [1 l]
        cond_mask = [i['cond_mask'] for i in batch]  # b [1 l]
        attention_mask = [torch.ones(self.latent_thw, dtype=i['pixel_values'].dtype) for i in batch] # b [t h w]
        
        batch_tubes = torch.stack(batch_tubes)  # b c t h w
        input_ids = torch.stack(input_ids)  # b 1 l
        cond_mask = torch.stack(cond_mask)  # b 1 l
        attention_mask = torch.stack(attention_mask)  # b t h w

        return batch_tubes, attention_mask, input_ids, cond_mask

def preprocess_images(ae, images, height, width, device, dtype):
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = transforms.Compose([
        ToTensorVideo(),
        CenterCropResizeVideo((height, width)),
        norm_fun
    ])
    if len(images) == 1:
        conditional_images_indices = [0]
    elif len(images) == 2:
        conditional_images_indices = [0, -1]
    else:
        print("Only support 1 or 2 condition images!")
        raise NotImplementedError
    
    try:
        conditional_images = [Image.open(image).convert("RGB") for image in images]
        conditional_images = [torch.from_numpy(np.copy(np.array(image))) for image in conditional_images]
        conditional_images = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in conditional_images]
        conditional_images = [transform(image).to(device=device, dtype=dtype) for image in conditional_images]
    except Exception as e:
        print('Error when loading images')
        print(f'condition images are {images}')
        raise e
    return dict(conditional_images=conditional_images, conditional_images_indices=conditional_images_indices)

def parquet(data_path):
    data = []

    with jsonlines.open(data_path, 'r') as reader:
        for obj in tqdm(reader):
            data.append(obj)

    df = pd.DataFrame(data, columns=['path', 'num_frames', 'height', 'width', 'cap'])
    save_path = data_path.replace('.jsonl', '.parquet')
    df.to_parquet(save_path)

if __name__ == "__main__":
    jsonl_path = "YOUR_JSONL_PATH"
    parquet()