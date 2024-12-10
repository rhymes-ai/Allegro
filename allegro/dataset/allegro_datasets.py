import os
import pickle
import random
import numpy as np
import pandas as pd
from einops import rearrange
from collections import OrderedDict
import hashlib
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from accelerate.logging import get_logger

from allegro.utils.utils import text_preprocessing, lprint

logger = get_logger(__name__)

def filter_resolution(height, width, max_height, max_width, hw_thr, hw_aspect_thr):
    aspect = max_height / max_width
    if height >= max_height * hw_thr and width >= max_width * hw_thr and height / width >= aspect / hw_aspect_thr and height / width <= aspect * hw_aspect_thr:
        return True
    return False

def filter_duration(num_frames, sample_frames, sample_rate):
    target_frames = (sample_frames - 1) * sample_rate + 1
    if num_frames >= target_frames:
        return True
    return False

def random_sample_rate(num_frames, sample_frames, sample_rate):
    supported_sample_rate = []
    for sr in sample_rate:
        if filter_duration(num_frames, sample_frames, sr):
            supported_sample_rate.append(sr)
    sr = None
    if len(supported_sample_rate) > 0:
        sr = random.choice(supported_sample_rate)
    return sr

class Allegro_dataset(Dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer):
        self.data_dir = args.data_dir
        self.meta_file = args.meta_file
        self.num_frames = args.num_frames
        self.sample_rate = sorted(list(map(int, args.sample_rate.split(','))))
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.model_max_length = args.model_max_length
        self.cfg = args.cfg
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.hw_thr = args.hw_thr
        self.hw_aspect_thr = args.hw_aspect_thr
        self.cache_dir = args.cache_dir

        self.filter_data_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            data = self.data_list.loc[idx]
            if data['path'].endswith('.mp4'):
                return self.get_video(data)
            else:
                return self.get_image(data)
        except Exception as e:
            logger.info(f"Error with {e}, file {data['path']}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))
    
    def get_video(self, data):
        vr = VideoReader(os.path.join(self.data_dir, data['path']))
        sr = random_sample_rate(len(vr), self.num_frames, self.sample_rate)
        if sr is None:
            raise ValueError(f'no supported sr for num_frames ({len(vr)}), sample_frames ({self.num_frames}), sample_rate ({self.sample_rate})')
        fidx = np.arange(0, len(vr), sr).astype(int)
        sidx, eidx = self.temporal_sample(len(fidx))
        fidx = fidx[sidx: eidx]
        if self.num_frames != len(fidx):
            raise ValueError(f'num_frames ({self.num_frames}) is not equal with frame_indices ({len(fidx)})')
        video = vr.get_batch(fidx).asnumpy()
        video = torch.from_numpy(video)
        video = video.permute(0, 3, 1, 2)
        video = self.transform(video)
        video = video.transpose(0, 1)

        text = text_preprocessing(data['cap']) if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']

        return dict(pixel_values=video, input_ids=input_ids, cond_mask=cond_mask)

    def get_image(self, data):
        image = Image.open(os.path.join(self.data_dir, data['path'])).convert('RGB')
        image = torch.from_numpy(np.array(image))
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)
        image = self.transform(image)
        image = image.transpose(0, 1)

        text = text_preprocessing(data['cap']) if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']

        image.close()
        return dict(pixel_values=image, input_ids=input_ids, cond_mask=cond_mask)

    def filter_data_list(self):
        lprint(f'Filter data {self.meta_file}')
        cache_path = self.check_cache()
        if os.path.exists(cache_path):
            lprint(f'Load cache {cache_path}')
            with open(cache_path, 'rb') as f:
                self.data_list = pickle.load(f)
            lprint(f'Data length: {len(self.data_list)}')
            return
        
        self.data_list = pd.read_parquet(self.meta_file)
        pick_list = []
        for i in range(len(self.data_list)):
            data = self.data_list.loc[i]
            is_pick = filter_resolution(data['height'], data['width'], self.max_height, self.max_width, self.hw_thr, self.hw_aspect_thr)
            if data['path'].endswith('.mp4'):
                is_pick = is_pick and filter_duration(data['num_frames'], self.num_frames, self.sample_rate[0])
            pick_list.append(is_pick)
            if i % 1000000 == 0:
                lprint(f'Filter {i}')
        self.data_list = self.data_list.loc[pick_list]
        self.data_list = self.data_list.reset_index(drop=True)
        lprint(f'Data length: {len(self.data_list)}')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.data_list, f)
            lprint(f'Save cache {cache_path}')

    def check_cache(self):
        unique_identifiers = OrderedDict()
        unique_identifiers['class'] = type(self).__name__
        unique_identifiers['data_dir'] = self.data_dir
        unique_identifiers['meta_file'] = self.meta_file
        unique_identifiers['num_frames'] = self.num_frames
        unique_identifiers['sample_rate'] = self.sample_rate[0]
        unique_identifiers['hw_thr'] = self.hw_thr
        unique_identifiers['hw_aspect_thr'] = self.hw_aspect_thr
        unique_identifiers['max_height'] = self.max_height
        unique_identifiers['max_width'] = self.max_width
        unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        unique_description_hash = hashlib.md5(unique_description.encode('utf-8')).hexdigest()
        path_to_cache = os.path.join(self.cache_dir, 'data_cache')
        os.makedirs(path_to_cache, exist_ok=True)
        cache_path = os.path.join(
                path_to_cache, f'{unique_description_hash}-{type(self).__name__}-filter_cache.pkl'
        )
        return cache_path
