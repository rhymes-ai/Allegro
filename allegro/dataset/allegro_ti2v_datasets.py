import os
import random
import numpy as np
from decord import VideoReader

import torch
from accelerate.logging import get_logger

from allegro.utils.utils import text_preprocessing
from allegro.dataset.allegro_datasets import Allegro_dataset, random_sample_rate

logger = get_logger(__name__)


class AllegroTI2V_dataset(Allegro_dataset):
    def __init__(self, args, transform, temporal_sample, tokenizer,):
        super().__init__(args, transform, temporal_sample, tokenizer,)

        if self.num_frames != 1:
            self.i2v_ratio = args.i2v_ratio
            self.interp_ratio = args.interp_ratio
            self.v2v_ratio = args.v2v_ratio
            self.clear_video_ratio = args.clear_video_ratio
            assert self.i2v_ratio + self.interp_ratio + self.v2v_ratio + self.clear_video_ratio <= 1, 'The sum of i2v_ratio, interp_ratio, v2v_ratio and clear video ratio should be less than 1.'
        
        self.default_text_ratio = args.default_text_ratio
        self.default_text = f"The {'video' if self.num_frames != 1 else 'image'} showcases a scene with coherent and clear visuals."

    def get_mask_masked_video(self, video):
        # video shape (T, C, H, W)
        # 1 means masked, 0 means not masked
        t, c, h, w = video.shape
        mask = torch.ones_like(video, device=video.device, dtype=video.dtype)
        rand_num = random.random()

        # i2v
        if rand_num <= self.i2v_ratio:
            mask[0] = 0
        # interpolation
        elif rand_num < self.i2v_ratio + self.interp_ratio:
            mask[0] = 0
            mask[-1] = 0
        # continuation
        elif rand_num < self.i2v_ratio + self.interp_ratio + self.v2v_ratio:
            end_idx = random.randint(1, t)
            mask[:end_idx] = 0
        # clear video
        elif rand_num < self.i2v_ratio + self.interp_ratio + self.v2v_ratio + self.clear_video_ratio:
            mask[:] = 0
        # random mask
        else:
            idx_to_select = random.randint(0, t - 1)
            selected_indices = random.sample(range(0, t), idx_to_select)
            mask[selected_indices] = 0

        return dict(mask=mask)

    def drop(self, text):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            text = self.default_text if rand_num_text < self.default_text_ratio else ''

        return dict(text=text)
    
    def read_video_decord(self, data) -> torch.Tensor:
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
        return video

    def get_video(self, data):
        # video = self.read_video_torchvision(data)
        video = self.read_video_decord(data)
        video = self.transform(video)
        # ti2v
        cond = self.get_mask_masked_video(video)
        mask = cond['mask']
        video = torch.cat([video, mask], dim=1) # T 2*C H W
        video = video.transpose(0, 1)  # T C H W -> C T H W

        text = text_preprocessing(data['cap'])
        text = self.drop(text)['text']

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