import math
import os
from typing import Optional, Tuple, Union
from einops import rearrange

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.attention_processor import SpatialNorm

from allegro.models.vae.modules import DownEncoderBlock3D, UNetMidBlock3DConv, UpDecoderBlock3D


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_blocks=4,
        blocks_temp_li=[False, False, False, False],
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.blocks_temp_li = blocks_temp_li

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temp_conv_in = nn.Conv3d(
            block_out_channels[0],
            block_out_channels[0],
            (3,1,1),
            padding = (1, 0, 0)
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i in range(num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                add_temp_downsample=blocks_temp_li[i],
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DConv(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels

        self.temp_conv_out = nn.Conv3d(block_out_channels[-1], block_out_channels[-1], (3,1,1), padding = (1, 0, 0))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        nn.init.zeros_(self.temp_conv_in.weight)
        nn.init.zeros_(self.temp_conv_in.bias)
        nn.init.zeros_(self.temp_conv_out.weight)
        nn.init.zeros_(self.temp_conv_out.bias)

        self.gradient_checkpointing = False

    def forward(self, x):
        '''
            x: [b, c, (tb f), h, w]
        '''
        bz = x.shape[0]
        sample = rearrange(x, 'b c n h w -> (b n) c h w')
        sample = self.conv_in(sample)

        sample = rearrange(sample, '(b n) c h w -> b c n h w', b=bz)
        temp_sample = sample
        sample = self.temp_conv_in(sample) 
        sample = sample+temp_sample
        # down
        for b_id, down_block in enumerate(self.down_blocks):
            sample = down_block(sample)
        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = rearrange(sample, 'b c n h w -> (b n) c h w')
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = rearrange(sample, '(b n) c h w -> b c n h w', b=bz)

        temp_sample = sample
        sample = self.temp_conv_out(sample) 
        sample = sample+temp_sample
        sample = rearrange(sample, 'b c n h w -> (b n) c h w')

        sample = self.conv_out(sample)
        sample = rearrange(sample, '(b n) c h w -> b c n h w', b=bz)
        return sample
    
class Decoder3D(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        num_blocks=4,
        blocks_temp_li=[False, False, False, False],
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.blocks_temp_li = blocks_temp_li

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.temp_conv_in = nn.Conv3d(
            block_out_channels[-1],
            block_out_channels[-1],
            (3,1,1),
            padding = (1, 0, 0)
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3DConv(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(num_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                add_temp_upsample=blocks_temp_li[i],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        self.temp_conv_out = nn.Conv3d(block_out_channels[0], block_out_channels[0], (3,1,1), padding = (1, 0, 0))
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        nn.init.zeros_(self.temp_conv_in.weight)
        nn.init.zeros_(self.temp_conv_in.bias)
        nn.init.zeros_(self.temp_conv_out.weight)
        nn.init.zeros_(self.temp_conv_out.bias)

        self.gradient_checkpointing = False

    def forward(self, z):
        bz = z.shape[0]
        sample = rearrange(z, 'b c n h w -> (b n) c h w')
        sample = self.conv_in(sample)

        sample = rearrange(sample, '(b n) c h w -> b c n h w', b=bz)
        temp_sample = sample
        sample = self.temp_conv_in(sample) 
        sample = sample+temp_sample

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample)
        sample = sample.to(upscale_dtype)

        # up
        for b_id, up_block in enumerate(self.up_blocks):
            sample = up_block(sample)

        # post-process
        sample = rearrange(sample, 'b c n h w -> (b n) c h w')
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        sample = rearrange(sample, '(b n) c h w -> b c n h w', b=bz)
        temp_sample = sample
        sample = self.temp_conv_out(sample)
        sample = sample+temp_sample
        sample = rearrange(sample, 'b c n h w -> (b n) c h w')

        sample = self.conv_out(sample)
        sample = rearrange(sample, '(b n) c h w -> b c n h w', b=bz)
        return sample
    


class AllegroAutoencoderKL3D(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `256`): Spatial Tiling Size.
        tile_overlap (`tuple`, *optional*, defaults to `(120, 80`): Spatial overlapping size while tiling (height, width)
        chunk_len (`int`, *optional*, defaults to `24`): Temporal Tiling Size.
        t_over (`int`, *optional*, defaults to `8`): Temporal overlapping size while tiling
        scaling_factor (`float`, *optional*, defaults to 0.13235):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        blocks_tempdown_li (`List`, *optional*, defaults to `[True, True, False, False]`): Each item indicates whether each TemporalBlock in the Encoder performs temporal downsampling.
        blocks_tempup_li (`List`, *optional*, defaults to `[False, True, True, False]`): Each item indicates whether each TemporalBlock in the Decoder performs temporal upsampling.
        load_mode (`str`, *optional*, defaults to `full`): Load mode for the model. Can be one of `full`, `encoder_only`, `decoder_only`. which corresponds to loading the full model state dicts, only the encoder state dicts, or only the decoder state dicts.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_num: int = 4,
        up_block_num: int = 4,
        block_out_channels: Tuple[int] = (128,256,512,512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 320,
        tile_overlap: tuple = (120, 80),
        force_upcast: bool = True,
        chunk_len: int = 24,
        t_over: int = 8,
        scale_factor: float = 0.13235,
        blocks_tempdown_li=[True, True, False, False],
        blocks_tempup_li=[False, True, True, False],
        load_mode = 'full',
    ):
        super().__init__()

        self.blocks_tempdown_li = blocks_tempdown_li
        self.blocks_tempup_li = blocks_tempup_li
        # pass init params to Encoder
        self.load_mode = load_mode
        if load_mode in ['full', 'encoder_only']:
            self.encoder = Encoder3D(
                in_channels=in_channels,
                out_channels=latent_channels,
                num_blocks=down_block_num,
                blocks_temp_li=blocks_tempdown_li,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                double_z=True,
            )
            self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

        if load_mode in ['full', 'decoder_only']:
            # pass init params to Decoder
            self.decoder = Decoder3D(
                in_channels=latent_channels,
                out_channels=out_channels,
                num_blocks=up_block_num,
                blocks_temp_li=blocks_tempup_li,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                norm_num_groups=norm_num_groups,
                act_fn=act_fn,
            )
            self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)


        # only relevant if vae tiling is enabled
        sample_size = (
            sample_size[0]
            if isinstance(sample_size, (list, tuple))
            else sample_size
        )
        self.tile_overlap = tile_overlap
        self.vae_scale_factor = [4, 8, 8]
        self.scale_factor = scale_factor
        self.sample_size = sample_size
        self.chunk_len = chunk_len
        self.t_over = t_over

        self.latent_chunk_len = self.chunk_len//4
        self.latent_t_over = self.t_over//4 
        self.kernel = (self.chunk_len, self.sample_size, self.sample_size) #(24, 256, 256)
        self.stride = (self.chunk_len - self.t_over, self.sample_size-self.tile_overlap[0], self.sample_size-self.tile_overlap[1])  # (16, 112, 192)


    def encode(self, input_imgs: torch.Tensor, return_dict: bool = True, local_batch_size=1) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        KERNEL = self.kernel
        STRIDE = self.stride
        LOCAL_BS = local_batch_size
        OUT_C = 8

        B, C, N, H, W = input_imgs.shape
        
       
        out_n = math.floor((N - KERNEL[0]) / STRIDE[0]) + 1
        out_h = math.floor((H - KERNEL[1]) / STRIDE[1]) + 1
        out_w = math.floor((W - KERNEL[2]) / STRIDE[2]) + 1
        
        ## cut video into overlapped small cubes and batch forward
        num = 0

        out_latent = torch.zeros((out_n*out_h*out_w, OUT_C, KERNEL[0]//4, KERNEL[1]//8, KERNEL[2]//8), device=input_imgs.device, dtype=input_imgs.dtype) 
        vae_batch_input = torch.zeros((LOCAL_BS, C, KERNEL[0], KERNEL[1], KERNEL[2]), device=input_imgs.device, dtype=input_imgs.dtype)

        for i in range(out_n):
            for j in range(out_h):
                for k in range(out_w):
                    n_start, n_end = i * STRIDE[0], i * STRIDE[0] + KERNEL[0]
                    h_start, h_end = j * STRIDE[1], j * STRIDE[1] + KERNEL[1]
                    w_start, w_end = k * STRIDE[2], k * STRIDE[2] + KERNEL[2]
                    video_cube = input_imgs[:, :, n_start:n_end, h_start:h_end, w_start:w_end]
                    vae_batch_input[num%LOCAL_BS] = video_cube
                    
                    if num%LOCAL_BS == LOCAL_BS-1 or num == out_n*out_h*out_w-1:                        
                        latent = self.encoder(vae_batch_input)
                        
                        if num == out_n*out_h*out_w-1 and num%LOCAL_BS != LOCAL_BS-1:
                            out_latent[num-num%LOCAL_BS:] = latent[:num%LOCAL_BS+1]
                        else:
                            out_latent[num-LOCAL_BS+1:num+1] = latent
                        vae_batch_input = torch.zeros((LOCAL_BS, C, KERNEL[0], KERNEL[1], KERNEL[2]), device=input_imgs.device, dtype=input_imgs.dtype)
                    num+=1
        
        ## flatten the batched out latent to videos and supress the overlapped parts
        B, C, N, H, W = input_imgs.shape

        out_video_cube = torch.zeros((B, OUT_C, N//4, H//8, W//8), device=input_imgs.device, dtype=input_imgs.dtype)
        OUT_KERNEL = KERNEL[0]//4, KERNEL[1]//8, KERNEL[2]//8
        OUT_STRIDE = STRIDE[0]//4, STRIDE[1]//8, STRIDE[2]//8
        OVERLAP = OUT_KERNEL[0]-OUT_STRIDE[0], OUT_KERNEL[1]-OUT_STRIDE[1], OUT_KERNEL[2]-OUT_STRIDE[2]
        
        for i in range(out_n):
            n_start, n_end = i * OUT_STRIDE[0], i * OUT_STRIDE[0] + OUT_KERNEL[0]
            for j in range(out_h):
                h_start, h_end = j * OUT_STRIDE[1], j * OUT_STRIDE[1] + OUT_KERNEL[1]
                for k in range(out_w):
                    w_start, w_end = k * OUT_STRIDE[2], k * OUT_STRIDE[2] + OUT_KERNEL[2]
                    latent_mean_blend = prepare_for_blend((i, out_n, OVERLAP[0]), (j, out_h, OVERLAP[1]), (k, out_w, OVERLAP[2]), out_latent[i*out_h*out_w+j*out_w+k].unsqueeze(0))
                    out_video_cube[:, :, n_start:n_end, h_start:h_end, w_start:w_end] += latent_mean_blend
        
        ## final conv
        out_video_cube = rearrange(out_video_cube, 'b c n h w -> (b n) c h w')
        out_video_cube = self.quant_conv(out_video_cube)
        out_video_cube = rearrange(out_video_cube, '(b n) c h w -> b c n h w', b=B)

        posterior = DiagonalGaussianDistribution(out_video_cube)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)
    

    def decode(self, input_latents: torch.Tensor, return_dict: bool = True, local_batch_size=1) -> Union[DecoderOutput, torch.Tensor]:
        KERNEL = self.kernel
        STRIDE = self.stride
        
        LOCAL_BS = local_batch_size
        OUT_C = 3
        IN_KERNEL = KERNEL[0]//4, KERNEL[1]//8, KERNEL[2]//8
        IN_STRIDE = STRIDE[0]//4, STRIDE[1]//8, STRIDE[2]//8

        B, C, N, H, W = input_latents.shape

        ## post quant conv (a mapping)
        input_latents = rearrange(input_latents, 'b c n h w -> (b n) c h w')
        input_latents = self.post_quant_conv(input_latents)
        input_latents = rearrange(input_latents, '(b n) c h w -> b c n h w', b=B)
        
        ## out tensor shape
        out_n = math.floor((N - IN_KERNEL[0]) / IN_STRIDE[0]) + 1
        out_h = math.floor((H - IN_KERNEL[1]) / IN_STRIDE[1]) + 1
        out_w = math.floor((W - IN_KERNEL[2]) / IN_STRIDE[2]) + 1

        ## cut latent into overlapped small cubes and batch forward
        num = 0
        decoded_cube = torch.zeros((out_n*out_h*out_w, OUT_C, KERNEL[0], KERNEL[1], KERNEL[2]), device=input_latents.device, dtype=input_latents.dtype) 
        vae_batch_input = torch.zeros((LOCAL_BS, C, IN_KERNEL[0], IN_KERNEL[1], IN_KERNEL[2]), device=input_latents.device, dtype=input_latents.dtype)
        for i in range(out_n):
            for j in range(out_h):
                for k in range(out_w):
                    n_start, n_end = i * IN_STRIDE[0], i * IN_STRIDE[0] + IN_KERNEL[0]
                    h_start, h_end = j * IN_STRIDE[1], j * IN_STRIDE[1] + IN_KERNEL[1]
                    w_start, w_end = k * IN_STRIDE[2], k * IN_STRIDE[2] + IN_KERNEL[2]
                    latent_cube = input_latents[:, :, n_start:n_end, h_start:h_end, w_start:w_end]
                    vae_batch_input[num%LOCAL_BS] = latent_cube
                    if num%LOCAL_BS == LOCAL_BS-1 or num == out_n*out_h*out_w-1:
                        
                        latent = self.decoder(vae_batch_input)
                        
                        if num == out_n*out_h*out_w-1 and num%LOCAL_BS != LOCAL_BS-1:
                            decoded_cube[num-num%LOCAL_BS:] = latent[:num%LOCAL_BS+1]
                        else:
                            decoded_cube[num-LOCAL_BS+1:num+1] = latent
                        vae_batch_input = torch.zeros((LOCAL_BS, C, IN_KERNEL[0], IN_KERNEL[1], IN_KERNEL[2]), device=input_latents.device, dtype=input_latents.dtype)
                    num+=1
        B, C, N, H, W = input_latents.shape
        
        out_video = torch.zeros((B, OUT_C, N*4, H*8, W*8), device=input_latents.device, dtype=input_latents.dtype)
        OVERLAP = KERNEL[0]-STRIDE[0], KERNEL[1]-STRIDE[1], KERNEL[2]-STRIDE[2]
        for i in range(out_n):
            n_start, n_end = i * STRIDE[0], i * STRIDE[0] + KERNEL[0]
            for j in range(out_h):
                h_start, h_end = j * STRIDE[1], j * STRIDE[1] + KERNEL[1]
                for k in range(out_w):
                    w_start, w_end = k * STRIDE[2], k * STRIDE[2] + KERNEL[2]
                    out_video_blend = prepare_for_blend((i, out_n, OVERLAP[0]), (j, out_h, OVERLAP[1]), (k, out_w, OVERLAP[2]), decoded_cube[i*out_h*out_w+j*out_w+k].unsqueeze(0))
                    out_video[:, :, n_start:n_end, h_start:h_end, w_start:w_end] += out_video_blend
       
        out_video = rearrange(out_video, 'b c t h w -> b t c h w').contiguous()

        decoded = out_video
        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        encoder_local_batch_size: int = 2,
        decoder_local_batch_size: int = 2,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*): 
                PyTorch random number generator.
            encoder_local_batch_size (`int`, *optional*, defaults to 2):
                Local batch size for the encoder's batch inference.
            decoder_local_batch_size (`int`, *optional*, defaults to 2):
                Local batch size for the decoder's batch inference.
        """
        x = sample
        posterior = self.encode(x, local_batch_size=encoder_local_batch_size).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, local_batch_size=decoder_local_batch_size).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        kwargs["torch_type"] = torch.float32
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


def prepare_for_blend(n_param, h_param, w_param, x):
    n, n_max, overlap_n = n_param
    h, h_max, overlap_h = h_param
    w, w_max, overlap_w = w_param
    if overlap_n > 0:
        if n > 0: # the head overlap part decays from 0 to 1
            x[:,:,0:overlap_n,:,:] = x[:,:,0:overlap_n,:,:] * (torch.arange(0, overlap_n).float().to(x.device) / overlap_n).reshape(overlap_n,1,1)
        if n < n_max-1:  # the tail overlap part decays from 1 to 0
            x[:,:,-overlap_n:,:,:] = x[:,:,-overlap_n:,:,:] * (1 - torch.arange(0, overlap_n).float().to(x.device) / overlap_n).reshape(overlap_n,1,1)
    if h > 0:
        x[:,:,:,0:overlap_h,:] = x[:,:,:,0:overlap_h,:] * (torch.arange(0, overlap_h).float().to(x.device) / overlap_h).reshape(overlap_h,1)
    if h < h_max-1:
        x[:,:,:,-overlap_h:,:] = x[:,:,:,-overlap_h:,:] * (1 - torch.arange(0, overlap_h).float().to(x.device) / overlap_h).reshape(overlap_h,1)
    if w > 0:
        x[:,:,:,:,0:overlap_w] = x[:,:,:,:,0:overlap_w] * (torch.arange(0, overlap_w).float().to(x.device) / overlap_w)
    if w < w_max-1:
        x[:,:,:,:,-overlap_w:] = x[:,:,:,:,-overlap_w:] * (1 - torch.arange(0, overlap_w).float().to(x.device) / overlap_w)
    return x