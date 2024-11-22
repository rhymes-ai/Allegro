import torch
import imageio
import os
import argparse
from PIL import Image
from einops import rearrange
import numpy as np
from torchvision.transforms import Lambda
from torchvision import transforms

from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer

from allegro.pipelines.pipeline_allegro_ti2v import AllegroTI2VPipeline
from allegro.pipelines.data_process import ToTensorVideo, CenterCropResizeVideo
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro_ti2v import AllegroTransformerTI2V3DModel


def preprocess_images(first_frame, last_frame, height, width, device, dtype):
    norm_fun = Lambda(lambda x: 2. * x - 1.)
    transform = transforms.Compose([
        ToTensorVideo(),
        CenterCropResizeVideo((height, width)),
        norm_fun
    ])
    images = []
    if first_frame is not None and len(first_frame.strip()) != 0: 
        print("first_frame:", first_frame)
        images.append(first_frame)
    else:
        print("ERROR: First frame must be provided in the Allegro-TI2V!")
        raise NotImplementedError
    if last_frame is not None and len(last_frame.strip()) != 0: 
        print("last_frame:", last_frame)
        images.append(last_frame)

    if len(images) == 1:    # first frame as condition
        print("Video generation with given first frame.")
        conditional_images_indices = [0]
    elif len(images) == 2:  # first&last frames as condition
        print("Video generation with given first and last frame.")
        conditional_images_indices = [0, -1]
    else:
        print("ERROR: Only support 1 or 2 conditional images!")
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


def single_inference(args):
    dtype=torch.bfloat16

    # vae have better formance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32).cuda()

    vae.eval()

    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder, 
        torch_dtype=dtype
    )
    text_encoder.eval()

    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer,
    )

    scheduler = EulerAncestralDiscreteScheduler()

    transformer = AllegroTransformerTI2V3DModel.from_pretrained(
        args.dit,
        torch_dtype=dtype
    ).cuda()
    transformer.eval()

    allegro_ti2v_pipeline = AllegroTI2VPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer
    ).to("cuda:0")


    positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

    negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

    user_prompt = positive_prompt.format(args.user_prompt.lower().strip())
    pre_results = preprocess_images(args.first_frame, args.last_frame, height=720, width=1280, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    cond_imgs = pre_results['conditional_images']
    cond_imgs_indices = pre_results['conditional_images_indices']

    if args.enable_cpu_offload:
        allegro_ti2v_pipeline.enable_sequential_cpu_offload()
        print("cpu offload enabled")
        
    out_video = allegro_ti2v_pipeline(
        user_prompt, 
        negative_prompt=negative_prompt,
        conditional_images=cond_imgs,
        conditional_images_indices=cond_imgs_indices,
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=512,
        generator=torch.Generator(device="cuda:0").manual_seed(args.seed),
    ).video[0]

    imageio.mimwrite(args.save_path, out_video, fps=15, quality=6)  # highest quality is 10, lowest is 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--user_prompt", type=str, default='Two skateboarders ride in a concrete skatepark. One, dressed in a red shirt and shorts, is in motion on the left. The other, wearing a light-colored sweatshirt and shorts, balances on the right. They glide along curved ramps, maneuvering skillfully within the open-air setting.')
    # parser.add_argument('--images', nargs='+', default=['/cpfs/data/user/zhouyuan/eval/ti2v_test_202409/test_005.jpg',])
    parser.add_argument("--user_prompt", type=str, default='')
    parser.add_argument('--first_frame', type=str, default='', help='A single image file as the first frame.')
    parser.add_argument('--last_frame', type=str, default='', help='A single image file as the last frame.')
    parser.add_argument("--vae", type=str, default='')
    parser.add_argument("--dit", type=str, default='')
    parser.add_argument("--text_encoder", type=str, default='')
    parser.add_argument("--tokenizer", type=str, default='')
    parser.add_argument("--save_path", type=str, default="./output_videos/test_video.mp4")
    parser.add_argument("--guidance_scale", type=float, default=8)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_cpu_offload", action='store_true')

    args = parser.parse_args()

    if os.path.dirname(args.save_path) != '' and (not os.path.exists(os.path.dirname(args.save_path))):
        os.makedirs(os.path.dirname(args.save_path))

    
    single_inference(args)
