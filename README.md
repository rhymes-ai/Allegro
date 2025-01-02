<p align="center">
<img src="https://github.com/rhymes-ai/Allegro/blob/main/assets/TI2V_banner.gif"/>
</p>

<p align="center">
 <a href="https://rhymes.ai/allegro_gallery" target="_blank"> Gallery</a> · <a href="https://huggingface.co/rhymes-ai/Allegro" target="_blank">Hugging Face</a> · <a href="https://rhymes.ai/blog-details/allegro-advanced-video-generation-model" target="_blank">Blog</a> · <a href="https://arxiv.org/abs/2410.15458" target="_blank">Paper</a> · <a href="https://discord.com/invite/u8HxU23myj" target="_blank">Discord</a> 
</p> 

[Allegro](huggingface.co/rhymes-ai/Allegro) is a powerful text-to-video model that generates high-quality videos up to 6 seconds at 15 FPS and 720p resolution from simple text input. [Allegro-TI2V](https://huggingface.co/rhymes-ai/Allegro-TI2V), a variant of [Allegro](https://huggingface.co/rhymes-ai/Allegro), extends this functionality by generating similar high-quality videos using text inputs along with first-frame and optionally last-frame image inputs.

## News 🔥

 - [2024/12/10] 🚀 We release the training code for further training / fine-tuning!

 - [2024/11/25] 🚀 [Allegro-TI2V](https://huggingface.co/rhymes-ai/Allegro-TI2V) is open sourced! 

 - [2024/10/30] 🚀 We release multi-card inference code and PAB in [Allegro-VideoSys](https://github.com/nightsnack/Allegro-VideoSys). With VideoSys framework, the inference time can be further reduced to 3 mins (8xH100) and 2 mins (8xH100+PAB). We also opened a PR to the original [VideoSys repo](https://github.com/NUS-HPC-AI-Lab/VideoSys).

 - [2024/10/29] 🎉 Congratulations that Allegro is merged into diffusers! Currently Allegro is supported in `0.32.0-dev0.` It will be integrated in the next release version. So for now, please use `pip install git+https://github.com/huggingface/diffusers.git` to install diffuser dev version. See [huggingface](https://huggingface.co/rhymes-ai/Allegro) for more details.

 - [2024/10/22]🚀 [Allegro](huggingface.co/rhymes-ai/Allegro) is open sourced! 




## Model Info
<table>
  <tr>
    <th>Model</th>
    <td>Allegro-TI2V</td>
    <td>Allegro</td>
  </tr>
  <tr>
    <th>Description</th>
    <td>Text-Image-to-Video Generation Model</td>
    <td>Text-to-Video Generation Model</td>
  </tr>
 <tr>
    <th>Download</th>
    <td><a href="https://huggingface.co/rhymes-ai/Allegro-TI2V">Hugging Face</a></td>
    <td><a href="https://huggingface.co/rhymes-ai/Allegro">Hugging Face</a></td>
</tr>
  <tr>
    <th rowspan="2">Parameter</th>
    <td colspan="2">VAE: 175M</td>
  </tr>
  <tr>
    <td colspan="2">DiT: 2.8B</td>
  </tr>
  <tr>
    <th rowspan="2">Inference Precision</th>
    <td colspan="2">VAE: FP32/TF32/BF16/FP16 (best in FP32/TF32)</td>
  </tr>
  <tr>
    <td colspan="2">DiT/T5: BF16/FP32/TF32</td>
  </tr>
  <tr>
    <th>Context Length</th>
    <td colspan="2">79.2K</td>
  </tr>
  <tr>
    <th>Resolution</th>
    <td colspan="2">720 x 1280</td>
  </tr>
  <tr>
    <th>Frames</th>
    <td colspan="2">88</td>
  </tr>
  <tr>
    <th>Video Length</th>
    <td colspan="2">6 seconds @ 15 FPS</td>
  </tr>
  <tr>
    <th>Single GPU Memory Usage</th>
    <td colspan="2">9.3G BF16 (with cpu_offload)</td>
  </tr>
    <tr>
    <th>Inference time</th>
    <td colspan="2">20 mins (single H100) / 3 mins (8xH100)</td>
  </tr>
</table>

## Quick Start
### Single Inference
#### Allegro
1. Download the [Allegro GitHub code](https://github.com/rhymes-ai/Allegro).
   
2. Install the necessary requirements.
   
   - Ensure Python >= 3.10, PyTorch >= 2.4, CUDA >= 12.4. For details, see [requirements.txt](https://github.com/rhymes-ai/Allegro/blob/main/requirements.txt).  
    
   - It is recommended to use Anaconda to create a new environment (Python >= 3.10) to run the following example.  
   
3. Download the [Allegro model weights](https://huggingface.co/rhymes-ai/Allegro).
   
4. Run inference.
   
    ```python
    python single_inference.py \
    --user_prompt 'A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats.' \
    --save_path ./output_videos/test_video.mp4 \
    --vae your/path/to/vae \
    --dit your/path/to/transformer \
    --text_encoder your/path/to/text_encoder \
    --tokenizer your/path/to/tokenizer \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --seed 42
    ```

    Use `--enable_cpu_offload` to offload the model into CPU for less GPU memory cost (about 9.3G, compared to 27.5G if CPU offload is not enabled), but the inference time will increase significantly.


5. (Optional) Interpolate the video to 30 FPS.

    It is recommended to use [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) to interpolate the video from 15 FPS to 30 FPS.
  
    For better visual quality, please use imageio to save the video.
   
#### Allegro TI2V
1. Download the [Allegro GitHub code](https://github.com/rhymes-ai/Allegro).
   
2. Install the necessary requirements.
   
   - Ensure Python >= 3.10, PyTorch >= 2.4, CUDA >= 12.4. For details, see [requirements.txt](https://github.com/rhymes-ai/Allegro/blob/main/requirements.txt).  
    
   - It is recommended to use Anaconda to create a new environment (Python >= 3.10) to run the following example.  
   
3. Download the [Allegro-TI2V model weights](https://huggingface.co/rhymes-ai/Allegro-TI2V).
   
4. Run inference.
   
    ```python
    python single_inference_ti2v.py \
    --user_prompt 'The car drives along the road' \
    --first_frame your/path/to/first_frame_image.png \
    --vae your/path/to/vae \
    --dit your/path/to/transformer \
    --text_encoder your/path/to/text_encoder \
    --tokenizer your/path/to/tokenizer \
    --guidance_scale 8 \
    --num_sampling_steps 100 \
    --seed 1427329220
    ```
  
    The output video resolution is fixed at 720 × 1280. Input images with different resolutions will be automatically cropped and resized to fit.

| Argument         | Description                                                                                                                              |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `--user_prompt`       | [Required] Text input for image-to-video generation.                                                                                          |
| `--first_frame`       | [Required] First-frame image input for image-to-video generation.                                                                             |
| `--last_frame`        | [Optional] If provided, the model will generate intermediate video content based on the specified first and last frame images.                |
| <span style="background-color: #f6f8fa;">`--enable_cpu_offload`</span> | <span style="background-color: #f6f8fa;">[Optional] Offload the model into CPU for less GPU memory cost (about 9.3G, compared to 27.5G if CPU offload is not enabled), but the inference time will increase significantly.</span> |


5. (Optional) Interpolate the video to 30 FPS.

    It is recommended to use [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) to interpolate the video from 15 FPS to 30 FPS.
  
    For better visual quality, please use imageio to save the video.

### Multi-Card Inference
For both Allegro & Allegro TI2V: We release multi-card inference code and PAB in [Allegro-VideoSys](https://github.com/nightsnack/Allegro-VideoSys). 

### Training / Fine-tuning

1. Download the [Allegro GitHub code](https://github.com/rhymes-ai/Allegro), [Allegro model weights](https://huggingface.co/rhymes-ai/Allegro) and prepare the environment in [requirements.txt](https://github.com/rhymes-ai/Allegro/blob/main/requirements.txt).
   
2. Our training code loads the dataset from `.parquet` files. We recommend first constructing a `.jsonl` file to store all data cases in a list. Each case should be stored as a dict, like this:

    ```json
    [
        {"path": "foo/bar.mp4", "num_frames": 123, "height": 1080, "width": 1920, "cap": "This is a fake caption."}
        ...
    ]
    ```
    
    After that, run [dataset_utils.py](https://github.com/rhymes-ai/Allegro/blob/main/allegro/utils/dataset_utils.py) to convert `.jsonl` into `.parquet`.

    > The absolute path to each video is constructed by joining `args.data_dir` in [train.py](https://github.com/rhymes-ai/Allegro/blob/main/train.py) with the `path` value from the dataset. Therefore, you may define `path` as a relative path within your dataset and set `args.data_dir` to the root dir when running training.

3. Run Training / Fine-tuning:

    ```bash
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    export WANDB_API_KEY=YOUR_WANDB_KEY

    accelerate launch \
        --num_machines 1 \
        --num_processes 8 \
        --machine_rank 0 \
        --config_file config/accelerate_config.yaml \
        train.py \
        --project_name Allegro_Finetune_88x720p \
        --dit_config /huggingface/rhymes-ai/Allegro/transformer/config.json \
        --dit /huggingface/rhymes-ai/Allegro/transformer/ \
        --tokenizer /huggingface/rhymes-ai/Allegro/tokenizer \
        --text_encoder /huggingface/rhymes-ai/Allegro/text_encoder \
        --vae /huggingface/rhymes-ai/Allegro/vae \
        --vae_load_mode encoder_only \
        --enable_ae_compile \
        --dataset t2v \
        --data_dir /data_root/ \
        --meta_file data.parquet \
        --sample_rate 2 \
        --num_frames 88 \
        --max_height 720 \
        --max_width 1280 \
        --hw_thr 1.0 \
        --hw_aspect_thr 1.5 \
        --dataloader_num_workers 10 \
        --gradient_checkpointing \
        --train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_train_steps 1000000 \
        --learning_rate 1e-4 \
        --lr_scheduler constant \
        --lr_warmup_steps 0 \
        --mixed_precision bf16 \
        --report_to wandb \
        --allow_tf32 \
        --enable_stable_fp32 \
        --model_max_length 512 \
        --cfg 0.1 \
        --checkpointing_steps 100 \
        --resume_from_checkpoint latest \
        --output_dir ./output/Allegro_Finetune_88x720p
    ```

4. (Optional) To customize the model training arguments, you may create a `.json` file following [config.json](https://huggingface.co/rhymes-ai/Allegro/blob/main/transformer/config.json). Feel free to use our training code to train a video diffusion model from scratch.

## Acceleration with PeRFlow
We also support acceleration with [PeRFlow](https://arxiv.org/abs/2405.07510), a state-of-the-art approach for rectified based diffusion models.

Use the following scripts to train the PeRFlow:

    ```bash
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

    export WANDB_API_KEY=YOUR_WANDB_KEY

    accelerate launch \
        --num_machines 1 \
        --num_processes 8 \
        --machine_rank 0 \
        --config_file config/accelerate_config.yaml \
        train_perflow.py \
        --project_name Allegro_PeRFlow_88x720p \
        --dit_config /huggingface/rhymes-ai/Allegro/transformer/config.json \
        --dit /huggingface/rhymes-ai/Allegro/transformer/ \
        --tokenizer /huggingface/rhymes-ai/Allegro/tokenizer \
        --text_encoder /huggingface/rhymes-ai/Allegro/text_encoder \
        --vae /huggingface/rhymes-ai/Allegro/vae \
        --vae_load_mode encoder_only \
        --enable_ae_compile \
        --dataset t2v \
        --data_dir /data_root/ \
        --meta_file data.parquet \
        --enable_ae_compile \
        --sample_rate 2 \
        --num_frames 88 \
        --max_height 720 \
        --max_width 1280 \
        --hw_thr 1.0 \
        --hw_aspect_thr 1.5 \
        --dataloader_num_workers 10 \
        --gradient_checkpointing \
        --train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_train_steps 1000000 \
        --learning_rate 8e-5 \
        --lr_scheduler constant \
        --lr_warmup_steps 0 \
        --mixed_precision bf16 \
        --report_to wandb \
        --allow_tf32 \
        --enable_stable_fp32 \
        --model_max_length 512 \
        --snr_gamma 5.0 \
        --use_ema \
        --ema_start_step 0 \
        --cfg 0.0 \
        --noise_offset 0.02 \
        --resume_from_checkpoint latest \
        --ema_decay 0.999 \
        --windows 4 \
        --discrete_timesteps -1 \
        --perflow_support_cfg \
        --pred_type diff_eps \
        --perflow_solving_steps 10 \
        --output_dir ./output/Allegro_Finetune_88x720p
    ```


## Limitation
- The model cannot render celebrities, legible text, specific locations, streets or buildings.

## Future Plan
- [x] Multiple GPU inference and further speed up (PAB)
- [x] Text & Image-To-Video (TI2V) video generation
- [ ] Motion-controlled video generation
- [ ] Visual quality enhancement

## Support
If you encounter any problems or have any suggestions, feel free to [open an issue](https://github.com/rhymes-ai/Allegro/issues/new) or send an email to huanyang@rhymes.ai. 

## Citation
Please consider citing our technical report if you find the code and pre-trained models useful for your project.

```
@article{allegro2024,
  title={Allegro: Open the Black Box of Commercial-Level Video Generation Model},
  author={Yuan Zhou and Qiuyue Wang and Yuxuan Cai and Huan Yang},
  journal={arXiv preprint arXiv:2410.15458},
  year={2024}
}
```

## License
This repo is released under the [Apache 2.0 License](https://github.com/rhymes-ai/Allegro/blob/main/LICENSE.txt).

## Disclaimer

The Allegro series models are provided on an "AS IS" basis, and we disclaim any liability for consequences or damages arising from your use. Users are kindly advised to ensure compliance with all applicable laws and regulations. This includes, but is not limited to, prohibitions against illegal activities and the generation of content that is violent, pornographic, obscene, or otherwise deemed non-safe, inappropriate, or illegal. By using these models, you agree that we shall not be held accountable for any consequences resulting from your use.

# Acknowledgment
We extend our heartfelt appreciation for the great contribution to the open-source community, especially Open-Sora-Plan, as we build our diffusion transformer (DiT) based on Open-Sora-Plan v1.2.
- [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan): A project aims to create a simple and scalable repo, to reproduce Sora.
- [Open-Sora](https://github.com/hpcaitech/Open-Sora): An initiative dedicated to efficiently producing high-quality video.
- [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization system.
- [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys): An open-source project that provides a user-friendly and high-performance infrastructure for video generation. 
- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
- [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
- [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
- [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
- [Playground](https://playground.com/blog/playground-v2-5): A state-of-the-art open-source model in text-to-image generation.
- [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI): A video frame interpolation model.
