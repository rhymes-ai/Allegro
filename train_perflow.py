import argparse
import logging
import math
import os
import shutil
import torch
import json
import random

from copy import deepcopy
from pathlib import Path
from packaging import version
from torch.nn import functional as F
from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed

import transformers
from transformers import T5EncoderModel, T5Tokenizer

import diffusers
from diffusers import DDPMScheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr

from allegro.utils.utils import ctime
from allegro.utils.adaptor import replace_with_fp32_forwards
from allegro.utils.dataset_utils import Collate
from allegro.dataset import getdataset
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel
from allegro.models.schedulers.scheduling_perflow import PeRFlowScheduler


logger = get_logger(__name__)

class ProgressInfo:
    def __init__(self, global_step, train_loss=0.0):
        self.global_step = global_step
        self.train_loss = train_loss

def main(args):    
    # ===== logger =≈====
    logging_dir = Path(args.output_dir, args.logging_dir)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.enable_stable_fp32:
        replace_with_fp32_forwards()

    # ===== Accelerator =====
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # ===== weight dtype =====
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    if args.allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    ## 1. Prepare models, noise scheduler, tokenizer.
    # create and freeze vae
    # vae have better performance in float32
    vae = AllegroAutoencoderKL3D.from_pretrained(args.vae, torch_dtype=torch.float32, load_mode=args.vae_load_mode).to(accelerator.device)
    vae.eval()
    vae.requires_grad_(False)
    if args.enable_ae_compile:
        vae.encoder = torch.compile(vae.encoder, mode='max-autotune', fullgraph=True)
    logger.info(f"VAE loaded from {args.vae} successfully")
    if args.vae_load_mode == "encoder_only":
        logger.info("VAE is loaded in encoder_only mode. It's normal that the decoder is not loaded.")
    args.vae_stride_t, args.vae_stride_h, args.vae_stride_w = vae.vae_scale_factor

    # create and freeze text encoder
    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer,
    )
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder, torch_dtype=weight_dtype, low_cpu_mem_usage=True).to(accelerator.device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    logger.info(f"Text encoder loaded from {args.text_encoder} successfully")

    # create teacher dit
    teacher_dit = AllegroTransformer3DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype
    ).to(accelerator.device, dtype=weight_dtype)
    teacher_dit.eval()
    teacher_dit.requires_grad_(False)

    # create teacher scheduler
    teacher_scheduler = DDPMScheduler.from_pretrained(args.scheduler,)
    teacher_num_train_timesteps = teacher_scheduler.config.num_train_timesteps

    from allegro.utils.pfode_solver import PFODESolver
    solver = PFODESolver(scheduler=teacher_scheduler, t_initial=1, t_terminal=0,)

    # create student dit
    student_dit = AllegroTransformer3DModel.from_pretrained(
        args.dit,
        torch_dtype=weight_dtype
    ).to(accelerator.device, dtype=weight_dtype)
    student_dit._set_gradient_checkpointing(value=args.gradient_checkpointing)
    student_dit.train()
    logger.info(f"Student model loaded from {args.dit} successfully")

    # create EMA for the student dit.
    if args.use_ema:
        ema_student_dit = deepcopy(student_dit)
        ema_student_dit = EMAModel(
            ema_student_dit.parameters(),
            decay=args.ema_decay,
            update_after_step=args.ema_start_step,
            model_cls=AllegroTransformer3DModel,
            model_config=ema_student_dit.config
        )
        ema_student_dit.to(accelerator.device)
        logger.info(f"Student EMA model created.")

    # create PeRFlow scheduler
    perflow_scheduler = PeRFlowScheduler(
        num_train_timesteps=teacher_scheduler.config.num_train_timesteps,
        beta_start = teacher_scheduler.config.beta_start,
        beta_end = teacher_scheduler.config.beta_end,
        beta_schedule = teacher_scheduler.config.beta_schedule,
        prediction_type=args.pred_type,
        t_noise = 1,
        t_clean = 0,
        num_time_windows=args.windows,
    )

    # register hook from saving and loading
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_student_dit.save_pretrained(os.path.join(output_dir, "model_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "model"))
                    if weights:  # Don't pop if empty
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), AllegroTransformer3DModel)
                ema_student_dit.load_state_dict(load_model.state_dict())
                ema_student_dit.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = AllegroTransformer3DModel.from_pretrained(input_dir, subfolder="model")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # create optimizer
    optimizer = torch.optim.AdamW(
        student_dit.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info(f"Optimizer created.")

    # create lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # ===== Data =====
    # prepare dataset and dataloader
    # CFG is in an online manner via ODE solver
    if args.perflow_support_cfg:
        cfg_drop_ratio = 0.1
    else:
        cfg_drop_ratio = 0.
    args.cfg = 0.
    train_dataset = getdataset(args)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=Collate(args),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True, 
    )
    logger.info(f'{len(train_dataset)} samples loaded from {args.meta_file} successfully')


    # ===== Prepare training =====
    # prepare everything with our `accelerator`.
    student_dit, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        student_dit, optimizer, train_dataloader, lr_scheduler
    )

    # initialize tracker and store configuration
    accelerator.init_trackers(
        project_name=args.project_name,
        init_kwargs={
            'wandb': {
                'name': os.path.basename(args.output_dir),
                'dir': args.output_dir,
                'config': vars(args)
            }
        }
    )

    # log the training configuration
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Model = {student_dit}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total training parameters = {sum(p.numel() for p in student_dit.parameters() if p.requires_grad) / 1e9} B")
    global_step = 0
    first_epoch = 0
    progress_info = ProgressInfo(global_step, train_loss=0.0)

    # resume
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch
   
    def sync_gradients_info():
        # Checks if the accelerator has performed an optimization step behind the scenes
        if args.use_ema:
            ema_student_dit.step(student_dit.parameters())
        progress_info.global_step += 1
        accelerator.log({"train_loss": progress_info.train_loss}, step=progress_info.global_step)
        accelerator.print('[%s] step %d, train_loss=%.6f' % (ctime(), progress_info.global_step, progress_info.train_loss), flush=True)
        progress_info.train_loss = 0.0

        # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
        if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
            if progress_info.global_step % args.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{progress_info.global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    def run(model_input, model_kwargs):
        bsz = model_input.shape[0] # model_input: latents in (B,C,T,H,W)
        ## Sample timesteps. NOTE: t \in [1, 0]
        ## Prepare the endpoints of windows, model inputs
        with torch.no_grad():
            # CFG
            text_embeddings = model_kwargs.get('encoder_hidden_states', None)
            null_text_ids = torch.stack(
                [tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt",).input_ids[0]] * bsz
            ).to(accelerator.device)
            null_embeddings = text_encoder(null_text_ids)[0]
            
            if args.perflow_support_cfg:
                assert null_embeddings.shape == text_embeddings.shape
                mask_text = torch.zeros((text_embeddings.shape[0],1,1), dtype=text_embeddings.dtype, device=text_embeddings.device)
                for i in range(text_embeddings.shape[0]):
                    mask_text[i] = 1 if random.random() > cfg_drop_ratio else 0
                text_embeddings_dropout =  mask_text * text_embeddings + (1-mask_text) * null_embeddings
            else:
                text_embeddings_dropout = text_embeddings

            model_kwargs["encoder_hidden_states"] = text_embeddings_dropout

            timepoints = torch.rand((bsz,), device=model_input.device) # [0,1)
            if args.discrete_timesteps == -1:
                timepoints = (timepoints * teacher_num_train_timesteps).floor() / teacher_num_train_timesteps # assert [0, 999/1000]
            else:
                assert isinstance(args.discrete_timesteps, int)
                timepoints = (timepoints * args.discrete_timesteps).floor() / args.discrete_timesteps # in [0, 39/40)
            timepoints = 1 - timepoints # [1, 1/1000], [1, 1/40]

            noise = torch.randn_like(model_input)

            t_start, t_end = perflow_scheduler.time_windows.lookup_window(timepoints)
            latents_start = teacher_scheduler.add_noise(model_input, noise, torch.clamp((t_start*teacher_num_train_timesteps).long()-1, min=0))
            if args.perflow_cfg_sync:
                latents_end = solver.solve(
                    latents                 = latents_start, 
                    t_start                 = t_start,
                    t_end                   = t_end,
                    model                   = teacher_dit, 
                    prompt_embeds           = text_embeddings_dropout, 
                    negative_prompt_embeds  = null_embeddings, 
                    guidance_scale          = 1.0, 
                    num_steps               = args.perflow_solving_steps,
                    num_windows             = args.windows,
                )
            else:
                latents_end = solver.solve(
                    latents                 = latents_start, 
                    t_start                 = t_start,
                    t_end                   = t_end,
                    model                   = teacher_dit, 
                    prompt_embeds           = text_embeddings, 
                    negative_prompt_embeds  = null_embeddings, 
                    guidance_scale          = 7.5, 
                    num_steps               = args.perflow_solving_steps,
                    num_windows             = args.windows,
                )
            latents_t = latents_start + (latents_end - latents_start) / (t_end[:,None,None,None] - t_start[:,None,None,None]) * (timepoints[:, None, None, None] - t_start[:, None, None, None])
            latents_t = latents_t.to(weight_dtype)


        ## prepare targets -> perform inference -> convert -> compute loss
        with torch.no_grad():
            # for noise matching
            # z_e = \lambda_s * z_s + \eta_s * \eps,  --->  \eps = (z_e - \lambda_s * z_s) / \eta_s
            # if args.loss_type == "velocity_matching" and args.pred_type == "velocity":
            if teacher_scheduler.config.prediction_type == "v_prediction" and args.pred_type == "velocity":  # flow matching
                targets = (latents_end - latents_start) / (t_end[:,None,None,None] - t_start[:,None,None,None])
            elif teacher_scheduler.config.prediction_type == "epsilon" and args.pred_type == "diff_eps":    # DDPM
                _, _, _, _, gamma_s_e, _, _ = perflow_scheduler.get_window_alpha(timepoints.float().cpu())
                gamma_s_e = gamma_s_e[:,None,None,None].to(device=model_input.device)
                lambda_s = 1 / gamma_s_e
                eta_s = -1 * ( 1- gamma_s_e**2)**0.5 / gamma_s_e
                targets = (latents_end - lambda_s * latents_start ) / eta_s
            elif teacher_scheduler.config.prediction_type == "epsilon" and args.pred_type == "ddim_eps":    # DDIM
                _, _, _, _, _, alphas_cumprod_start, alphas_cumprod_end = perflow_scheduler.get_window_alpha(timepoints.float().cpu())
                alphas_cumprod_start = alphas_cumprod_start[:,None,None,None].to(device=model_input.device)
                alphas_cumprod_end = alphas_cumprod_end[:,None,None,None].to(device=model_input.device)
                lambda_s = (alphas_cumprod_end / alphas_cumprod_start)**0.5
                eta_s = (1-alphas_cumprod_end)**0.5 - ( alphas_cumprod_end / alphas_cumprod_start * (1-alphas_cumprod_start) )**0.5
                targets = (latents_end - lambda_s * latents_start ) / eta_s
            else:
                raise NotImplementedError

        model_pred = student_dit(
            latents_t, 
            timestep=timepoints.float() * teacher_num_train_timesteps,
            **model_kwargs)[0]
        
        mask = model_kwargs.get('attention_mask', None)
        if torch.all(mask.bool()):
            mask = None
        b, c, _, _, _ = model_pred.shape
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, c, 1, 1, 1).float()  # b t h w -> b c t h w
            mask = mask.reshape(b, -1)
        
        if args.reweighting_scheme is None:
            # loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")
            loss = F.mse_loss(model_pred.float(), targets.float(), reduction="none")
            loss = loss.reshape(b, -1)
            if mask is not None:
                loss = (loss * mask).sum() / mask.sum()  # mean loss on unpad patches
            else:
                loss = loss.mean()
        else:   # TODO:not supported
            if args.reweighting_scheme == 'reciprocal':
                loss_weights = 1.0 / torch.clamp(1.0 - timepoints, min=0.1) / 2.3 # \int_0^{0.9} 1/(1-t)dt = 2.3
                loss = (((model_pred.float() -  targets.float())**2).mean(dim=[1,2,3]) * loss_weights).mean()
            else:
                raise NotImplementedError



        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        progress_info.train_loss += avg_loss.detach().item() / args.gradient_accumulation_steps

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            params_to_clip = student_dit.parameters()
            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            sync_gradients_info()

    def train_one_step(step_, data_item_):
        x, attn_mask, input_ids, cond_mask = data_item_
        assert not torch.any(torch.isnan(x)), 'torch.any(torch.isnan(x))'
        x = x.to(accelerator.device, dtype=vae.dtype)  # B C T H W, 16+4

        attn_mask = attn_mask.to(accelerator.device)  # B T H W
        input_ids = input_ids.to(accelerator.device)  # B 1 L
        cond_mask = cond_mask.to(accelerator.device)  # B 1 L

        with torch.no_grad():
            B, N, L = input_ids.shape  # B 1 L
            # use batch inference
            input_ids_ = input_ids.reshape(-1, L)
            cond_mask_ = cond_mask.reshape(-1, L)
            cond = text_encoder(input_ids_, cond_mask_)['last_hidden_state'].detach()  # B 1 L D
            cond = cond.reshape(B, N, L, -1)
            # Map input images to latent space + normalize latents

            x = torch.cat(
                [
                    vae.encode(x[i:i+1]).latent_dist.sample().mul_(vae.scale_factor)\
                        for i in range(x.shape[0])
                ]
            ) # B C T H W

        with accelerator.accumulate(student_dit):
            assert not torch.any(torch.isnan(x)), 'after vae'
            x = x.to(weight_dtype)
            model_kwargs = dict(encoder_hidden_states=cond, attention_mask=attn_mask,
                                encoder_attention_mask=cond_mask)
            run(x, model_kwargs)

        if progress_info.global_step >= args.max_train_steps:
            return True

        return False

    def train_all_epoch(num_train_epochs):
        for epoch in range(first_epoch, num_train_epochs):
            progress_info.train_loss = 0.0
            if progress_info.global_step >= args.max_train_steps:
                return True

            for step, data_item in enumerate(train_dataloader):
                if train_one_step(step, data_item):
                    break

    train_all_epoch(num_train_epochs)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # dataset & dataloader
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--meta_file", type=str, required=True)
    parser.add_argument("--sample_rate", type=str, default='1')
    parser.add_argument("--num_frames", type=int, default=88)
    parser.add_argument("--max_height", type=int, default=720)
    parser.add_argument("--max_width", type=int, default=1280)
    parser.add_argument("--hw_thr", type=float, default=1.0)
    parser.add_argument("--hw_aspect_thr", type=float, default=1.5)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument('--cfg', type=float, default=0.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")

    # text encoder & vae & diffusion model
    parser.add_argument("--dit", type=str, default=None, help="Path to the Teacher Diffusion model.")
    # parser.add_argument("--dit_config", type=str, default=None)
    parser.add_argument("--vae", type=str, default=None, help="Path to the VAE model.")
    parser.add_argument("--vae_load_mode", type=str, default="encoder_only")
    parser.add_argument("--enable_ae_compile", action="store_true")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to the Tokenizer model.")
    parser.add_argument("--text_encoder", type=str, default=None, help="Path to the Text Encoder model.")
    parser.add_argument("--scheduler", type=str, default=None, help="Path to the Teacher diffusion scheduler.")
    parser.add_argument("--cache_dir", type=str, default="./.cache")
    parser.add_argument('--enable_stable_fp32', action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")

    # diffusion setting
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556.")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--noise_offset", type=float, default=0.02, help="The scale of noise offset.")

    # perflow setting
    parser.add_argument("--windows", type=int, default=16,)
    parser.add_argument("--discrete_timesteps", type=int, default=-1,)
    parser.add_argument("--perflow_support_cfg", action="store_true", default=False,)
    parser.add_argument("--perflow_cfg_sync", action="store_true", default=False,)
    parser.add_argument("--pred_type", type=str, default=None, help="The prediction_type that shall be used for teacher training. Choose in ddim_eps, diff_eps, velocity or leave `None`. If left to `None` the default prediction type of the scheduler: `teacher_scheduler.config.prediciton_type` is chosen.")
    parser.add_argument("--perflow_solving_steps", type=int, default=2,)
    parser.add_argument("--reweighting_scheme", type=str, default=None,)



    # validation & logs
    parser.add_argument("--output_dir", type=str, default="./output", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help=("Max number of checkpoints to store."))
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help=(
                            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                            " training using `--resume_from_checkpoint`."
                        ),
                        )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=(
                            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
                        ),
                        )
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help=(
                            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    # optimizer & scheduler
    parser.add_argument("--max_train_steps", type=int, default=1000000, help="Total number of training steps to perform. ")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimizer", type=str, default="adamW", help='The optimizer type to use. Choose between ["AdamW", "prodigy"]')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer and Prodigy optimizers.")
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True, help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True, help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. Ignored if optimizer is adamW")
    parser.add_argument("--prodigy_beta3", type=float, default=None,
                        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
                             "uses the value of square root of beta2. Ignored if optimizer is adamW",
                        )
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    main(args)
