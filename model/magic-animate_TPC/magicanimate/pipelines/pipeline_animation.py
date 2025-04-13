# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TODO:
1. support multi-controlnet
2. [DONE] support DDIM inversion
3. support Prompt-to-prompt
"""

import inspect, math
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.pipelines.context import (
    get_context_scheduler,
    get_total_steps
)
from magicanimate.utils.util import get_tensor_interpolation_method
from myutil import *
import pdb
import os

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        controlnet: ControlNetModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents, rank, decoder_consistency=None):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0]), disable=(rank!=0)):
            if decoder_consistency is not None:
                video.append(decoder_consistency(latents[frame_idx:frame_idx+1]))
            else:
                video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, clip_length=16):
        shape = (batch_size, num_channels_latents, clip_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                
            latents = latents.repeat(1, 1, video_length//clip_length, 1, 1)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(self, condition, num_videos_per_prompt, device, dtype, do_classifier_free_guidance):
        # prepare conditions for controlnet
        condition = torch.from_numpy(condition.copy()).to(device=device, dtype=dtype) / 255.0
        condition = torch.stack([condition for _ in range(num_videos_per_prompt)], dim=0)
        condition = rearrange(condition, 'b f h w c -> (b f) c h w').clone()
        if do_classifier_free_guidance:
            condition = torch.cat([condition] * 2)
        return condition

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def images2latents(self, images, dtype):
        """
        Convert RGB image to VAE latents
        """
        device = self._execution_device
        images = torch.from_numpy(images).float().to(dtype) / 127.5 - 1
        images = rearrange(images, "f h w c -> f c h w").to(device)
        latents = []
        for frame_idx in range(images.shape[0]):
            latents.append(self.vae.encode(images[frame_idx:frame_idx+1])['latent_dist'].mean * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=20,
        num_actual_inference_steps=10,
        eta=0.0,
        return_intermediates=False,
        **kwargs):
        """
        Adapted from: https://github.com/Yujun-Shi/DragDiffusion/blob/main/drag_pipeline.py#L440
        invert a real image into noise map with determinisc DDIM inversion
        """
        device = self._execution_device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        print("input text embeddings :", text_embeddings.shape)
        # define initial latents
        latents = self.images2latents(image)

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        latents_list = [latents]
        pred_x0_list = [latents]
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):

            if num_actual_inference_steps is not None and i >= num_actual_inference_steps:
                continue
            model_inputs = latents

            # predict the noise
            # NOTE: the u-net here is UNet3D, therefore the model_inputs need to be of shape (b c f h w)
            model_inputs = rearrange(model_inputs, "f c h w -> 1 c f h w")
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            noise_pred = rearrange(noise_pred, "b c f h w -> (b f) c h w")
            
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            return latents, latents_list
        return latents
    
    def interpolate_latents(self, latents: torch.Tensor, interpolation_factor:int, device ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
                    (latents.shape[0],latents.shape[1],((latents.shape[2]-1) * interpolation_factor)+1, latents.shape[3],latents.shape[4]),
                    device=latents.device,
                    dtype=latents.dtype,
                )

        org_video_length = latents.shape[2]
        rate = [i/interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0,i1 in zip( range( org_video_length ),range( org_video_length )[1:] ):
            v0 = latents[:,:,i0,:,:]
            v1 = latents[:,:,i1,:,:]

            new_latents[:,:,new_index,:,:] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(v0.to(device=device),v1.to(device=device),f)
                new_latents[:,:,new_index,:,:] = v.to(latents.device)
                new_index += 1

        new_latents[:,:,new_index,:,:] = v1
        new_index += 1

        return new_latents
    
    def select_controlnet_res_samples(self, controlnet_res_samples_cache_dict, context, do_classifier_free_guidance, b, f):
        _down_block_res_samples = []
        _mid_block_res_sample = []
        for i in np.concatenate(np.array(context)):
            _down_block_res_samples.append(controlnet_res_samples_cache_dict[i][0])
            _mid_block_res_sample.append(controlnet_res_samples_cache_dict[i][1])
        down_block_res_samples = [[] for _ in range(len(controlnet_res_samples_cache_dict[i][0]))]
        for res_t in _down_block_res_samples:
            for i, res in enumerate(res_t):
                down_block_res_samples[i].append(res)
        down_block_res_samples = [torch.cat(res) for res in down_block_res_samples]
        mid_block_res_sample = torch.cat(_mid_block_res_sample)
        
        # reshape controlnet output to match the unet3d inputs
        b = b // 2 if do_classifier_free_guidance else b
        _down_block_res_samples = []
        for sample in down_block_res_samples:
            sample = rearrange(sample, '(b f) c h w -> b c f h w', b=b, f=f)
            if do_classifier_free_guidance:
                sample = sample.repeat(2, 1, 1, 1, 1)
            _down_block_res_samples.append(sample)
        down_block_res_samples = _down_block_res_samples
        mid_block_res_sample = rearrange(mid_block_res_sample, '(b f) c h w -> b c f h w', b=b, f=f)
        if do_classifier_free_guidance:
            mid_block_res_sample = mid_block_res_sample.repeat(2, 1, 1, 1, 1)
            
        return down_block_res_samples, mid_block_res_sample

    def generate_calibration_image_latents(
    self,
    cal: bool,
    spc: int,
    pose_name: str,
    ref_name: str,
    context_frames: int,
    width: int,
    height: int,
    latents_dtype,
    ref_image_latents: torch.Tensor,
    video_length: int,
    max_human_num: int,
    images2latents_func
    ):
        """
        Generates (or repeats) calibrated image latents and corresponding masks based on the given parameters.

        Parameters
        ----------
        cal : bool
            If True, actually load calibrated images from disk and create latents.
            If False, simply repeat ref_image_latents instead.
        spc : int
            Sparsity step for selecting frames. For example, if spc=10, only every 10th frame is processed 
            and the intermediate frames are duplicated from the previous one.
        pose_name : str
            Name for the pose (video).
        ref_name : str
            Name for the reference image.
        context_frames : int
            Batch or group size for padding frames (e.g., if context_frames=8, pads the frame count to the nearest multiple of 8).
        width, height : int
            Target width and height for resizing loaded images.
        latents_dtype :
            The data type used to store latents (e.g., torch.float16).
        ref_image_latents : torch.Tensor
            Latents for the reference image.
        video_length : int
            Total number of video frames (used when cal=False to replicate latents/masks).
        images2latents_func : Callable
            A function that takes (N, H, W, 3) NumPy arrays and converts them into latents 
            (e.g., shape [N, C, H, W]) of type latents_dtype, optionally moves them to GPU, etc.

        Returns
        -------
        cal_image_latents : Optional[torch.Tensor]
            The concatenated latents of calibrated images (if cal=True) or repeated ref_image_latents (if cal=False).
            Shape could be [N, C, H, W], depending on your data.
        cal_mask : Optional[torch.Tensor]
            The concatenated mask tensor for each person (if cal=True) or 
            a repeated white mask (if cal=False). Shape could be [N, H, W, 3] or similar.

        Notes
        -----
        - This function expects that each person's calibrated images and masks 
        are located in directories named:
            ./inputs/applications/calibrated_image/pose_<pose_name>_ref_<ref_name>/<e>-person_cal_ref_img
            ./inputs/applications/calibrated_image/pose_<pose_name>_ref_<ref_name>/<e>-person_cal_ref_mask_img
        - If these directories do not exist or are empty, no latents/masks will be generated for that person.
        - Adjust as needed if you have a different directory structure.
        """
        sparsity = spc

        if cal:
            print("Calibration images: ON")
            cal_image_latents = []
            cal_mask = []

            # Process up to max_human_num directories for images and masks
            for e in range(max_human_num):
                fpath = f'./inputs/applications/calibrated_image/pose_{pose_name}_ref_{ref_name}/{e}-person_cal_ref_img'
                mpath = f'./inputs/applications/calibrated_image/pose_{pose_name}_ref_{ref_name}/{e}-person_cal_ref_mask_img'

                if os.path.isdir(fpath):
                    # --- Generate latents from calibrated images ---
                    flst = sorted(os.listdir(fpath))
                    tmp_images = []
                    for i, fname in enumerate(flst):
                        img_path = os.path.join(fpath, fname)
                        img_pil = Image.open(img_path).convert('RGB')
                        if i % sparsity == 0:
                            tmp_images.append(np.array(img_pil.resize((width, height))))
                        else:
                            # Repeat the last image to fill in the gap
                            tmp_images.append(tmp_images[-1])

                    img_stack = np.stack(tmp_images, axis=0)
                    pad_size = (context_frames - (img_stack.shape[0] % context_frames)) % context_frames
                    padded_imgs = np.pad(
                        img_stack,
                        ((0, pad_size), (0, 0), (0, 0), (0, 0)),
                        mode='edge'
                    )

                    # Convert images to latents
                    calibrated_latents = images2latents_func(padded_imgs, latents_dtype).cuda()
                    cal_image_latents.append(calibrated_latents)

                    # --- Generate masks ---
                    flst_mask = sorted(os.listdir(mpath))
                    tmp_masks = []
                    for i, mname in enumerate(flst_mask):
                        msk_path = os.path.join(mpath, mname)
                        msk_pil = Image.open(msk_path).convert('RGB')
                        if i % sparsity == 0:
                            tmp_masks.append(np.array(msk_pil.resize((width, height))))
                        else:
                            tmp_masks.append(tmp_masks[-1])

                    mask_stack = np.stack(tmp_masks, axis=0)
                    pad_size_mask = (context_frames - (mask_stack.shape[0] % context_frames)) % context_frames
                    padded_masks = np.pad(
                        mask_stack,
                        ((0, pad_size_mask), (0, 0), (0, 0), (0, 0)),
                        mode='edge'
                    )

                    cal_mask.append(torch.from_numpy(padded_masks))

            # Concatenate latents/masks if they exist
            cal_image_latents = torch.cat(cal_image_latents, dim=0) if cal_image_latents else None
            cal_mask = torch.cat(cal_mask, dim=0) if cal_mask else None

        else:
            print("Calibration images: OFF")
            # Repeat reference latents for the entire video length
            cal_image_latents = ref_image_latents.repeat([video_length, 1, 1, 1])

            print("Calibration masks: OFF")
            # Create a white mask (all 255) for each frame
            cal_mask_np = np.ones((video_length, width, height, 3), dtype='uint8') * 255
            cal_mask = torch.from_numpy(cal_mask_np)

        return cal_image_latents, cal_mask

    def create_ref_pose_masks(
        self,
        cal: bool,
        pose_name: str,
        ref_name: str,
        width: int,
        height: int,
        context_frames: int,
        video_length: int,
        max_human_num: int = 3
    ):
        """
        Creates or loads reference masks (ref_mask) and pose masks (pose_mask).

        If cal == True, reads the mask files for each person from:
        ./inputs/applications/calibrated_image/pose_<pose_name>_ref_<ref_name>/
            <e>-person_ref_mask_img/ref.png  (for reference masks)
            <e>-person_pose_mask_img/        (for pose masks)
        and stacks them into tensors.

        If cal == False, uses white (all 255) placeholders.

        Parameters
        ----------
        cal : bool
            Whether calibration is enabled (i.e., actual mask files exist).
        pose_name : str
            Name of the pose (video).
        ref_name : str
            Name of the reference.
        width : int
            Desired image width to resize.
        height : int
            Desired image height to resize.
        context_frames : int
            Number of frames to pad up to (multiple).
        video_length : int
            Number of frames in the video (when calibration is off).
        max_human_num : int, optional
            Maximum number of people to handle, by default 3.

        Returns
        -------
        E : int
            The index of the last found person (e.g., 0, 1, or 2).
        ref_mask : torch.Tensor
            The reference mask(s). Shape depends on how many people and calibration mode.
        pose_mask : torch.Tensor
            The pose mask(s). Possibly stacked across frames and people, or a placeholder if cal == False.

        Notes
        -----
        - All directories checked must exist and contain the expected files if cal == True.
        - Output masks are loaded and resized to (width, height).
        """
        E = 0
        if cal:
            ref_mask_list = []
            # 1) Load reference masks
            for e in range(max_human_num):
                rfpath = f'./inputs/applications/calibrated_image/pose_{pose_name}_ref_{ref_name}/{e}-person_ref_mask_img'
                if os.path.isdir(rfpath):
                    # Reference mask path: .../ref.png
                    ref_png_path = os.path.join(rfpath, 'ref.png')
                    ref_mask_ = Image.open(ref_png_path).convert('RGB')
                    ref_mask_ = np.array(ref_mask_.resize((width, height)))
                    ref_mask_ = np.expand_dims(ref_mask_, axis=0)
                    ref_mask_ = torch.from_numpy(ref_mask_)
                    ref_mask_list.append(ref_mask_)
                    E = e

            # Combine reference masks
            if ref_mask_list:
                ref_mask = torch.cat(ref_mask_list, dim=0)
            else:
                # If no directories found
                ref_mask = torch.ones((1, width, height, 3), dtype=torch.uint8) * 255

            # 2) Load pose masks
            pose_mask_list = []
            for e in range(max_human_num):
                ppath = f'./inputs/applications/calibrated_image/pose_{pose_name}_ref_{ref_name}/{e}-person_pose_mask_img'
                if os.path.isdir(ppath):
                    flst = os.listdir(ppath)
                    flst.sort()
                    tmp = []
                    for i, fname in enumerate(flst):
                        pmask_name = os.path.join(ppath, fname)
                        tmp_ = Image.open(pmask_name).convert('RGB')
                        tmp.append(tmp_)

                    # Pad to match context_frames
                    tmp2 = np.stack(tmp, axis=0)
                    pad_size = (context_frames - (tmp2.shape[0] % context_frames)) % context_frames
                    pose_mask_ = np.pad(tmp2, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='edge')
                    pose_mask_list.append(torch.from_numpy(pose_mask_))

            if pose_mask_list:
                pose_mask = torch.cat(pose_mask_list, dim=0)
            else:
                # If no directories found
                pose_mask = torch.ones((1, width, height, 3), dtype=torch.uint8) * 255

        else:
            # If calibration is off, just fill with white placeholders
            print("reference masks: X")
            ref_mask = np.ones((1, width, height, 3), dtype='uint8') * 255
            ref_mask = torch.from_numpy(ref_mask)

            pose_mask = np.ones((video_length, width, height, 3), dtype='uint8') * 255
            pose_mask = torch.from_numpy(pose_mask)

        return E, ref_mask, pose_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controlnet_condition: list = None,
        controlnet_conditioning_scale: float = 1.0,
        context_frames: int = 16,
        context_stride: int = 1,
        context_overlap: int = 4,
        context_batch_size: int = 1, 
        context_schedule: str = "uniform",
        init_latents: Optional[torch.FloatTensor] = None,
        num_actual_inference_steps: Optional[int] = None,
        appearance_encoder = None, 
        reference_control_writer = None,
        reference_control_reader = None,
        source_image: str = None,
        decoder_consistency = None,
        pose_name = None,
        ref_name = None,
        cal = None,
        spc = None,
        **kwargs,
    ):
        """
        New args:
        - controlnet_condition          : condition map (e.g., depth, canny, keypoints) for controlnet
        - controlnet_conditioning_scale : conditioning scale for controlnet
        - init_latents                  : initial latents to begin with (used along with invert())
        - num_actual_inference_steps    : number of actual inference steps (while total steps is num_inference_steps) 
        """
        controlnet = self.controlnet

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        text_embeddings = torch.cat([text_embeddings] * context_batch_size)
        
        reference_control_writer = ReferenceAttentionControl(appearance_encoder, do_classifier_free_guidance=True, mode='write', batch_size=context_batch_size)
        reference_control_reader = ReferenceAttentionControl(self.unet, do_classifier_free_guidance=True, mode='read', batch_size=context_batch_size)
        

        is_dist_initialized = kwargs.get("dist", False)
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)

        # Prepare video
        assert num_videos_per_prompt == 1   # FIXME: verify if num_videos_per_prompt > 1 works
        assert batch_size == 1              # FIXME: verify if batch_size > 1 works
        control = self.prepare_condition(
                condition=controlnet_condition,
                device=device,
                dtype=controlnet.dtype,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        controlnet_uncond_images, controlnet_cond_images = control.chunk(2)

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        if init_latents is not None:
            latents = rearrange(init_latents, "(b f) c h w -> b c f h w", f=video_length)
        else:
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
            )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare text embeddings for controlnet
        controlnet_text_embeddings = text_embeddings.repeat_interleave(video_length, 0)
        _, controlnet_text_embeddings_c = controlnet_text_embeddings.chunk(2)
        
        controlnet_res_samples_cache_dict = {i:None for i in range(video_length)}

        # For img2img setting
        if num_actual_inference_steps is None:
            num_actual_inference_steps = num_inference_steps

        if isinstance(source_image, str):
            ref_image_latents = self.images2latents(np.array(Image.open(source_image).resize((width, height)))[None, :], latents_dtype).cuda()
        elif isinstance(source_image, np.ndarray):
            ref_image_latents = self.images2latents(source_image[None, :], latents_dtype).cuda()
        
        ## add calibrated image latents
        cal_image_latents, cal_mask = self.generate_calibration_image_latents(
            cal=cal,
            spc=spc,
            pose_name=pose_name,
            ref_name=ref_name,
            context_frames=context_frames,
            width=width,
            height=height,
            latents_dtype=latents_dtype,
            ref_image_latents=ref_image_latents,
            video_length=video_length,
            max_human_num=3,
            images2latents_func=self.images2latents
        )
        
        E, ref_mask, pose_mask = self.create_ref_pose_masks(cal, pose_name, ref_name, width, height, context_frames, video_length)
        
        ref_cal_mask = torch.cat([ref_mask, cal_mask], dim=0)
        ref_cal_mask = ref_cal_mask[:,:,:,0]
        ref_cal_mask = ref_cal_mask.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1) # 2 x (ref+ref...cal+cal...) x 512 x 512
        
        pose_mask = pose_mask[:,:,:,0]
        pose_mask = pose_mask.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1) # 2 x (p1, p2, ...) x 512 x 512

        register_p_mask(self, pose_mask)
        register_mask(self, ref_cal_mask)
        register_cal(self, cal)
        register_person(self, E+1)

        context_scheduler = get_context_scheduler(context_schedule)
        
        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=(rank!=0)):
            if num_actual_inference_steps is not None and i < num_inference_steps - num_actual_inference_steps:
                continue

            noise_pred = torch.zeros(
                (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            counter = torch.zeros(
                (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
            )

            context_batch_size = 1 + video_length*(E+1) # E+1: number of person, which corresponds to each calibrated human
            ref_latent = torch.cat([ref_image_latents, cal_image_latents], dim=0)
            ref_latent = ref_latent.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1)

            appearance_encoder(
                ref_latent,
                t,
                encoder_hidden_states=text_embeddings.repeat_interleave(context_batch_size, 0),
                return_dict=False,
            )
            context_batch_size = 1

            context_queue = list(context_scheduler(
                0, num_inference_steps, latents.shape[2], context_frames, context_stride, 0
            ))

            # 
            register_time(self, t)

            num_context_batches = math.ceil(len(context_queue) / context_batch_size)
            for i in range(num_context_batches):
                context = context_queue[i*context_batch_size: (i+1)*context_batch_size]
                # expand the latents if we are doing classifier free guidance
                controlnet_latent_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                )
                controlnet_latent_input = self.scheduler.scale_model_input(controlnet_latent_input, t)

                # prepare inputs for controlnet
                b, c, f, h, w = controlnet_latent_input.shape
                controlnet_latent_input = rearrange(controlnet_latent_input, "b c f h w -> (b f) c h w")
                
                # controlnet inference
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    controlnet_latent_input,
                    t,
                    encoder_hidden_states=torch.cat([controlnet_text_embeddings_c[c] for c in context]),
                    controlnet_cond=torch.cat([controlnet_cond_images[c] for c in context]),
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )

                for j, k in enumerate(np.concatenate(np.array(context))):
                    controlnet_res_samples_cache_dict[k] = ([sample[j:j+1] for sample in down_block_res_samples], mid_block_res_sample[j:j+1])
            context_queue = list(context_scheduler(
                0, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
            ))

            num_context_batches = math.ceil(len(context_queue) / context_batch_size)
            global_context = []
            for i in range(num_context_batches):
                global_context.append(context_queue[i*context_batch_size: (i+1)*context_batch_size])
            
            for context in global_context[rank::world_size]:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents[:, :, c] for c in context])
                    .to(device)
                    .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                b, c, f, h, w = latent_model_input.shape
                down_block_res_samples, mid_block_res_sample = self.select_controlnet_res_samples(
                    controlnet_res_samples_cache_dict, 
                    context,
                    do_classifier_free_guidance,
                    b, f
                )
                
                reference_control_reader.update(reference_control_writer)
                
                # register localization of current feeding
                register_feed(self, context[0])
                
                # predict the noise residual
                
                pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=text_embeddings[:b],
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                reference_control_reader.clear()
                
                pred_uc, pred_c = pred.chunk(2)
                pred = torch.cat([pred_uc.unsqueeze(0), pred_c.unsqueeze(0)])
                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + pred[:, j]
                    counter[:, :, c] = counter[:, :, c] + 1
                    
            if is_dist_initialized:
                noise_pred_gathered = [torch.zeros_like(noise_pred) for _ in range(world_size)]
                if rank == 0:
                    dist.gather(tensor=noise_pred, gather_list=noise_pred_gathered, dst=0)
                else:
                    dist.gather(tensor=noise_pred, gather_list=[], dst=0)
                dist.barrier()

                if rank == 0:
                    for k in range(1, world_size):
                        for context in global_context[k::world_size]:
                            for j, c in enumerate(context):
                                noise_pred[:, :, c] = noise_pred[:, :, c] + noise_pred_gathered[k][:, :, c] 
                                counter[:, :, c] = counter[:, :, c] + 1

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            
            if is_dist_initialized:
                dist.broadcast(latents, 0)
                dist.barrier()
            
            reference_control_writer.clear()

        interpolation_factor = 1
        latents = self.interpolate_latents(latents, interpolation_factor, device)
        # Post-processing

        video = self.decode_latents(latents, rank, decoder_consistency=decoder_consistency)

        if is_dist_initialized:
            dist.barrier()

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video
        
        return AnimationPipelineOutput(videos=video)
