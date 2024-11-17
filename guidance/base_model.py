import os 
import torch
from abc import *
from pathlib import Path
from datetime import datetime

from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline

from diffusion.stable_diffusion import StableDiffusion


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

    
class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.init_model()
        self.init_mapper()
        
    def initialize(self):
        now = get_current_time()
        save_top_dir = self.config.save_top_dir
        tag = self.config.tag
        save_dir_now = self.config.save_dir_now 
        
        if save_dir_now:
            self.output_dir = Path(save_top_dir) / f"{tag}/{now}"
        else:
            self.output_dir = Path(save_top_dir) / f"{tag}"
        
        if not os.path.isdir(self.output_dir):
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            print(f"Results exist in the output directory, use time string to avoid name collision.")
            exit(0)
            
        print("[*] Saving at ", self.output_dir)
    
    
    @abstractmethod
    def init_mapper(self, **kwargs):
        pass
    
    
    @abstractmethod
    def forward_mapping(self, z_t, **kwargs):
        pass
    
    
    @abstractmethod
    def inverse_mapping(self, x_ts, **kwargs):
        pass
    
    
    @abstractmethod
    def compute_noise_preds(self, xts, ts, **kwargs):
        pass
        
    
    def init_model(self):
        if self.config.model == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_path,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            self.model = StableDiffusion(
                **pipe.components,
            )
            
            del pipe
            
        elif self.config.model == "deepfloyd":
            self.stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            )
            self.stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            
            scheduler = DDIMScheduler.from_config(self.stage_1.scheduler.config)
            self.stage_1.scheduler = self.stage_2.scheduler = scheduler
            
        else:
            raise NotImplementedError(f"Invalid model: {self.config.model}")
        
        
        if self.config.model in ["sd"]:
            self.model.text_encoder.requires_grad_(False)
            self.model.unet.requires_grad_(False)
            if hasattr(self.model, "vae"):
                self.model.vae.requires_grad_(False)
        else:
            self.stage_1.text_encoder.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            
            self.stage_1 = self.stage_1.to(self.device)
            self.stage_2 = self.stage_2.to(self.device)
                
                
    def compute_tweedie(self, xts, eps, timestep, alphas, sigmas, **kwargs):
        """
        Input:
            xts, eps: [B,*]
            timestep: [B]
            x_t = alpha_t * x0 + sigma_t * eps
        Output:
            pred_x0s: [B,*]
        """

        pred_x0s = (xts - sigmas[timestep] * eps) / alphas[timestep]
        return pred_x0s

        
    def compute_prev_state(
        self, xts, pred_x0s, timestep, **kwargs,
    ):
        """
        Input:
            xts: [N,C,H,W]
        Output:
            pred_prev_sample: [N,C,H,W]
        """
        
        """
        task 1 (wide image)
        """
        # s = self.model.scheduler
        """
        task 2 (ambiguous image)
        """
        s = self.stage_1.scheduler

        prev_t = timestep - s.config.num_train_timesteps // s.num_inference_steps
        alpha_t = torch.sqrt(s.alphas_cumprod[timestep])
        one_minus_alpha_t = torch.sqrt(1 - s.alphas_cumprod[timestep])
        alpha_prev_t = torch.where(prev_t >= 0, torch.sqrt(s.alphas_cumprod[prev_t]), torch.tensor(1.0, device=xts.device))
        one_minus_alpha_prev_t = torch.where(prev_t >= 0, torch.sqrt(1 - s.alphas_cumprod[prev_t]), torch.tensor(0.0, device=xts.device))

        xt = one_minus_alpha_prev_t / one_minus_alpha_t
        x0 = alpha_prev_t - alpha_t / one_minus_alpha_t * one_minus_alpha_prev_t

        pred_prev_sample = xts * xt + pred_x0s * x0

        return pred_prev_sample

    def one_step_process(
        self, input_params, timestep, alphas, sigmas, **kwargs
    ):
        """
        Input:
            latents: either xt or zt. [B,*]
        Output:
            output: the same with latent.
        """
        
        xts = input_params["xts"]

        eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
        x0s = self.compute_tweedie(
            xts, eps_preds, timestep, alphas, sigmas, **kwargs
        )
        
        # Synchronization using SyncTweedies 
        z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
        x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization
        
        x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)

        out_params = {
            "x0s": x0s,
            "z0s": None,
            "x_t_1": x_t_1,
            "z_t_1": None,
        }

        return out_params