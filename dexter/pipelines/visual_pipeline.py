"""Lightweight wrapper around the diffusion UNet and scheduler for sampling."""

import torch

from dexter.config import RunConfig
from ..utils import latents_to_torch_img


class VisualPipeline:
    """Generates images from text embeddings using the diffusion UNet and scheduler."""

    def __init__(self, scheduler, unet, vae, device: str = "cuda") -> None:
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae
        self.device = device

    def generate(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cfg: RunConfig,
        dim: int = 512,
        guidance_scale: float = 1.0,
    ):
        """Generate a batch of latents and decoded images given conditional text embeddings.

        Args:
            cond: Conditional text embeddings [1, seq, hidden].
            uncond: Unconditional embeddings [1, seq, hidden].
            cfg: Run configuration including batch size and diffusion steps.
            dim: Spatial size of output image (latent is dim/8).
            guidance_scale: CFG scale for classifier-free guidance.

        Returns:
            Tuple (latents, img_tensor) where img_tensor is decoded to [B,3,H,W].
        """
        cond = cond.repeat(cfg.bs, 1, 1)
        uncond = uncond.repeat(cfg.bs, 1, 1)
        emb = torch.cat([uncond, cond])

        latents = (
            torch.randn(
                (cfg.bs, self.unet.config.in_channels, dim // 8, dim // 8),
                generator=torch.Generator().manual_seed(0),
            )
            .to(self.device)
            .half()
        )

        self.scheduler.set_timesteps(cfg.diff_steps)
        latents = latents.to(self.device) * self.scheduler.init_noise_sigma

        for ts in self.scheduler.timesteps:
            with torch.no_grad():
                inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)

            u, t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
            pred = u + guidance_scale * (t - u)
            latents = self.scheduler.step(pred, ts, latents).prev_sample

        img = latents_to_torch_img(latents, self.vae)
        return latents, img
