"""Utilities to sample images and evaluate classifier activation scores."""

import ast
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torchvision import transforms as tfms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import CLIPTextModel

from ..utils import latents_to_torch_img, text_enc

# Load the top-5 feature indices per class once.
_TOPK_PATH = Path(__file__).resolve().parent.parent / "tools" / "top_k_feats.py"
with open(_TOPK_PATH, "r") as f:
    TOP5_FEATURES = ast.literal_eval(f.read())

# Default transforms that match the ImageNet pretrained normalization used before.
DEFAULT_TRANSFORMS = tfms.Compose(
    [
        tfms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=None),
        tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _set_seed(seed: Optional[int]) -> None:
    """Seed all major RNGs for reproducible activation scores."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _forward_with_features(
    classifier: torch.nn.Module, inputs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the classifier and return (logits, features).
    Prefers calling with with_latent=True if supported; otherwise captures the
    pre-classifier activations via a forward hook on the last linear layer.
    """
    try:
        outputs = classifier(inputs, with_latent=True)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            return outputs  # type: ignore[return-value]
    except TypeError:
        pass  # fall back to hook-based feature capture

    feature_container: dict = {}

    def _hook(module, hook_inputs, hook_output):
        feature_container["feat"] = hook_inputs[0]

    if hasattr(classifier, "fc"):
        hook_module = classifier.fc
    elif hasattr(classifier, "classifier"):
        classifier_module = classifier.classifier
        hook_module = (
            classifier_module[-1]
            if isinstance(classifier_module, torch.nn.Sequential)
            else classifier_module
        )
    else:
        raise ValueError(
            "classifier must support `with_latent=True` or expose `fc`/`classifier`."
        )

    handle = hook_module.register_forward_hook(_hook)
    try:
        logits = classifier(inputs)
    finally:
        handle.remove()

    features = feature_container.get("feat")
    if features is None:
        raise ValueError("Failed to capture classifier features via forward hook.")
    return logits, features


def activation_score(
    classifier: torch.nn.Module,
    prompt: str,
    *,
    class_to_activate: int,
    n_images: int = 100,
    diff_steps: int = 4,
    batch_size: int = 1,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    transforms: Optional[tfms.Compose] = DEFAULT_TRANSFORMS,
    feature_indices: Optional[Sequence[int]] = None,
    guidance_scale: float = 1.0,
    latent_dim: int = 512,
    save_images_to: Optional[str] = None,
    pipe: Optional[DiffusionPipeline] = None,
) -> Tuple[float, Image.Image, float]:
    """
    Generate samples conditioned on a text prompt and score them with the classifier.

    The classifier must accept `with_latent=True` and return a tuple of (logits, features).

    Returns:
        - Activation score (correct_preds / n_images)
        - Last generated PIL image
        - Mean activation value across steps
    """
    _set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).half().eval()

    feats = feature_indices or TOP5_FEATURES[class_to_activate]

    own_pipe = pipe is None
    if own_pipe:
        pipe = DiffusionPipeline.from_pretrained(
            "compvis/stable-diffusion-v1-4",
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
        )
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device=device, dtype=torch.float16)

    tokenizer = pipe.tokenizer
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).half()
    vae = pipe.vae
    scheduler = pipe.scheduler
    unet = pipe.unet

    if save_images_to is not None:
        os.makedirs(save_images_to, exist_ok=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    correct_preds = 0
    activation_values = []
    last_image = None

    scheduler.set_timesteps(diff_steps)

    with torch.inference_mode():
        for step in tqdm(range(n_images)):
            scheduler.set_timesteps(diff_steps)
            cond = text_enc(prompt, tokenizer, clip_text_encoder, device=device)
            uncond = text_enc([""], tokenizer, clip_text_encoder, cond.shape[1], device=device)
            cond = cond.repeat(batch_size, 1, 1)
            uncond = uncond.repeat(batch_size, 1, 1)
            emb = torch.cat([uncond, cond])

            latent_kwargs = {"device": device, "dtype": torch.float16}
            if generator is not None:
                latent_kwargs["generator"] = generator
            latents = torch.randn(
                (batch_size, unet.config.in_channels, latent_dim // 8, latent_dim // 8),
                **latent_kwargs,
            )
            latents = latents * scheduler.init_noise_sigma

            for ts in scheduler.timesteps:
                model_in = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
                u, t = unet(model_in, ts, encoder_hidden_states=emb).sample.chunk(2)
                pred = u + guidance_scale * (t - u)
                latents = scheduler.step(pred, ts, latents).prev_sample

            img_tensor = latents_to_torch_img(latents, vae)
            classifier_input = transforms(img_tensor) if transforms is not None else img_tensor

            logits, features = _forward_with_features(classifier, classifier_input)

            targets = torch.full((classifier_input.shape[0],), class_to_activate, device=device, dtype=torch.long)
            ce_loss = torch.nn.functional.cross_entropy(logits, targets)
            max_losses = [-features[:, idx].mean() for idx in feats[:5]]

            predicted = logits.argmax(dim=1)
            correct_preds += (predicted == class_to_activate).sum().item()

            activation_values.append(ce_loss.item() + float(np.mean([loss.item() for loss in max_losses])))

            last_image = Image.fromarray(
                (img_tensor[0].detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255)
                .astype("uint8")
            )
            if save_images_to is not None:
                last_image.save(os.path.join(save_images_to, f"{step:04d}.png"))

    mean_activation = float(np.mean(activation_values)) if activation_values else 0.0
    activation_score_value = correct_preds / float(n_images) if n_images else 0.0
    return activation_score_value, last_image, mean_activation


if __name__ == "__main__":
    # Basic sanity check to run the module directly.
    from dexter.classifiers import build_classifier_and_transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf, tfms = build_classifier_and_transforms("resnet50", device=device, use_tfms=True)
    score, _,_ = activation_score(
        clf,
        prompt="A picture of a penguin",
        n_images=100,
        transforms=tfms,
        device=device,
        class_to_activate=145
    )
    print(f"Activation score: {score}")
