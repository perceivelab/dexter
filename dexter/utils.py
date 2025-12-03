"""Utility helpers for diffusion setup, prompt handling, image I/O, and scoring."""

import ast
import base64
import datetime
import json
import os
import random
import uuid
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as tfms
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from transformers import BertTokenizer, CLIPTextModel



PACKAGE_ROOT = Path(__file__).resolve().parent
IMAGENET_CLASSES_PATH = PACKAGE_ROOT / "data" / "imagenet_classes.json"


def select_top_features(classifier, class_to_activate):
    """Retrieve the top-k discriminative features for a given model/class pair.

    Args:
        classifier: One of the supported Classifier identifiers in `feat_files`.
        class_to_activate: Target ImageNet class index.

    Returns:
        List[int]: Five feature indices sorted by importance for that class.
    """
    feat_files = {
        "robust50": "top_k_feats.py",
        "resnet50": "top_k_feats_resnet50.py",
        "alexnet": "top_k_feats_alexnet.py",
        "vit_b_16": "top_k_feats_vit.py",
    }
    if classifier not in feat_files:
        raise ValueError(
            "classifier must be one of [robust50, resnet50, alexnet, vit_b_16] for feature selection"
        )

    topk_path = PACKAGE_ROOT / "tools" / feat_files[classifier]
    with open(topk_path, "r") as file_handle:
        top5_feats = ast.literal_eval(file_handle.read())
    return top5_feats[class_to_activate]


def set_random_seed(seed):
    """Configure reproducible seeds across common randomness sources.

    Args:
        seed: Seed value or None to skip seeding.
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def init_diffusion_pipeline(device):
    """Load Stable Diffusion with LCM LoRA weights and return ready-to-use pieces.

    Args:
        device: Device string for model weights ("cuda" or "cpu").

    Returns:
        Tuple (pipeline, tokenizer, CLIP text encoder) ready for generation.
    """
    model_id = "compvis/stable-diffusion-v1-4"
    lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ),
    )
    pipe.load_lora_weights(lcm_lora_id)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device=device, dtype=torch.float16)
    tokenizer = pipe.tokenizer
    clip_orig_text_encoder = (
        CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).half()
    )
    return pipe, tokenizer, clip_orig_text_encoder


def load_translation_matrix():
    """Load the cached CLIP->BERT translation matrix from checkpoints.

    Returns:
        torch.Tensor containing the translation matrix.
    """
    checkpoints_dir = PACKAGE_ROOT.parent / "checkpoints"
    return torch.load(checkpoints_dir / "translation_matrix.pt")


def ensure_output_dir(outdir, class_to_activate, args):
    """Create a timestamped output folder and persist the run arguments.

    Args:
        outdir: Base output directory.
        class_to_activate: Class id used to tag the run folder.
        args: Serializable config dictionary written to args.json.

    Returns:
        Absolute path to the created run folder.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    experiment_name = (
        str(class_to_activate)
        + "_"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + str(uuid.uuid4())[:8]
    )
    out_folder = os.path.join(outdir, experiment_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with open(os.path.join(out_folder, "args.json"), "w") as f:
        json.dump(args, f)
    return out_folder


def validate_options(sp_init, no_match_uniform, mask_type):
    """Validate CLI options early to surface typos with a clear error.

    Raises:
        ValueError: When any option is outside the allowed set.
    """
    if sp_init not in ["none", "cls", "sep"]:
        raise ValueError("sp_init must be one of [none, cls, sep]")
    if no_match_uniform not in ["none", "clip_vocab"]:
        raise ValueError("no_match_uniform must be one of [none, clip_vocab]")
    if mask_type not in ["single_mask", "multi_mask"]:
        raise ValueError("mask_type must be one of [single_mask, multi_mask]")


def forward_with_features(model, images):
    """
    Returns logits and penultimate features when available.

    Prefers models that accept `with_latent=True`; otherwise captures activations
    via a forward hook on avgpool if present.

    Args:
        model: Torch vision model; ideally supports `with_latent=True`.
        images: Tensor batch shaped [B, C, H, W].

    Returns:
        Tuple (logits, features) where features may be None if not obtainable.
    """
    try:
        logits, feats = model(images, with_latent=True)
        return logits, feats
    except TypeError:
        pass
    except ValueError:
        pass

    # Vision Transformers expose a `forward_features` helper we can reuse to
    # grab token embeddings alongside the normal logits.
    if hasattr(model, "forward_features"):
        feats = model.forward_features(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        logits = model(images)
        return logits, feats

    # Older torchvision ViT implementations don't have forward_features; manually
    # mirror the forward path so we can surface the encoder tokens.
    try:
        from torchvision.models.vision_transformer import VisionTransformer
    except ImportError:  # pragma: no cover - defensive fallback for unusual installs
        VisionTransformer = None

    if VisionTransformer is not None and isinstance(model, VisionTransformer):
        x = model._process_input(images)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        feats = model.encoder(x)
        logits = model.heads(feats[:, 0])
        return logits, feats

    feats = None
    if hasattr(model, "avgpool"):
        captured = {}

        def _hook(_, __, output):
            captured["feat"] = torch.flatten(output, 1)

        handle = model.avgpool.register_forward_hook(_hook)
        logits = model(images)
        handle.remove()
        feats = captured.get("feat")
        return logits, feats

    logits = model(images)
    return logits, feats


## Helper functions
def load_image(p):
    """Load and normalize an RGB image from disk to 512x512 PIL.

    Args:
        p: Path to an image file.

    Returns:
        A resized RGB PIL.Image.
    """
    return Image.open(p).convert("RGB").resize((512, 512))


def pil_to_latents(image, vae):
    """Encode a PIL image into latent space using the VAE.

    Args:
        image: PIL image in RGB.
        vae: Diffusers VAE model.

    Returns:
        Latent tensor suitable for UNet conditioning.
    """
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda")
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist


def latents_to_pil(latents, vae):
    """Decode latent tensors to a list of PIL images.

    Args:
        latents: Tensor of latent samples.
        vae: Diffusers VAE model.

    Returns:
        List of PIL.Image objects.
    """
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def latents_to_torch_img(latents, vae):
    """Decode latent tensors to torch image tensors in [0,1].

    Args:
        latents: Tensor of latent samples.
        vae: Diffusers VAE model.

    Returns:
        Torch tensor shaped [B, 3, H, W] in [0,1].
    """
    latents = (1 / 0.18215) * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def text_enc(prompts, tokenizer, text_encoder, maxlen=None, device="cuda"):
    """Tokenize prompts and run them through the provided text encoder.

    Args:
        prompts: List[str] or str of prompts.
        tokenizer: Matching tokenizer.
        text_encoder: Matching text encoder model.
        maxlen: Optional override for max token length.
        device: Device string for computation.

    Returns:
        Tensor of embeddings shaped [B, seq_len, hidden_dim].
    """
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    inp = tokenizer(
        prompts,
        padding="max_length",
        max_length=maxlen,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(inp.input_ids.to(device))[0]

def initialize_dynamic_target(mask_type):
    """Sample initial target word(s) depending on mask type.

    Args:
        mask_type: "single_mask" or "multi_mask".

    Returns:
        A string (single mask) or tuple of six strings (multi mask).
    """

    assert mask_type in ["single_mask", "multi_mask"], "Mask type not supported"

    if mask_type == "single_mask":
        # select a valid target word
        right_target = False
        while not right_target:
            target_word1 = random.choice(
                list(
                    BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
                    .get_vocab()
                    .keys()
                )
            )
            if "#" not in target_word1 and "[" not in target_word1:
                right_target = True

        print("Target word1", target_word1)

        return target_word1

    elif mask_type == "multi_mask":
        # select a valid target word
        right_target = False
        while not right_target:
            vocab = list(BertTokenizer.from_pretrained("google-bert/bert-base-uncased").get_vocab().keys())
            target_word1 = random.choice(vocab)
            target_word2 = random.choice(vocab)
            target_word3 = random.choice(vocab)
            target_word4 = random.choice(vocab)
            target_word5 = random.choice(vocab)
            target_word6 = random.choice(vocab)
            if (
                "#" not in target_word1
                and "#" not in target_word2
                and "#" not in target_word3
                and "#" not in target_word4
                and "#" not in target_word5
                and "#" not in target_word6
                and "[" not in target_word1
                and "[" not in target_word2
                and "[" not in target_word3
                and "[" not in target_word4
                and "[" not in target_word5
                and "[" not in target_word6
            ):
                right_target = True
        print(
            "Target word1, word2, word3, word4, word5, word6:",
            target_word1,
            target_word2,
            target_word3,
            target_word4,
            target_word5,
            target_word6,
        )
        return (
            target_word1,
            target_word2,
            target_word3,
            target_word4,
            target_word5,
            target_word6,
        )


def update_target(target_word, prompt, mask_type, suffix=None, mask_position_ids=None):
    """Render the textual label by injecting target words into the prompt.

    Args:
        target_word: Word or tuple of words to insert.
        prompt: Base prompt string.
        mask_type: "single_mask" or "multi_mask".
        suffix: Optional suffix appended in single-mask mode.
        mask_position_ids: Unused placeholder for backwards compatibility.

    Returns:
        Full label string used for conditioning.
    """

    assert mask_type in ["single_mask", "multi_mask"], "Mask type not supported"
    assert type(prompt) == str, "Prompt must be a string"

    if mask_type == "multi_mask":
        label = (
            prompt
            + " "
            + target_word[0]
            + " with "
            + target_word[1]
            + " and "
            + target_word[2]
            + " and "
            + target_word[3]
            + " and "
            + target_word[4]
            + " and "
            + target_word[5]
            + "."
        )
    else:
        if suffix is not None:
            label = prompt + " " + target_word + " " + suffix + "."
        else:
            label = prompt + " " + target_word + "."

    return label


def change_target(loss, pred_word, loss_num):
    """Update the target word when a new loss minimum is found.

    Args:
        loss: Current loss tensor for the word.
        pred_word: Newly predicted word(s) to consider.
        loss_num: 1-based index of the word slot being updated.

    Returns:
        Tuple (history_list, new_target_word).
    """

    def _format_word(word):
        if isinstance(word, (list, tuple)):
            word = " ".join(str(w).strip() for w in word if str(w).strip())
        word = str(word).strip()
        return word if word else "<none>"

    best_loss = [loss.item()]
    target_word = pred_word
    print("\n")
    print(
        f"New best loss {loss_num}: {torch.mean(torch.tensor(best_loss)):.4f}"
    )
    print(f"New target word {loss_num}: {_format_word(target_word)}")

    return best_loss, target_word
