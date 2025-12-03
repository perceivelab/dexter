"""Core DEXTER orchestration logic for prompt optimization and analysis."""

import logging
import os
from statistics import mean
from typing import Optional

import torch
from PIL import Image
from sentence_transformers.util import dot_score, normalize_embeddings, semantic_search
from tqdm.auto import tqdm

from .pipelines import VisualPipeline, TextPipeline
from .config import DISALLOWED_TARGET_CHARS, RunConfig
from .utils import (
    change_target,
    ensure_output_dir,
    init_diffusion_pipeline,
    initialize_dynamic_target,
    latents_to_pil,
    latents_to_torch_img,
    load_translation_matrix,
    select_top_features,
    set_random_seed,
    text_enc,
    forward_with_features,
    update_target,
    validate_options,
)

# Silence noisy framework logging in the CLI.
logging.disable(logging.WARNING)


class DEXTER:
    """Orchestrates the prompt optimization and classifier analysis workflow."""

    def __init__(self, config: RunConfig, device: str = "cuda") -> None:
        """Set up diffusion/text pipelines, output folders, and bookkeeping."""
        self.config = config
        self.device = device
        self.feat_to_activate = select_top_features(
            config.classifier, config.class_to_activate
        )

        set_random_seed(config.seed)

        self.pipe, self.tokenizer, self.clip_orig_text_encoder = (
            init_diffusion_pipeline(device)
        )

        try:
            self.translation_matrix = load_translation_matrix()
        except Exception as e:
            print(f"Warning: Translation Matrix not found, it will be created")
            self.translation_matrix = None

        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.unet = self.pipe.unet
        self.visual_pipeline = VisualPipeline(
            self.scheduler, self.unet, self.vae, device=device
        )

        self.out_folder = ensure_output_dir(
            config.outdir, config.class_to_activate, config.to_kwargs()
        )
        validate_options(config.sp_init, config.no_match_uniform, config.mask_type)

        self.frame_folder = os.path.join(self.out_folder, "frames")
        os.makedirs(self.frame_folder, exist_ok=True)

        self.previous_target_words = []

    @staticmethod
    def _format_words(words) -> str:
        """Readable string for optional/iterable word collections."""
        if words in (None, ""):
            return "-"
        if isinstance(words, (list, tuple, set)):
            if len(words) == 0:
                return "-"
            return ", ".join(str(w) for w in words)
        return str(words)

    @staticmethod
    def _print_section(title: str) -> None:
        """Pretty-print a section header for CLI logging."""
        line = "=" * 60
        print(f"\n{line}\n{title}\n{line}")

    @staticmethod
    def _print_subsection(title: str) -> None:
        """Pretty-print a subsection header for CLI logging."""
        line = "-" * 50
        print(f"\n{line}\n{title}\n{line}")

    @staticmethod
    def _print_line(label: str, value) -> None:
        """Consistent, padded key/value output."""
        print(f"{label:<18} {value}")

    def analyze_classifier(
        self, target_classifier, transforms: Optional[torch.nn.Module] = None
    ) -> str:
        """Optimize text prompts against the classifier and persist artifacts.

        Args:
            target_classifier: Torch vision model used for scoring generated images.
            transforms: Optional preprocessing applied before classifier forward.

        Returns:
            Path to the run output folder containing frames, prompts, and logs.
        """
        cfg = self.config
        feat_to_activate = self.feat_to_activate
        device = self.device

        dummy_prompt = "X " * cfg.n_tokens
        dummy_prompt = dummy_prompt.strip()

        if cfg.prefix not in ["", None]:
            prompts = [cfg.prefix + " " + dummy_prompt]
        else:
            prompts = [dummy_prompt]

        dim = 512
        g = 1.0

        captions = []
        convergence = 0

        for outer_idx in range(cfg.topk):

            self.text_encoder = TextPipeline(translation_matrix=self.translation_matrix, n_soft_prompt=self.config.n_soft_prompt)
            self.text_encoder = self.text_encoder.to(device).half()
            self.optimizer = torch.optim.AdamW(self.text_encoder.prompt_parameters, lr=self.config.lr, eps=1e-4 )

            for inner_optim_step in range(cfg.optim_steps):
                self._print_subsection(
                    f"Step {inner_optim_step + 1}/{cfg.optim_steps}"
                )
                if len(captions) == 0:
                    prompt = "A picture of a"

                if inner_optim_step == 0:
                    target_words = initialize_dynamic_target(cfg.mask_type)

                label = update_target(target_words, prompt, cfg.mask_type)

                self._print_line("Target prompt:", label)
                text = text_enc(prompts, self.tokenizer, self.clip_orig_text_encoder, device=device)
                pred_words = [""] * 6

                cond, masked_loss, pred_word = self.text_encoder(
                    prompt=prompt,
                    label=label,
                    tau=cfg.tau,
                    mask_type=cfg.mask_type,
                    target_words=target_words,
                )

                pred_words = pred_word
                self._print_line("Predicted words:", self._format_words(pred_word))
                uncond = text_enc(
                    [""],
                    self.tokenizer,
                    self.clip_orig_text_encoder,
                    text.shape[1],
                    device=device,
                )
                latents, img = self.visual_pipeline.generate(
                    cond, uncond, cfg, dim=dim, guidance_scale=g
                )
                if cfg.use_tfms and transforms is not None:
                    img = transforms(img)
                output, features = forward_with_features(target_classifier, img)
                if cfg.mask_type == "multi_mask" and features is None:
                    raise RuntimeError(
                        "Multi-mask mode requires feature tensors, "
                        "but the classifier did not return any."
                    )

                ce_loss = torch.nn.CrossEntropyLoss()(
                    output,
                    torch.tensor([cfg.class_to_activate] * img.shape[0]).to(device),
                )
                predicted_class = torch.argmax(output, dim=1).cpu().numpy()

                if predicted_class[0] == cfg.class_to_activate:
                    pil_image = latents_to_pil(latents.detach(), self.vae)[0]
                    pil_image.save(os.path.join(self.frame_folder, f"{inner_optim_step}.png"))

                max_losses = []
                if cfg.mask_type == "multi_mask":
                    if cfg.classifier == "vit_b_16":
                        feature_slices = [features[:, :, idx] for idx in feat_to_activate]
                    else:
                        feature_slices = [features[:, idx] for idx in feat_to_activate]

                    max_losses = [-feat_slice.mean() for feat_slice in feature_slices]

                if inner_optim_step == 0:
                    if cfg.mask_type == "multi_mask":
                        best_loss = [[ce_loss.item()]] + [[loss.item()] for loss in max_losses]
                    else:
                        best_loss = [ce_loss.item()]

                if cfg.mask_type == "multi_mask":
                    self._print_line("Masked loss:", f"{masked_loss.item():.4f}")
                    self._print_line("Class loss:", f"{ce_loss.item():.4f}")
                    feature_losses = " | ".join(
                        f"F{idx}={m_loss.item():.4f}"
                        for idx, m_loss in enumerate(max_losses, 1)
                    )
                    best_losses = " | ".join(
                        f"B{idx}={torch.mean(torch.tensor(b_loss)):.4f}"
                        for idx, b_loss in enumerate(best_loss, 1)
                    )
                    self._print_line("Feature losses:", feature_losses)
                    self._print_line("Best losses:", best_losses)
                else:
                    self._print_line("Masked loss:", f"{masked_loss.item():.4f}")
                    self._print_line("Class loss:", f"{ce_loss.item():.4f}")
                    self._print_line(
                        "Best loss:", f"{torch.mean(torch.tensor(best_loss)):.4f}"
                    )

                if cfg.mask_type == "multi_mask":
                    loss = ce_loss + sum(max_losses) + masked_loss
                else:
                    loss = ce_loss + masked_loss

                self._print_line(
                    "Step summary:",
                    f"loss={loss.item():.4f} | pred class={predicted_class}",
                )

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if cfg.mask_type == "multi_mask":
                    target_words = list(target_words)

                    (
                        ce_history,
                        f1_history,
                        f2_history,
                        f3_history,
                        f4_history,
                        f5_history,
                    ) = best_loss

                    ce_loss_improved = ce_loss.item() < torch.mean(
                        torch.tensor(ce_history)
                    )
                    loss_improvements = [
                        max_losses[0].item() < torch.mean(torch.tensor(f1_history))
                        and ce_loss_improved,
                        max_losses[1].item() < torch.mean(torch.tensor(f2_history))
                        and ce_loss_improved,
                        max_losses[2].item() < torch.mean(torch.tensor(f3_history))
                        and ce_loss_improved,
                        max_losses[3].item() < torch.mean(torch.tensor(f4_history))
                        and ce_loss_improved,
                        max_losses[4].item() < torch.mean(torch.tensor(f5_history))
                        and ce_loss_improved,
                    ]

                    img_torch = latents_to_torch_img(latents, self.vae)
                    img_torch, nsfw_detected = self.pipe.run_safety_checker(
                        img_torch.detach(), device, "cuda"
                    )
                    img_torch = img_torch[0]
                    nsfw_detected = nsfw_detected[0]

                    loss_vals = [ce_loss] + list(max_losses)
                    improvements = [ce_loss_improved] + loss_improvements
                    pred_sequence_seen = tuple(pred_word) in self.previous_target_words

                    for idx, (loss_val, improved) in enumerate(
                        zip(loss_vals, improvements)
                    ):
                        feature_pred = pred_word[idx]
                        target_word = target_words[idx]

                        if feature_pred == target_word:
                            best_loss[idx].append(loss_val.item())
                            continue

                        if not improved:
                            continue

                        invalid_feature = any(
                            char in feature_pred for char in DISALLOWED_TARGET_CHARS
                        )
                        if invalid_feature or pred_sequence_seen:
                            continue

                        history, target_word = change_target(
                            loss_val, feature_pred, idx + 1
                        )
                        # Persist both the refreshed baseline and the new target word.
                        best_loss[idx] = history
                        target_words[idx] = target_word

                    target_words = tuple(target_words)
                    pred_words = tuple(pred_word[:6])

                elif cfg.mask_type == "single_mask":
                    pred_token = pred_word[0] if len(pred_word) > 0 else ""
                    invalid_feature = any(
                        char in pred_token for char in DISALLOWED_TARGET_CHARS
                    )

                    all_words_path = os.path.join(self.out_folder, "all_words.txt")
                    with open(
                        all_words_path, "a" if os.path.exists(all_words_path) else "w"
                    ) as f:
                        f.write(f"{pred_token}\n")

                    if pred_token == target_words:
                        best_loss.append(ce_loss.item())
                    elif ce_loss.item() < torch.mean(torch.tensor(best_loss)):
                        if (
                            not invalid_feature
                            and pred_token not in self.previous_target_words
                        ):
                            best_loss = [ce_loss.item()]
                            target_words = pred_token

                            words_path = os.path.join(self.out_folder, "words.txt")
                            with open(
                                words_path, "a" if os.path.exists(words_path) else "w"
                            ) as f:
                                f.write(f"{target_words}\n")

                if (
                    cfg.mask_type == "multi_mask"
                    and predicted_class[0] == cfg.class_to_activate
                ):
                    with open(
                        os.path.join(self.out_folder, f"pred_words.txt"), "a"
                    ) as f:
                        f.write(f"{pred_words}\n")

                img_torch = latents_to_torch_img(latents, self.vae)
                img_torch, nsfw_detected = self.pipe.run_safety_checker(
                    img_torch.detach(), device, "cuda"
                )
                img_torch = img_torch[0]
                nsfw_detected = nsfw_detected[0]

                if (
                    predicted_class[0] == cfg.class_to_activate
                    and cfg.mask_type == "single_mask"
                ):
                    target_prob_value = torch.nn.functional.softmax(output, dim=1)[
                        :, cfg.class_to_activate
                    ]
                    target_output = output[:, cfg.class_to_activate]
                    best_words_path = os.path.join(
                        self.out_folder, "best_words.txt"
                    )
                    with open(
                        best_words_path,
                        "a" if os.path.exists(best_words_path) else "w",
                    ) as f:
                        f.write(
                            f"{target_words};{target_prob_value.item()};{target_output.item()}\n"
                        )

                if (
                    predicted_class[0] == cfg.class_to_activate
                    and cfg.mask_type == "multi_mask"
                ):
                    saved_prompt = (
                        f" a picture of a {pred_word[0]} with {pred_word[1]} and {pred_word[2]} "
                        f"and {pred_word[3]} and {pred_word[4]} and {pred_word[5]}."
                    )
                    new_target_prompt = (
                        f" a picture of a {target_words[0]} with {target_words[1]} and {target_words[2]} "
                        f"and {target_words[3]} and {target_words[4]} and {target_words[5]}."
                    )

                    if os.path.exists(
                        os.path.join(
                            self.out_folder, f"{cfg.class_to_activate}_prompt.txt"
                        )
                    ):
                        with open(
                            os.path.join(
                                self.out_folder, f"{cfg.class_to_activate}_prompt.txt"
                            ),
                            "a",
                        ) as f:
                            f.write(f"Step prompt:  {saved_prompt}")
                            f.write("\n")
                            f.write(f"Target prompt: {new_target_prompt}")
                            f.write("\n")
                            f.write("*" * 50)
                            f.write("\n")
                    else:
                        with open(
                            os.path.join(
                                self.out_folder, f"{cfg.class_to_activate}_prompt.txt"
                            ),
                            "w",
                        ) as f:
                            f.write(f"Step prompt:  {saved_prompt}")
                            f.write("\n")
                            f.write(f"Target prompt: {new_target_prompt}")
                            f.write("\n")
                            f.write("*" * 50)
                            f.write("\n")

                if len(captions) == 50:
                    break

                if predicted_class[0] == cfg.class_to_activate:
                    convergence += 1

                    if convergence == 3:
                        print("Convergence reached at step", inner_optim_step)
                        break
                else:
                    convergence = 0

            if cfg.save_images and predicted_class[0] == cfg.class_to_activate:
                img = latents_to_pil(latents, self.vae)
                img[0].save(os.path.join(self.out_folder, "final_img.png"))

            self.previous_target_words.append(target_words)

        return self.out_folder
