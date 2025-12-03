"""Configuration objects and constants shared across the DEXTER pipeline."""

from dataclasses import asdict, dataclass
from typing import Optional

# Characters that invalidate a generated target word.
DISALLOWED_TARGET_CHARS = {"#", "[", ".", "\\", "/"}


@dataclass
class RunConfig:
    """Typed container for the main optimization run parameters."""

    prefix: str = ""
    n_tokens: int = 1
    approx_opt_prompt: bool = False
    nn_loss: bool = False
    init_mode: str = ""
    class_to_activate: int = 416
    seed: Optional[int] = None
    optim_steps: int = 5000
    diff_steps: int = 4
    lr: float = 0.1
    outdir: str = "out"
    run_idx: int = 0
    classifier: str = "robust50"
    bs: int = 1
    use_tfms: bool = True
    stop_after: int = 20
    sp_init: str = "none"
    no_match_uniform: str = "none"
    tau: float = 1.0
    save_images: bool = True
    n_soft_prompt: int = 1
    mask_type: str = "multi_mask"
    topk: int = 1
    feat_idx: int = 0
    vlm_id: str = "llava:13b"
    llm_id: str = "gpt-oss:120b"
    api_key: str = ""
    base_url: str = ""
    system_prompt_config_path: str = "sp_config"

    def to_kwargs(self):
        """Return config as a plain dict for JSON dumps or logging."""
        return asdict(self)
