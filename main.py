"""Command-line entry point for running DEXTER prompt optimization."""

import logging

import click

from dexter import DEXTER, RunConfig, build_classifier_and_transforms
from dexter.metrics import Metrics

# Silence noisy framework logging in the CLI.
logging.disable(logging.WARNING)


@click.command()
@click.option("--prefix", type=str, default="")
@click.option("--n_tokens", type=int, default=1)
@click.option("--approx_opt_prompt", type=bool, default=False, is_flag=True)
@click.option("--nn_loss", type=bool, default=False, is_flag=True)
@click.option("--init_mode", type=str, default="")
@click.option("--class_to_activate", type=int, default=416)
@click.option("--seed", type=int, default=None, help="random seed")
@click.option("--optim_steps", type=int, default=5000, help="number of steps")
@click.option("--diff_steps", type=int, default=4)
@click.option("--lr", type=float, default=0.1)
@click.option("--outdir", type=str, default="out")
@click.option("--run_idx", type=int, default=0)
@click.option("--classifier", type=str, default="robust50")
@click.option("--bs", type=int, default=1)
@click.option("--use_tfms", type=bool, default=True, is_flag=True)
@click.option("--stop_after", type=int, default=20)
@click.option("--sp_init", type=str, default="none")
@click.option("--no_match_uniform", type=str, default="none")
@click.option("--tau", type=float, default=1.0)
@click.option("--save_images", type=bool, default=True)
@click.option("--n_soft_prompt", type=int, default=1)
@click.option("--mask_type", type=str, default="multi_mask")
@click.option("--topk", type=int, default=1)
@click.option("--feat_idx", type=int, default=0)
@click.option("--vlm_id", type=str, default="llava:13b")
@click.option("--llm_id", type=str, default="gpt-oss:120b")
@click.option("--api_key", type=str, default="your_api_key")
@click.option("--base_url", type=str, default="http://localhost:11434/v1")
@click.option("--system_prompt_config_path", type=str, default="sp_config")

def main(**kwargs):
    """Entry point for prompt optimization."""
    config = RunConfig(**kwargs)
    device = "cuda"

    classifier, transforms = build_classifier_and_transforms(
        config.classifier, device, config.use_tfms
    )

    dexter = DEXTER(config=config, device=device)
    class_path = dexter.analyze_classifier(classifier, transforms)

    metrics = Metrics(
        class_to_activate=config.class_to_activate,
        class_path=class_path,
        classifier=classifier,
        classifier_transforms=transforms,
        vlm_id=config.vlm_id,
        llm_id=config.llm_id,
        api_key=config.api_key,
        base_url=config.base_url,
        system_prompt_config_path=config.system_prompt_config_path
    )
    metrics.compute()


if __name__ == "__main__":
    main()
