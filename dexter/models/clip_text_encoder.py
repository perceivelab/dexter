"""CLIP text encoder wrapper that accepts translated BERT token embeddings."""

import importlib.util
import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling


def _patch_transformers_clip():
    """Load local modeling_clip.py in place of the transformers implementation."""
    local_model_path = Path(__file__).resolve().parent / "modeling_clip.py"
    spec = importlib.util.spec_from_file_location(
        "transformers.models.clip.modeling_clip", local_model_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local modeling_clip from {local_model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformers.models.clip.modeling_clip"] = module
    spec.loader.exec_module(module)


_patch_transformers_clip()


class CLIPTextEncoder(nn.Module):
    """Build CLIP text embeddings from raw text or translated BERT token logits."""

    def __init__(self, checkpoint: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(checkpoint)
        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint)
        self.sos_clip = (
            self.model.text_model.embeddings.token_embedding(
                torch.tensor([self.tokenizer.bos_token_id])
            )
            .unsqueeze(0)
            .to("cuda")
        )
        self.eos_clip = (
            self.model.text_model.embeddings.token_embedding(
                torch.tensor([self.tokenizer.eos_token_id])
            )
            .unsqueeze(0)
            .to("cuda")
        )

        self.sos_clip = nn.Parameter(self.sos_clip.detach().clone(), requires_grad=True)
        self.eos_clip = nn.Parameter(self.eos_clip.detach().clone(), requires_grad=True)

        self.soft_prompt = nn.Parameter(
            torch.randn(1, 1, self.model.text_model.config.hidden_size),
            requires_grad=True,
        )

    def _tokenize_and_embed(
        self, prompt: str, input_embs: Optional[torch.Tensor], inference: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tokenize prompt text and optionally project BERT outputs into CLIP space."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_emb = self.model.text_model.embeddings.token_embedding(
            inputs["input_ids"].to("cuda")
        )
        if inference:
            return prompt_emb, None

        prompt_emb = prompt_emb[:, 1:-1]
        emb = torch.matmul(
            input_embs, self.model.text_model.embeddings.token_embedding.weight
        )
        return prompt_emb, emb

    def _compose_prompt(
        self,
        prompt_emb: torch.Tensor,
        emb: Optional[torch.Tensor],
        mask_type: str,
        suffix: Optional[str],
        inference: bool,
    ) -> torch.Tensor:
        if inference:
            return prompt_emb

        if mask_type == "multi_mask":
            parts = [" with ", " and ", " and ", " and ", " and "]
            tokenized = [self.tokenizer(p, return_tensors="pt") for p in parts]
            prompt_embs = [
                self.model.text_model.embeddings.token_embedding(
                    t["input_ids"].to("cuda")
                )[:, 1:-1]
                for t in tokenized
            ]

            return torch.cat(
                [
                    self.sos_clip,
                    prompt_emb,
                    emb[:, :1, :],
                    prompt_embs[0],
                    emb[:, 1:2, :],
                    prompt_embs[1],
                    emb[:, 2:3, :],
                    prompt_embs[2],
                    emb[:, 3:4, :],
                    prompt_embs[3],
                    emb[:, 4:5, :],
                    prompt_embs[4],
                    emb[:, 5:, :],
                    self.eos_clip,
                ],
                dim=1,
            )

        if suffix is not None:
            prompt2 = suffix
            inputs2 = self.tokenizer(prompt2, return_tensors="pt")
            prompt_emb2 = self.model.text_model.embeddings.token_embedding(
                inputs2["input_ids"].to("cuda")
            )
            prompt_emb2 = prompt_emb2[:, 1:-1]
            return torch.cat(
                [self.sos_clip, prompt_emb, emb, prompt_emb2, self.eos_clip], dim=1
            )

        return torch.cat([self.sos_clip, prompt_emb, emb, self.eos_clip], dim=1)

    def forward(
        self,
        input_embs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        init_prompt: str = "a picture of a",
        suffix: Optional[str] = None,
        mask_type: str = "single_mask",
        inference: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """Compose a CLIP-ready embedding sequence from translated tokens and run it through the encoder."""
        if mask_type not in ("single_mask", "multi_mask"):
            raise ValueError("Mask type not supported")
        if input_embs is None:
            raise ValueError("You have to specify input_embs")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.text_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.text_model.config.output_hidden_states
        )
        prompt_emb, emb = self._tokenize_and_embed(
            prompt=init_prompt, input_embs=input_embs, inference=inference
        )

        input_embs = self._compose_prompt(
            prompt_emb=prompt_emb,
            emb=emb,
            mask_type=mask_type,
            suffix=suffix,
            inference=inference,
        )

        input_shape = input_embs.size()[:-1]
        hidden_states = self.model.text_model.embeddings(
            inputs_embeds=input_embs, position_ids=position_ids
        )

        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )

        encoder_outputs = self.model.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.model.text_model.final_layer_norm(last_hidden_state)
        return last_hidden_state
