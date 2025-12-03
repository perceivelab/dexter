"""Text-side helpers to bridge BERT masked language modeling and CLIP prompts."""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, CLIPTokenizer

from ..models.bert import BertMasked
from ..models.clip_text_encoder import CLIPTextEncoder


class TranslationMatrix(nn.Module):
    """Maps overlapping vocabulary entries between CLIP and BERT tokenizers."""

    def __init__(self, clip_checkpoint, bert_checkpoint):
        super(TranslationMatrix, self).__init__()
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_checkpoint)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)

        self.common_words = self.match()
        self.translation_matrix = torch.zeros(
            self.clip_tokenizer.vocab_size, self.bert_tokenizer.vocab_size
        )
        self.fill_translation_matrix()

    def match(self):
        """Collect shared tokens across CLIP and BERT vocabularies.

        Returns:
            List of token strings present in both vocabularies.
        """
        clip_tokens = self.clip_tokenizer.get_vocab().keys()
        bert_tokens = self.bert_tokenizer.get_vocab().keys()

        clip_tokens = [key for key in clip_tokens if key.endswith("</w>")]
        keys_cleaned = [key.replace("</w>", "") for key in clip_tokens]
        match = []

        for word in tqdm(bert_tokens, desc="Matching Tokens"):
            if word in keys_cleaned:
                match.append(word)

        return match

    def fill_translation_matrix(self):
        """Populate a one-hot translation matrix for the matched tokens."""
        for word in tqdm(self.common_words, desc="Filling  Matrix"):
            try:
                clip_id = self.clip_tokenizer.get_vocab()[word + "</w>"]
            except KeyError:
                clip_id = self.clip_tokenizer.get_vocab()[word]

            bert_id = self.bert_tokenizer.get_vocab()[word]
            self.translation_matrix[clip_id, bert_id] = 1


class TextPipeline(nn.Module):
    """Predict masked tokens with BERT and translate them into CLIP embeddings."""

    def __init__(
        self,
        clip_checkpoint="openai/clip-vit-large-patch14",
        bert_checkpoint="google-bert/bert-base-uncased",
        translation_matrix=None,
        n_soft_prompt=1,
    ):
        super(TextPipeline, self).__init__()
        self.bert = BertMasked(checkpoint=bert_checkpoint, n_soft_prompt=n_soft_prompt)
        self.clip = CLIPTextEncoder(clip_checkpoint)

        if translation_matrix is None:
            translation_matrix = TranslationMatrix(
                clip_checkpoint, bert_checkpoint
            ).translation_matrix

            # Keep in sync with utils.load_translation_matrix (repo-level checkpoints).
            checkpoints_dir = Path(__file__).resolve().parents[2] / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            torch.save(
                translation_matrix,
                checkpoints_dir / "translation_matrix.pt",
            )

        self.translation_matrix = translation_matrix
        self.translation_matrix = self.translation_matrix.detach().clone()
        self.translation_matrix = nn.Parameter(
            self.translation_matrix, requires_grad=True
        )

    @property
    def soft_prompt(self):
        return self.bert.model.bert.soft_prompt

    @property
    def prompt_parameters(self):
        return [self.soft_prompt]

    def build_masked_prompt(self, prompt, mask_type, suffix=None):
        """Insert the correct number of `[MASK]` tokens based on the selected scheme.

        Args:
            prompt: Base prompt text (e.g., "a picture of a").
            mask_type: "single_mask" or "multi_mask".
            suffix: Optional suffix appended after the mask in single-mask mode.

        Returns:
            String with mask tokens inserted.
        """
        if mask_type == "single_mask":
            if suffix is not None:
                return f"{prompt} [MASK] {suffix}."
            return f"{prompt} [MASK]."
        if mask_type == "multi_mask":
            return f"{prompt} [MASK] with [MASK] and [MASK] and [MASK] and [MASK] and [MASK]."
        raise ValueError("Mask Type not supported")

    def forward(
        self,
        prompt,
        label,
        tau=1,
        mask_type="single_mask",
        target_words=None,
        inference=False,
        suffix=None,
    ):
        """Predict target words and return translated CLIP embeddings plus losses.

        Args:
            prompt: Human-readable prompt prefix used for CLIP conditioning.
            label: Full label string fed to BERT for masked LM training.
            tau: Temperature for the softmax over masked tokens.
            mask_type: "single_mask" or "multi_mask".
            target_words: Optional target tokens for logging/comparison.
            inference: Whether to skip translation and reuse existing embeddings.
            suffix: Optional suffix appended in single-mask mode.

        Returns:
            Tuple (clip_emb, masked_loss, pred_word):
                clip_emb: CLIP embeddings padded to length 77.
                masked_loss: BERT masked LM loss tensor.
                pred_word: List/tuple of predicted tokens.
        """
        if mask_type not in ("single_mask", "multi_mask"):
            raise ValueError("Mask Type not supported")

        bert_prompt = self.build_masked_prompt(prompt, mask_type, suffix)
        print("BERT Prompt: ", bert_prompt)
        predicted_token, pred_word, masked_loss = self.bert(bert_prompt, label, tau=tau)
        init_prompt = prompt

        predicted_emb = torch.matmul(predicted_token, self.translation_matrix.T)
        clip_emb = self.clip(
            predicted_emb.unsqueeze(0),
            init_prompt=init_prompt,
            mask_type=mask_type,
            inference=inference,
            suffix=suffix,
        )

        max_len = 77
        clip_emb = torch.cat(
            [
                clip_emb,
                clip_emb[:, -1, :].repeat(max_len - clip_emb.size(1), 1).unsqueeze(0),
            ],
            dim=1,
        )

        return clip_emb, masked_loss, pred_word


# Backwards compatibility for existing imports
PromptTextEncoder = TextPipeline
MyModel = TextPipeline
