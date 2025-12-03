"""Local BERT wrapper that supports soft prompts and custom translation logic."""

import importlib.util
import sys
from pathlib import Path
from torch import nn
import torch
from typing import Optional, Tuple, Union, List


def _patch_transformers_bert():
    """Load local modeling_bert.py in place of the transformers implementation."""
    local_model_path = Path(__file__).resolve().parent / "modeling_bert.py"
    spec = importlib.util.spec_from_file_location(
        "transformers.models.bert.modeling_bert", local_model_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local modeling_bert from {local_model_path}")
    module = importlib.util.module_from_spec(spec)
    # Register module before execution so downstream lookups during class creation succeed.
    sys.modules["transformers.models.bert.modeling_bert"] = module
    spec.loader.exec_module(module)


_patch_transformers_bert()

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class Bert(BertModel):
    def __init__(self, config=None, sp_init="none", n_sp=1):
        if config is None:
            config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        super().__init__(config)
        self.n_sp = n_sp
        self._init_prompts(sp_init)

    def _init_prompts(self, sp_init="none"):
        if sp_init == "none":
            self.soft_prompt = torch.randn(1, self.n_sp, self.config.hidden_size)
            nn.init.uniform_(self.soft_prompt, -1, 1)

        self.soft_prompt = self.soft_prompt.detach().clone().cuda().half()
        self.soft_prompt.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        use_cache = use_cache if self.config.is_decoder else False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((batch_size, self.soft_prompt.shape[1]), device=device),
            ],
            dim=1,
        )

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        encoder_extended_attention_mask = None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        embedding_output = torch.cat([self.soft_prompt, embedding_output], dim=1)
        embedding_output = self.embeddings(inputs_embeds=embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertMasked(nn.Module):
    def __init__(
        self, checkpoint: str = "google-bert/bert-base-uncased", n_soft_prompt: int = 1
    ):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.model.bert = Bert.from_pretrained(checkpoint, n_sp=n_soft_prompt)
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint)

    def forward(self, text, label, tau):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        label = self.tokenizer(label, return_tensors="pt").to("cuda")

        out = self.model(**inputs, labels=label["input_ids"], output_attentions=True)
        logits = out.logits
        masked_loss = out.loss

        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        predicted_token = logits[0, mask_token_index]
        predicted_token = torch.nn.functional.gumbel_softmax(
            predicted_token, tau=tau, hard=True
        )
        predicted_token_id = torch.argmax(predicted_token, dim=-1)

        text_tokens = []
        for i in range(predicted_token_id.shape[0]):
            pred_token = predicted_token_id[i].item()
            token = self.tokenizer.decode([pred_token], skip_special_tokens=True)
            text_tokens.append(token)

        return predicted_token, text_tokens, masked_loss
