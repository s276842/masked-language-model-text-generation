import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np


DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MaskedLMGenerator(torch.nn.Module):
    def __init__(self, model_name: str, device: str =DEFAULT_DEVICE,  use_fast=True, do_basic_tokenize=True):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')


    #todo fix return_attention
    def forward(self, input_embeddings, p=None, generation_max_len: int =40, context_offset: int =0, return_attention: bool =True):

        batch_size = input_embeddings['input_ids'].shape[0]

        if p is None:
            p = torch.tensor([[1/generation_max_len] * generation_max_len] * batch_size)

        # 1. Sample position
        dist = torch.distributions.categorical.Categorical(probs=p)
        idx_to_mask = dist.sample()

        # 2. Mask tokens
        #todo fix context_offset to consider +1 for CLS token (or <s>)
        input_embeddings['input_ids'][np.arange(batch_size), idx_to_mask + context_offset] = self.tokenizer.mask_token_id

        # 3. Generate predictions
        out = self.model(**input_embeddings)
        logits = out['logits'][torch.arange(batch_size), idx_to_mask + context_offset]

        attention = torch.stack(out['attentions'])[:, :, :, idx_to_mask + context_offset, :]

        return logits, idx_to_mask, attention



