import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MaskedLMGenerator:
    def __init__(self, model_name, device=DEFAULT_DEVICE,  use_fast=True, do_basic_tokenize=True):
        self.device = device
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')



def init_batch(tokenizer, batch, generation_max_len=40):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    pad_token_id = tokenizer.pad_token_id

    batch_size = len(batch[0])
    seed = batch[0]
    text = [' '.join([tokenizer.mask_token] * generation_max_len)] * batch_size

    # note that the input_embeddings must be truncated manually since there is no option to truncate on left side.
    # Passing truncation=True, will truncate on the right-side of the dialog context removing the most recent utterances
    # todo suppress warning for exceding tokens
    input_embeddings = tokenizer(seed, text, return_tensors='pt', padding=True, return_token_type_ids = True)
    input_ids = input_embeddings['input_ids'][:, -tokenizer.model_max_length:]          # truncate manually removing tokens from left
    input_ids[input_ids[:, 0] != pad_token_id, 0] = bos_token_id                        # check lines that overflow (first token is not pad) and add bos token

    attention_mask = input_embeddings['attention_mask'][:, -tokenizer.model_max_length:]
    token_type_ids = input_embeddings['token_type_ids'][:, -tokenizer.model_max_length:]


    target = []
    for response in batch[1]:
        response_tokens = tokenizer.encode(response, add_special_tokens=False)[:generation_max_len]
        response_tokens += [tokenizer.pad_token_id] * (generation_max_len - len(response_tokens))
        target.append(response_tokens)

    target = torch.tensor(target)

    #todo change to return only input_embeddings directly
    return input_ids, attention_mask, token_type_ids, target


def finetune(model, tokenizer, dataloader, generation_max_len=40, num_epochs=8, num_iter=150):
    device = 'cpu'
    batch_size = dl.batch_size
    total_steps = len(dl) * num_epochs * num_iter
    # optimzer = AdamW()
    # scheduler = get_linear_schedule_with_warmup()

    for epoch in num_epochs:
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, response = init_batch(tokenizer, batch, generation_max_len=generation_max_len)

            input_ids.to(device)
            attention_mask.to(device)
            token_type_ids.to(device)
            response.to(device)

            # at first iteration start with the first token with probability 1
            p = torch.tensor([[1] + [0] * (generation_max_len - 1)] * batch_size)
            num_tokens = input_ids.shape[1]
            context_offset = num_tokens - generation_max_len - 1

            for i in range(num_iter):
                model.zero_grad()

                # 1. Select tokens to mask
                idx_to_mask = np.array([np.random.choice(range(generation_max_len), p=p_i)  for p_i in p])

                # 2. Mask tokens selected
                # old_token_ids = input_ids[np.arange(batch_size), idx_to_mask].numpy()
                input_ids[np.arange(batch_size), idx_to_mask + context_offset] = tokenizer.mask_token_id

                # 3. Pass through the model
                out = model(input_ids, attention_mask)

                # 4Retrieve logits of the masked tokens
                logits = out['logits'][np.arange(batch_size), idx_to_mask + context_offset]
                del out

                # Retrieve target tokens from ground-truth:
                target = response[np.arange(batch_size), idx_to_mask]

                # 4.1 Compute loss w.r.t. ground truth token
                loss = F.cross_entropy(logits, target)

                # 4. Replacement
                dist = torch.distributions.categorical.Categorical(logits=logits)
                replacing_ids = dist.sample().squeeze(-1)
                input_ids[np.arange(batch_size), idx_to_mask] = replacing_ids


if __name__ == '__main__':
    textgenerator = MaskedLMGenerator('bert-base-uncased')
    tokenizer = textgenerator.tokenizer
    model = textgenerator.model
    from dataset import MultiWOZDataset
    m = MultiWOZDataset(tokenizer, 'data/val/logs.json', 'data/val/labels.json', 'data/knowledge.json')

    from torch.utils.data import DataLoader
    dl = DataLoader(m, batch_size=8)
    batch = next(iter(dl))
    num_epochs = 8
    num_iter = 150
    generation_max_len = 40
