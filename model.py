import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
    batch_size = len(batch[0])
    seed = batch[0]
    text = [' '.join([tokenizer.mask_token] * generation_max_len)] * batch_size

    # note that the input_embeddings must be truncated manually since there is no option to truncate on left side.
    # Passing truncation=True, will truncate on the right-side of the dialog context removing the most recent utterances
    input_embeddings = tokenizer(seed, text, return_tensors='pt', padding=True, return_token_type_ids = True)
    input_ids = input_embeddings['input_ids'][:, -tokenizer.model_max_length:]      # truncate manually removing tokens from left
    input_ids[input_ids[:, 0] != 1, 0] = tokenizer.bos_token_id                     # check lines that overflow (first token is not pad) and add bos token

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


def finetune(model, tokenizer, dataloader, generation_max_len=40, num_iter=150):
    device = 'cpu'
    batch_size = dl.batch_size

    for batch in dataloader:
        input_ids, attention_mask, token_type_ids, target = init_batch(tokenizer, batch, generation_max_len=generation_max_len)

        input_ids.to(device)
        attention_mask.to(device)
        token_type_ids.to(device)
        target.to(device)

        # at first iteration start with the first token with probability 1
        p = torch.tensor([[1] + [0] * (generation_max_len - 1)] * batch_size)
        num_tokens = input_ids.shape[1]

        for i in range(num_iter):
            # 1. Select tokens to mask
            idx_to_mask = [num_tokens - generation_max_len - 1 + np.random.choice(range(generation_max_len), p=p_i)  for p_i in p]

            # 2. Mask tokens selected
            old_token_ids = input_ids[np.arange(batch_size), idx_to_mask].numpy()
            input_ids[np.arange(batch_size), idx_to_mask] = tokenizer.mask_token_id

            # 3. Pass through the model
            out = model(input_ids, attention_mask)

            # 4. Retrieve logits
            logits = out['logits']

            # 4.1 Compute loss w.r.t. ground truth token

            # 4.2 Predict the token to replace and replace it (?) for next iteration
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)


if __name__ == '__main__':
    textgenerator = MaskedLMGenerator('bert-base-uncased')
    tokenizer = textgenerator.tokenizer

    from dataset import MultiWOZDataset
    m = MultiWOZDataset(tokenizer, 'data/val/logs.json', 'data/val/labels.json', 'data/knowledge.json')

    from torch.utils.data import DataLoader
    dl = DataLoader(m)

