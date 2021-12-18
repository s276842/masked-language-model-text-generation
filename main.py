from model import MaskedLMGenerator
from dataset import MultiWOZDataset

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

use_apex = False
try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def truncate_lef_input_embeddings(tokenizer, input_embeddings):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    pad_token_id = tokenizer.pad_token_id

    input_ids = input_embeddings['input_ids'][:,-tokenizer.model_max_length:]
    input_ids[input_ids[:,0] != pad_token_id, 0] = bos_token_id
    input_embeddings['input_ids'] = input_ids

    input_embeddings['attention_mask'] = input_embeddings['attention_mask'][:, -tokenizer.model_max_length:]
    input_embeddings['token_type_ids'] = input_embeddings['token_type_ids'][:, -tokenizer.model_max_length:]

    return input_embeddings


def init_batch(tokenizer, batch, generation_max_len=40):

    batch_size = len(batch[0])
    seed = batch[0]
    text = [' '.join([tokenizer.mask_token] * generation_max_len)] * batch_size

    # note that the input_embeddings must be truncated manually since there is no option to truncate on left side.
    # Passing truncation=True, will truncate on the right-side of the dialog context removing the most recent utterances
    # todo suppress warning for exceeding tokens
    input_embeddings = tokenizer(seed, text, return_tensors='pt', padding=True, return_token_type_ids = True)
    input_embeddings = truncate_lef_input_embeddings(tokenizer, input_embeddings)

    target = []
    for response in batch[1]:
        response_tokens = tokenizer.encode(response, add_special_tokens=False)[:generation_max_len]
        response_tokens += ['<eos>']
        response_tokens += [tokenizer.pad_token_id] * (generation_max_len - len(response_tokens) - 1)
        target.append(response_tokens)

    target = torch.tensor(target)

    return input_embeddings, target

def negative_attention(attentions, counter):
    avg_attentions = attentions.cpu().mean(axis=(0, 2))





def finetune(textgenerator, dataloader, generation_max_len=40, num_epochs=8, num_iter=150, backprop_every=25,
             use_apex=False,
             device = 'cpu'):

    total_steps = len(dl) * num_epochs * num_iter
    optimizer = AdamW(textgenerator.model.parameters(), lr=2e-5, eps=1e-8)

    #todo implement nvidia apex
    if use_apex:
        device = 'cuda'
        textgenerator.model, optimizer = amp.initialize(textgenerator.model,
                                                        optimizer,
                                                        opt_level="O2",
                                                        keep_batchnorm_fp32=True,
                                                        loss_scale="dynamic")


    tokenizer = textgenerator.tokenizer
    model = textgenerator.model
    # todo defjne total steps (w.r.t num of iterations or number of batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=0)


    for epoch in num_epochs:
        for batch in dataloader:

            input_embeddings, response = init_batch(tokenizer, batch, generation_max_len=generation_max_len)
            input_embeddings = input_embeddings.to(device)
            response = response.to(device)

            batch_size, num_tokens = input_embeddings['input_ids'].shape
            context_offset = num_tokens - generation_max_len - 1

            p = torch.tensor([[1] + [0] * (num_tokens - 1)] * batch_size)
            counter = torch.ones((batch_size, generation_max_len))
            optimizer.zero_grad()

            for i in range(num_iter):
                #todo decide where to zeroing the gradients (every replacement, every n-replacements, every batch)

                # Mask and compute predictions
                logits, positions, attentions  = textgenerator(input_embeddings, p=p,
                                                               context_offset=context_offset,
                                                               generation_max_len=generation_max_len)

                counter[torch.arange(batch_size), positions] += 1

                # Retrieve target tokens from ground-truth:
                target = response[torch.arange(batch_size), positions]

                loss = F.cross_entropy(logits, target)
                if use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                # Replace tokens
                dist = torch.distributions.categorical.Categorical(logits=logits)
                tokens = dist.sample()
                input_embeddings['input_ids'][torch.arange(batch_size), positions] = tokens


                # Zero gradients, perform a backward pass, and update the weights.
                if i % backprop_every == 0:
                    # Clip the norm of the gradients to 1.0.
                    # This is to help prevent the "exploding gradients" problem.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Compute negative attention
                p = negative_attention(attentions, counter)




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
