import json

from torch.utils.data import Dataset

class MultiWOZDataset(Dataset):
    def __init__(self, tokenizer, path_to_logs, path_to_labels, path_to_knowledge, max_length=40):

        self.tokenizer = tokenizer

        knowledge_base = json.load(open(path_to_knowledge))
        labels = json.load(open(path_to_labels))
        logs = json.load(open(path_to_logs))

        self.context_list = []
        self.answer_list = []
        self.response_list = []

        # move to get_item?
        for label, log in zip(labels, logs):
            if label['target'] == True:

                #todo consider to add also special tokens for S and U
                dialog_context = (' ').join([utterance['text'] for utterance in log])

                # retrieve info of the knowledge snippet
                knowledge_snippet = label['knowledge'][0]
                domain = knowledge_snippet['domain']
                entity_id = str(knowledge_snippet['entity_id'])
                doc_id = str(knowledge_snippet['doc_id'])

                # retrieve question and answer part of the knowledge snippet
                document = knowledge_base[domain][entity_id]['docs'][doc_id]
                question = document['title'].strip()
                answer = document['body'].strip()

                # preparing answer
                padding_length = max_length - len(self.tokenizer.tokenize(answer))
                padded_answer = ' '.join([answer] + [self.tokenizer.mask_token] * padding_length)

                # retrieve ground-truth response
                response = label['response'].strip()
                padding_length = max_length - len(self.tokenizer.tokenize(response))
                padded_response = ' '.join([response] + [self.tokenizer.mask_token] * padding_length)

                # store data
                self.context_list.append(dialog_context)
                self.answer_list.append(padded_answer)
                self.response_list.append(padded_response)

        # self.input_embeddings =  self.tokenizer(context_list,
        #                                         answer_list,
        #                                         padding=True)
        # self.target_embeddings = self.tokenizer.batch_encode_plus(response_list,
        #                                                           return_attention_mask = False,
        #                                                           add_special_tokens=False)

        del knowledge_base, labels, logs,

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, item):
        return self.context_list[item], self.answer_list[item], self.response_list[item]



if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', padding_side='left')
    m = MultiWOZDataset(tokenizer, 'data/val/logs.json', 'data/val/labels.json', 'data/knowledge.json')

    from torch.utils.data import DataLoader
    dl = DataLoader(m)