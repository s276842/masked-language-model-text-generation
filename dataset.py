import json
from torch.utils.data import Dataset

SPECIAL_TOKENS = {
    'U' : '<U>',    # Token for beginning of User utterance
    'S' : '<S>',    # Token for beginning of System utterance
    'Ka' : '<Ka>',  # Token for beginning of Answer
    'Kq' : '<Kq>'   # Token for beginning of Question
}

#todo implement truncation of context (given number of utternaces)
#todo implement choice for special tokens to use
#todo implement alignement of knowledge to ground-truth
class MultiWOZDataset(Dataset):

    def __init__(self, path_to_logs, path_to_labels, path_to_knowledge):
        self.knowledge_base = json.load(open(path_to_knowledge))
        labels = json.load(open(path_to_labels))
        logs = json.load(open(path_to_logs))

        knowledge_seeking_turns = [i for i, label in enumerate(labels) if label['target'] is True]
        self.labels = [labels[i] for i in knowledge_seeking_turns]
        self.logs = [logs[i] for i in knowledge_seeking_turns]

        del logs, labels, knowledge_seeking_turns

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        dialog_context = self.logs[item]

        # Retrieve knowledge document
        knowledge_snippet = label['knowledge'][0]
        domain = knowledge_snippet['domain']
        entity_id = str(knowledge_snippet['entity_id'])
        doc_id = str(knowledge_snippet['doc_id'])
        document = self.knowledge_base[domain][entity_id]['docs'][doc_id]
        question = document['title'].strip()
        answer = document['body'].strip()

        # Retrieve ground-truth response
        response = label['response'].strip()

        # Create input as concatenation of utterances and knowledge document
        input = ' '.join([f"{SPECIAL_TOKENS[utterance['speaker']]} {utterance['text']}" for utterance in dialog_context]
                         + [SPECIAL_TOKENS['Ka'], answer])

        return input, response



if __name__ == '__main__':
    m = MultiWOZDataset('data/val/logs.json', 'data/val/labels.json', 'data/knowledge.json')

    from torch.utils.data import DataLoader
    dl = DataLoader(m, batch_size=8)

    print(m[0])