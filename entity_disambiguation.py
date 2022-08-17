import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import XLNetForSequenceClassification, XLNetConfig, XLNetTokenizer
import json
import time
import ujson


class XlnetModelTest(nn.Module):
    def __init__(self):
        super(XlnetModelTest, self).__init__()
        config = XLNetConfig.from_pretrained('entity_disamb/model/config.json')
        self.xlnet = XLNetForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        logits = self.xlnet(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                            token_type_ids=batch_seq_segments)
        probabilities = nn.functional.softmax(logits[0], dim=-1)
        return logits, probabilities


class DataPrecessForSentence(Dataset):
    def __init__(self, bert_tokenizer, entity_context, max_char_len=100):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments = self.get_input(
            entity_context)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx]

    def get_input(self, entity_context):
        tokens_seq = entity_context['description']
        surface_form = entity_context['entity']
        truple_1 = entity_context['truple_1']
        truple_2 = entity_context['truple_2']
        tokens_seq = list(map(self.bert_tokenizer.tokenize, tokens_seq))
        surface_form = list(map(self.bert_tokenizer.tokenize, surface_form))
        truple_1 = list(map(self.bert_tokenizer.tokenize, truple_1))
        truple_2 = list(map(self.bert_tokenizer.tokenize, truple_2))
        result = list(map(self.trunate_and_pad, tokens_seq,
                      surface_form, truple_1, truple_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(seq_segments).type(torch.long)

    def trunate_and_pad(self, tokens_seq, surface_form, truple_1, truple_2):
        if len(tokens_seq) > ((self.max_seq_len - 5)//4):
            tokens_seq = tokens_seq[0:(self.max_seq_len - 5)//4]
        if len(surface_form) > ((self.max_seq_len - 5)//4):
            surface_form = surface_form[0:(self.max_seq_len - 5)//4]
        if len(truple_1) > ((self.max_seq_len - 5)//4):
            truple_1 = truple_1[0:(self.max_seq_len - 5)//4]
        if len(truple_2) > ((self.max_seq_len - 5)//4):
            truple_2 = truple_2[0:(self.max_seq_len - 5)//4]
        seq = seq = tokens_seq + ['<sep>'] + surface_form + ['<sep>'] + \
            truple_1 + ['<sep>'] + truple_2 + ['<sep>'] + ['<cls>']
        seq_segment = [0] * (len(tokens_seq) + 1) + [1] * (len(surface_form) + 1) + [
            2] * (len(truple_1)+1) + [3] * (len(truple_2)+1) + [4]
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0] * (self.max_seq_len - len(seq))
        seq_mask = [1] * len(seq) + padding
        seq_segment = seq_segment + padding
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment


class EntityDisamb:
    def __init__(self, checkpoint):
        self.bert_tokenizer = XLNetTokenizer.from_pretrained(
            'ED/xlnet', do_lower_case=True)
        self.device = torch.device("cuda")
        self.checkpoint = torch.load(checkpoint)
        self.model = XlnetModelTest().to(self.device)
        self.model.load_state_dict(self.checkpoint['model'])

    def classify(self, entity_context):
        data = DataPrecessForSentence(self.bert_tokenizer, entity_context)
        seqs, seq_masks, seq_segments = data.seqs, data.seq_masks, data.seq_segments
        self.model.eval()
        with torch.no_grad():
            seqs, masks, segments = seqs.to(self.device), seq_masks.to(
                self.device), seq_segments.to(self.device)
            _, probabilities = self.model(seqs, masks, segments)
        idx = probabilities.argmax(dim=0)[1].item()
        return idx

id_entity = ujson.loads(open('knowledge_graph/items_wikidata_n.json').read())
id_relation = ujson.loads(open('knowledge_graph/filtered_property_wikidata4.json').read())
subject_triples_1 = ujson.loads(open('knowledge_graph/wikidata_short_1.json').read())
subject_triples_2 = ujson.loads(open('knowledge_graph/wikidata_short_2.json').read())
subject_triples = {**subject_triples_1, **subject_triples_2}
object_triples = ujson.loads(open('knowledge_graph/comp_wikidata_rev.json').read())
surface_id = {}
for key in id_entity:
    if not surface_id.get(id_entity[key]):
        surface_id[id_entity[key]] = [key]
    else:
        surface_id[id_entity[key]].append(key)

def get_truples(entity_id):
    surface_form = id_entity[entity_id]
    truples = []
    sub_trp = subject_triples.get(entity_id)
    if sub_trp:
        for rel in sub_trp:
            if len(truples) == 2:
                break
            if not id_relation.get(rel):
                continue
            if sub_trp[rel]:
                obj = sub_trp[rel][0]
                truples.append(f'{surface_form} {id_relation[rel]} {id_entity[obj]}')
    obj_trp = object_triples.get(entity_id)
    if obj_trp:
        for rel in obj_trp:
            if len(truples) == 2:
                break
            if not id_relation.get(rel):
                continue
            if obj_trp[rel]:
                sub = obj_trp[rel][0]
            truples.append(f'{id_entity[sub]} {id_relation[rel]} {surface_form}')
    while len(truples) < 2:
        truples.append(surface_form)
    return truples

entity_dis = EntityDisamb('ED/model/best.pth.tar')

data_path = 'experiments/inference/ep16_test_Comparative Reasoning (Count) (All).json'

data = []
with open(data_path) as json_file:
    data = json.load(json_file)
count_total = 0
tic = time.perf_counter()
inference_actions = []
ambiguation = 0
correct = 0
correct_surface = 0
for i, d in enumerate(data):
    count_total += 1
    try:
        if d['actions'] is not None:
            for cnt in range(len(d['actions'])):
                if d['actions'][cnt][0] == 'entity' and d['actions'][cnt][1].startswith("Q"):
                    surface_form = id_entity[d['actions'][cnt][1]]
                    entity = surface_id[surface_form]
                    if len(entity) == 1:
                        continue
                    truples = []
                    for j in entity:
                        truples.append(get_truples(j))
                    entity_context = {
                        'description': [d['question'] for _ in entity],
                        'entity': [surface_form for _ in entity],
                        'truple_1': [_[0] for _ in truples],
                        'truple_2': [_[1] for _ in truples]
                    }
                    idx = entity_dis.classify(entity_context)
                    if entity[idx] != d["actions"][cnt][1]:
                        print(f'{idx}:{entity[idx]}------>{d["actions"][cnt][1]}')
                        ambiguation += 1
                        if d['gold_actions'] is not None and len(d["gold_actions"]) == len(d['actions']):
                            if id_entity[d["actions"][cnt][1]] == id_entity[d["gold_actions"][cnt][1]]:
                                correct_surface += 1
                                if d["actions"][cnt][1] != d["gold_actions"][cnt][1] and entity[idx] == d["gold_actions"][cnt][1]:
                                    correct += 1
                    d['actions'][cnt][1] = entity[idx]
                    
    except Exception as ex:
        print(d['question'])
        print(d['actions'])
    inference_actions.append(d)
    toc = time.perf_counter()
    print(f'==> Finished {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')
print(f'disamb:{ambiguation}， total:{len(data)},  correct surface:{correct_surface} ,correct disamb:{correct}')
with open(f'for_abstudy_{data_path[:-5]}_ED.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json.dumps(inference_actions, indent=4))
