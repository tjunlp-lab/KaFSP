import re
import json
import torch
from type_dataset import TypeDataset
import os
import sys
sys.path.append('..')
from dataset import CSQADataset
import flair
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings, DocumentPoolEmbeddings
import ujson
from tqdm import tqdm


torch.cuda.set_device(3)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flair.device = DEVICE

bert = DocumentPoolEmbeddings([BertEmbeddings('bert-base-uncased')])

graph_nodes = list(CSQADataset().get_vocabs()['graph'].stoi.keys())

id_entity = json.loads(open('../knowledge_graph/items_wikidata_n.json').read())
# print(len(id_entity))
id_relation = json.loads(open('../knowledge_graph/filtered_property_wikidata4.json').read())
subject_type_truples = json.loads(open('../knowledge_graph/wikidata_type_dict.json').read())
object_type_truples = json.loads(open('../knowledge_graph/wikidata_rev_type_dict.json').read())
relation_subject_object = {}
for key in subject_type_truples:
    for rel in subject_type_truples[key]:
        if rel in relation_subject_object.keys():
            relation_subject_object[rel][key] = subject_type_truples[key][rel]
        else:
            relation_subject_object[rel] = {}
            relation_subject_object[rel][key] = subject_type_truples[key][rel]
relation_object_subject = {}
for key in object_type_truples:
    for rel in object_type_truples[key]:
        if rel in relation_object_subject.keys():
            relation_object_subject[rel][key] = object_type_truples[key][rel]
        else:
            relation_object_subject[rel] = {}
            relation_object_subject[rel][key] = object_type_truples[key][rel]
# relation_subject_object = ujson.loads(open(f'../knowledge_graph/relation_subject_object.json').read())
# relation_object_subject = ujson.loads(open(f'../knowledge_graph/relation_object_subject.json').read())
# print(f'Loaded relation_triples {time.perf_counter()-tic:0.2f}s')


def get_rel_context(rel_id):
    rel = id_relation[rel_id]
    triples = rel
    cnt = 0
    if rel_id in relation_subject_object.keys():
        for sub_id in relation_subject_object[rel_id]:
            if not relation_subject_object[rel_id][sub_id]:
                continue
            obj_id = relation_subject_object[rel_id][sub_id][0]
            triples += ' [SEP] ' + id_entity[sub_id] + ' ' + rel + ' ' + id_entity[obj_id]
            cnt += 1
            if cnt >= 3:
                break
    if rel_id in relation_object_subject.keys() and cnt < 3:
        for obj_id in relation_object_subject[rel_id]:
            if not relation_object_subject[rel_id][obj_id]:
                continue
            sub_id = relation_object_subject[rel_id][obj_id][0]
            triples += ' [SEP] ' + id_entity[sub_id] + ' ' + rel + ' ' + id_entity[obj_id]
            cnt += 1
            if cnt >= 3:
                break
    while cnt < 3:
        triples += ' [SEP] ' + rel
        cnt += 1
    return triples


def get_type_context(type_id):
    truples = id_entity[type_id]
    cnt = 0
    if type_id in subject_type_truples.keys():
        subject_text = id_entity[type_id]
        for rel_id in subject_type_truples[type_id]:
            relation = id_relation[rel_id]
            object_text = id_entity[subject_type_truples[type_id][rel_id][0]]
            truples += ' ' + '[SEP]' + ' ' + subject_text + ' ' + relation + ' ' + object_text
            cnt += 1
            if cnt >= 3:
                break
    if type_id in object_type_truples.keys() and cnt < 3:
        object_text = id_entity[type_id]
        for rel_id in object_type_truples[type_id]:
            relation = id_relation[rel_id]
            subject_text = id_entity[object_type_truples[type_id][rel_id][0]]
            truples += ' ' + '[SEP]' + ' ' + subject_text + ' ' + relation + ' ' + object_text
            cnt += 1
            if cnt >= 3:
                break
    while cnt < 3:
        truples += ' ' + '[SEP]' + ' ' + id_entity[type_id]
        cnt += 1
    return truples



na_node = Sentence(graph_nodes[0])
pad_node = Sentence(graph_nodes[1])
bert.embed(na_node)
bert.embed(pad_node)
node_embeddings = {
    graph_nodes[0]: na_node.embedding.detach().cpu().tolist(),
    graph_nodes[1]: na_node.embedding.detach().cpu().tolist()
}
for node in tqdm(graph_nodes[2:]):
    # print(get_rel_context(node) if node.startswith('P') else ' ')
    node_label = Sentence(get_type_context(node) if node.startswith('Q') else get_rel_context(node))
    bert.embed(node_label)
    node_embeddings[node] = node_label.embedding.detach().cpu().tolist()

with open(f'node_embeddings.json', 'w') as outfile:
    json.dump(node_embeddings, outfile, indent=4)



# for node in graph_nodes:
#     print(get_type_context(node) if node.startswith('Q') else 'error')