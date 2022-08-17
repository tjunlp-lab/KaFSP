import json
import ujson
from tqdm import tqdm
import os
import sys
import pandas as pd
from sklearn.utils import shuffle
sys.path.append('..')


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


def main():
    train_path = 'data/CSQA_v9/train'
    val_path = 'data/CSQA_v9/valid'
    test_path = 'data/CSQA_v9/test'
    train = pd.DataFrame(columns=['description', 'entity', 'truple_1', 'truple_2', 'label'])
    valid = pd.DataFrame(columns=['description', 'entity', 'truple_1', 'truple_2', 'label'])
    test = pd.DataFrame(columns=['description', 'entity', 'truple_1', 'truple_2', 'label'])
    train_files = []
    for root, dirs, files in os.walk(train_path):
        for file in files:
            temp = os.path.join(root, file)
            if '.json' in temp:
                train_files.append(temp)
    for f in tqdm(train_files):
        with open(f) as json_file:
            convs = json.load(json_file)
            for conv in convs:
                if conv['speaker'] == 'SYSTEM':
                    continue
                description = conv['utterance']
                entities = conv.get('entities_in_utterance')
                if not entities:
                    continue
                for entity in entities:
                    surface_form = id_entity[entity]
                    truples = get_truples(entity)
                    train = train.append({'description':description, 'entity':surface_form, 'truple_1':truples[0], 'truple_2':truples[1], 'label':1}, ignore_index=True)
                    # get same surface entity
                    same_entities = surface_id[surface_form]
                    if len(same_entities) == 1:
                        continue
                    else:
                        for same_entity in same_entities:
                            if same_entity == entity:
                                continue
                            truples = get_truples(same_entity)
                            train = train.append({'description':description, 'entity':surface_form, 'truple_1':truples[0], 'truple_2':truples[1], 'label':0}, ignore_index=True)
            if len(train) > 500000:
                break
    train = shuffle(train)
    train.to_csv('ED/train.csv')

    val_files = []
    for root, dirs, files in os.walk(val_path):
        for file in files:
            temp = os.path.join(root, file)
            if '.json' in temp:
                val_files.append(temp)
    for f in tqdm(val_files):
        with open(f) as json_file:
            convs = json.load(json_file)
            for conv in convs:
                description = conv['utterance']
                entities = conv.get('entities_in_utterance')
                if not entities:
                    continue
                for entity in entities:
                    surface_form = id_entity[entity]
                    truples = get_truples(entity)
                    valid = valid.append({'description':description, 'entity':surface_form, 'truple_1':truples[0], 'truple_2':truples[1], 'label':1}, ignore_index=True)
                    # get same surface entity
                    same_entities = surface_id[surface_form]
                    if len(same_entities) == 1:
                        continue
                    else:
                        for same_entity in same_entities:
                            if same_entity == entity:
                                continue
                            truples = get_truples(same_entity)
                            valid = valid.append({'description':description, 'entity':surface_form, 'truple_1':truples[0], 'truple_2':truples[1], 'label':0}, ignore_index=True)
            if len(valid) > 40000:
                break    
    valid = shuffle(valid)
    valid.to_csv('ED/valid.csv')

    test_files = []
    for root, dirs, files in os.walk(test_path):
        for file in files:
            temp = os.path.join(root, file)
            if '.json' in temp:
                test_files.append(temp)
    for f in tqdm(test_files):
        with open(f) as json_file:
            convs = json.load(json_file)
            for conv in convs:
                description = conv['utterance']
                entities = conv.get('entities_in_utterance')
                if not entities:
                    continue
                for entity in entities:
                    surface_form = id_entity[entity]
                    truples = get_truples(entity)
                    test = test.append({'description':description, 'entity':surface_form, 'truple_1':truples[0], 'truple_2':truples[1], 'label':1}, ignore_index=True)
                    # get same surface entity
                    same_entities = surface_id[surface_form]
                    if len(same_entities) == 1:
                        continue
                    else:
                        for same_entity in same_entities:
                            if same_entity == entity:
                                continue
                            truples = get_truples(same_entity)
                            test = test.append({'description':description, 'entity':surface_form, 'truple_1':truples[0], 'truple_2':truples[1], 'label':0}, ignore_index=True)
    test = shuffle(test)
    test.to_csv('ED/test.csv')

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




if __name__ == '__main__':
    main()
