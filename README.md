## Requirements and Setup

Python version >= 3.7

PyTorch version >= 1.6.0

PyTorch Geometric (PyG) >= 1.6.1

``` bash
cd KaFSP
pip install -r requirements.txt
```

## CSQA dataset
### Download
We evaluate LASAGNE on [CSQA](https://amritasaha1812.github.io/CSQA/) dataset. You can download the dataset from [here](https://amritasaha1812.github.io/CSQA/download/).

### Wikidata Knowlegde Graph
CSQA dataset is based on Wikidata [Knowlegde Graph](https://www.wikidata.org/wiki/Wikidata:Main_Page), the authors provide a preproccesed version of it which can be used when working with the dataset.
You can download the preprocessed Wikidata knowlegde graph files from [here](https://zenodo.org/record/4052427#.YBU7xHdKjfZ).
After dowloading you will need to move them under the [knowledge_graph](knowledge_graph) directory.

We prefer to merge some JSON files from the preprocessed Wikidata, for accelerating the process of reading all the knowledge graph files. In particular, we create three new JSON files using the script [prepare_data.py](scripts/prepare_data.py). Please execute the script as below.
``` bash
# prepare knowlegde graph files
python scripts/prepare_data.py
```

### Inverted index on Wikidata entities
For building an inverted index on wikidata entities we use [elastic](https://www.elastic.co/) search. Consider the script file [csqa_elasticse.py](scripts/csqa_elasticse.py) for doing so.

### Annotate Dataset
Next, using the preproccesed Wikidata files we can annotate CSQA dataset with our grammar. At the same time we also annotate the entity spans for all utterances.
``` bash
# annotate CSQA dataset with entity spans and our grammar
python annotate_csqa/preprocess.py --partition train --annotation_task all --read_folder /path/to/CSQA --write_folder /path/to/write
```

## BERT embeddings
Before training the framework, we need to create BERT embeddings for the knowledge graph (entity) types and relations. You can do that by running.
``` bash
# create bert embeddings
python scripts/node_embeddings.py
```

## Train Framework
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train framework
python train.py
```

## Test
For testing we have two steps.
### Generate Actions
First, we generate the actions and save then in JSON file using the trained model.
``` bash
# generate actions for a specific question type
python inference.py --question_type Clarification
```

### Execute Actions
Second, we execute the actions and get the results from Wikidata files.
``` bash
# execute actions for a specific question type
python action_executor/run.py --file_path /path/to/actions.json --question_type Clarification
```

### Entity Disambiguation
We need to change the "data_path" in the file entity_disambiguation.py to get the disambiguated files.
``` bash
python entity_disambiguation.py 
```
