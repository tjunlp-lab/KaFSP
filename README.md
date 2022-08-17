# KaFSP: Knowledge-Aware Fuzzy Semantic Parsing for Conversational Question Answering over a Large-Scale Knowledge Base

## Requirements and Setup

Python version >= 3.7

PyTorch version >= 1.6.0

PyTorch Geometric (PyG) >= 1.6.1

``` bash
# clone the repository
git clone https://github.com/endrikacupaj/LASAGNE.git
cd LASAGNE
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
python scripts/bert_embeddings.py
```

## Train Framework
For training you will need to adjust the paths in [args](args.py) file. At the same file you can also modify and experiment with different model settings.
``` bash
# train framework
python train.py
```


## Test
For testing we have three steps.
### Generate Actions
First, we generate the actions and save then in JSON file using the trained model.
``` bash
# generate actions for a specific question type
python inference.py --question_type Clarification
```
### Entity Disambiguation

Since the entity disambiguation module is trained separately, you need to build the training data first.
``` bash
cd ED
python prepare.py
python train.py
cd ../
```
After training, the model can be used to disambiguate the entities in /path/to/actions.json.
``` bash
python entity_disambiguation.py
```

### Execute Actions
Finally, we execute the actions and get the results from Wikidata files.
``` bash
# execute actions for a specific question type
python action_executor/run.py --file_path /path/to/actions.json --question_type Clarification
```



## License
The repository is under [MIT License](LICENCE).

## Cite
```bash
@inproceedings{li-xiong-2022-kafsp,
    title = "{K}a{FSP}: Knowledge-Aware Fuzzy Semantic Parsing for Conversational Question Answering over a Large-Scale Knowledge Base",
    author = "Li, Junzhuo  and
      Xiong, Deyi",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.35",
    doi = "10.18653/v1/2022.acl-long.35",
    pages = "461--473",
    abstract = "In this paper, we study two issues of semantic parsing approaches to conversational question answering over a large-scale knowledge base: (1) The actions defined in grammar are not sufficient to handle uncertain reasoning common in real-world scenarios. (2) Knowledge base information is not well exploited and incorporated into semantic parsing. To mitigate the two issues, we propose a knowledge-aware fuzzy semantic parsing framework (KaFSP). It defines fuzzy comparison operations in the grammar system for uncertain reasoning based on the fuzzy set theory. In order to enhance the interaction between semantic parsing and knowledge base, we incorporate entity triples from the knowledge base into a knowledge-aware entity disambiguation module. Additionally, we propose a multi-label classification framework to not only capture correlations between entity types and relations but also detect knowledge base information relevant to the current utterance. Both enhancements are based on pre-trained language models. Experiments on a large-scale conversational question answering benchmark demonstrate that the proposed KaFSP achieves significant improvements over previous state-of-the-art models, setting new SOTA results on 8 out of 10 question types, gaining improvements of over 10{\%} F1 or accuracy on 3 question types, and improving overall F1 from 83.01{\%} to 85.33{\%}. The source code of KaFSP is available at https://github.com/tjunlp-lab/KaFSP.",
}
```
