user = {
            "ques_type_id": 7,
            "question-type": "Quantitative Reasoning (Count) (All)",
            "description": "Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Single entity type",
            "entities_in_utterance": [],
            "speaker": "USER",
            "relations": [
                  "P530"
            ],
            "count_ques_sub_type": 5,
            "type_list": [
                  "Q15617994",
                  "Q15617994"
            ],
            "count_ques_type": 1,
            "utterance": "How many administrative territories have diplomatic relations with exactly 9 administrative territories ?",
            "is_incomplete": 0,
            "context": [
                  [
                        0,
                        "how",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        1,
                        "many",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        2,
                        "administrative",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        3,
                        "territories",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        4,
                        "have",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        5,
                        "diplomatic",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        6,
                        "relations",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        7,
                        "with",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        8,
                        "exactly",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        9,
                        "9",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        10,
                        "administrative",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        11,
                        "territories",
                        "NA",
                        "NA",
                        "O"
                  ],
                  [
                        12,
                        "?",
                        "NA",
                        "NA",
                        "O"
                  ]
            ],
            "is_ner_spurious": 'false'
      }
system = {
            "all_entities": [
                  "Q43",
                  "Q79"
            ],
            "speaker": "SYSTEM",
            "entities_in_utterance": [],
            "utterance": "2",
            "active_set": [
                  "(c(Q15617994),P530,c(Q15617994))"
            ],
            "gold_actions": [
                  [
                        "action",
                        "count"
                  ],
                  [
                        "action",
                        "equal"
                  ],
                  [
                        "action",
                        "find_tuple_counts"
                  ],
                  [
                        "relation",
                        "P530"
                  ],
                  [
                        "type",
                        "Q15617994"
                  ],
                  [
                        "type",
                        "Q15617994"
                  ],
                  [
                        "value",
                        "9"
                  ]
            ],
            "is_spurious": 'true',
            "context": [
                  [
                        0,
                        "num",
                        "NA",
                        "NA",
                        "O"
                  ]
            ],
            "is_ner_spurious": 'false'
      }

import os
import time
import logging
import json
import argparse
from glob import glob
from pathlib import Path
import sys
sys.path.append("..")
from knowledge_graph.knowledge_graph import KnowledgeGraph
from action_annotators.annotate import ActionAnnotator
from ner_annotators.annotate import NERAnnotator
ROOT_PATH = Path(os.path.dirname(__file__)).parent

kg = KnowledgeGraph()
action_annotator = ActionAnnotator(kg)
conversation = action_annotator({user, system})
print(conversation)