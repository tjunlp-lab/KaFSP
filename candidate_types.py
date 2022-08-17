import json
import torch
import os
import sys

from torch.nn.functional import embedding
sys.path.append('..')
from constants import *
class Candidate:
    def __init__(self, vocabs) -> None:
        embeddings = json.loads(open('knowledge_graph/kg_embeddings.json').read())
        self.can_tensor = [embeddings[keys] for keys in vocabs.itos]
        self.can_tensor = torch.Tensor(self.can_tensor).to(DEVICE)