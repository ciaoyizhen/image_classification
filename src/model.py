# -*- encoding: utf-8 -*-
# @Time    :   2024/08/15 23:26:40
# @File    :   model.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   model

from torch import nn
from .utils import getClass

class Model():
    def __init__(self, cfg, id2label, label2id) -> None:
        """init, model in yaml

        Args:
            cfg (dict): model in yaml
        """
        self.cfg = cfg
        self.model = getClass(self.cfg["type"]).from_pretrained(**self.cfg["args"])
        num_labels = len(id2label)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_labels)
        self.model.num_labels = num_labels
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id