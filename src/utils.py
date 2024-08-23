# -*- encoding: utf-8 -*-
# @Time    :   2024/08/15 23:19:36
# @File    :   utils.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   utils function
import importlib
from typing import Tuple, Dict, Type

def getClass(module:str) -> Type:
    """return class based on module

    Args:
        module (str): format `module,class`

    Returns:
        Type: The class object corresponding to the specified module and class name.
    
    Example:
        >>> MyClass = getClass("my_module,MyClass")
        >>> instance = MyClass()
        >>> print(isinstance(instance, MyClass))
        True
    """
    module, class_ = module.split(",")
    module = importlib.import_module(module)
    class_ = getattr(module, class_)
    return class_

def loadLabelFile(config:dict) -> Tuple[Dict[int, str], Dict[str, int]]:
    """generate id2label and label2id based on label_config in yaml
    Args:
        config (dict): config yaml dict
        
    Returns:
        Tuple[Dict[int, str], Dict[str, int]]:
            - id2label (Dict[int, str]): int -> label
            - label2id (Dict[str, int]): label -> int

    Example:
        >>> config = {"label_config": "path/to/label_config.txt"}
        >>> id2label, label2id = loadLabelFile(config)
        >>> print(id2label)
        {0: 'label1', 1: 'label2'}
        >>> print(label2id)
        {'label1': 0, 'label2': 1}
    """
    id2label = {}
    label2id = {}
    id_ = 0
    with open(config["label_config"], "r", encoding="utf-8") as f:
        for line in f.readlines():
            if label := line.strip():
                id2label[id_] = label
                label2id[label] = id_
                id_ += 1
                
    return id2label, label2id
