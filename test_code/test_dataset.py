# -*- encoding: utf-8 -*-
# @Time    :   2024/08/16 15:41:07
# @File    :   test_dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   test dataset
import torch
from src.dataset import Dataset
from src.utils import loadLabelFile
from datasets import set_caching_enabled
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

set_caching_enabled(False)

# def test_single_dataset(yaml_data):
#     id2label, label2id = loadLabelFile(yaml_data)
#     dataset = Dataset(yaml_data["train_dataset"]["args"], id2label, label2id)
#     dataloader = DataLoader(dataset.data, batch_size=yaml_data["train_args"]["train_batch_size"], collate_fn=DefaultDataCollator())
#     for batch in dataloader:
#         img = batch["pixel_values"]
#         labels = batch["labels"]
#         assert isinstance(labels, torch.Tensor)
#         assert labels.dim() == 1
#         assert isinstance(img, torch.Tensor)
#         assert img.dim() == 4
#         break