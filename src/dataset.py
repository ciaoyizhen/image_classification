# -*- encoding: utf-8 -*-
# @Time    :   2024/08/15 23:27:00
# @File    :   dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   data
import os
import torch
import traceback
from PIL import Image
from .utils import getClass
from datasets import load_dataset, concatenate_datasets


class Dataset():
    def __init__(self, cfg, id2label, label2id):
        """init

        Args:
            cfg (dict): args of *_dataset in yaml
        """
        self.cfg = cfg
        self.id2label = id2label
        self.label2id = label2id
        self.format_map = {
            ".txt": "text",
            ".csv": "csv",
            ".json": "json",
            ".tsv": "csv"
        }
        self._loadData()  # str format not Tensor
        self._createPreProcess()
        self._process()
    
    def _chooseFileFormat(self, file_path:str) -> str:
        """get function `load_dataset` format based on file_path

        Args:
            file_path (str): input_file_path
        
        Returns:
            str: function `load_dataset` format

        Example:
            >>> format_ = self._chooseFileFormat("data/file.txt")
            >>> print(format_)
            "text"
        """
        _, ext = os.path.splitext(file_path)
        format_ = self.format_map.get(ext, None)
        assert format_ is not None, f"file_path only support {set(self.format_map.keys())}"
        if format_ not in {"text"}:
            raise NotImplementedError(f"currently not implement {ext}")
        return format_
    
    def _loadData(self):
        """load data
        """
        print("loading Data...")
        
        data_path_list = self.cfg["data_paths"]
        assert len(data_path_list) != 0, "data_paths length not be zero!"
        
        if len(data_path_list) == 1:
            data_path = data_path_list[0]
            format_ = self._chooseFileFormat(data_path)
            self.data = load_dataset(format_, data_files=data_path, split="train")
        else:        
            datasets = []
            for data_path in data_path_list:
                format_ = self._chooseFileFormat(data_path)
                datasets.append(load_dataset(format_, data_files=data_path, split="train"))
            self.data = concatenate_datasets(datasets)

    def _createPreProcess(self):
        """create AutoImageProcessor
        """
        print("creating processor")
        self.processor = getClass(self.cfg["processor"]["type"]).from_pretrained(**self.cfg["processor"]["args"])

    def _applyTransforms(self, examples):
        img_list = []
        for img_path, img in zip(examples["img_path"], examples["img"]):
            try:
                img = self.transforms(img)
            except:
                print(traceback.format_exc())
                print(f"{img_path} is error, pass!")
                img = None
            img_list.append(img)
        examples["img"] = img_list
        return examples
    
    def _process(self):
        """load image to tensor and apply transformer
        """
        print("processing ...")
        def process(example):
            text = example["text"]
            img_path, label = text.split("\t")
            img = Image.open(img_path)
            try:
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {"pixel_values": inputs["pixel_values"].squeeze(0)}  # remove batch dim
            except KeyboardInterrupt:
                raise
            except:
                print(traceback.format_exc())
                print(f"process {img_path} error,pass")
                inputs = {"pixel_values": None}
            label = self.label2id[label]
            label = torch.tensor(label)
            inputs["labels"] = label
            return inputs
        
        
        self.data = self.data.map(process, batched=False, desc="process", num_proc=self.cfg.get("num_proc", 1))
        self.data = self.data.filter(lambda x: x["pixel_values"] is not None)
        self.data = self.data.remove_columns(["text"])
    
