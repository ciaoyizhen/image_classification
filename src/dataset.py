# -*- encoding: utf-8 -*-
# @Time    :   2024/08/15 23:27:00
# @File    :   dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   data
import os
import torch
import numpy as np
import albumentations as A
from PIL import Image
from .utils import getClass
from datasets import load_dataset, concatenate_datasets


class Dataset():
    def __init__(self, cfg, id2label, label2id, processor):
        """init

        Args:
            cfg (dict): args of *_dataset in yaml
        """
        self.cfg = cfg
        self.id2label = id2label
        self.label2id = label2id
        self.processor = processor
        self.format_map = {
            ".txt": "text",
            ".csv": "csv",
            ".json": "json",
            ".tsv": "csv"
        }
        self._loadData()  # str format not Tensor
        self._createTransform()
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

    def _createTransform(self):
        if "albumentations" in self.cfg:
            albumentations_ = self.cfg["albumentations"]
        else:
            self.transform = None
            return
        A_list = []
        for item in albumentations_:
            kwargs = item.get("args", None)
            module_ = getClass(item["type"])
            if kwargs is not None:
                class_ = module_(**kwargs)
            else:
                class_ = module_()
            A_list.append(class_)
        self.transform = A.Compose(A_list)
    
    def _process(self):
        """load image to tensor and apply transformer
        """
        print("processing ...")
        
        def process(examples):
            text = examples["text"]
            images = []
            labels = []
            for item in text:
                img_path, label = item.split("\t")
                try:
                    img = Image.open(img_path)
                    img = img.convert("RGB")  # may png or other format
                    if self.transform is not None:
                        img = np.asarray(img)
                        transform_img = self.transform(image = img)
                        img = transform_img["image"]
                    inputs = self.processor(images=img, return_tensors="pt")
                    img = inputs["pixel_values"].squeeze(0)
                    label = self.label2id[label]
                    label = torch.tensor(label)
                    images.append(img)
                    labels.append(label)
                except:

                    continue
            return {"pixel_values": torch.stack(images), "labels": torch.stack(labels)}

        
        self.data = self.data.with_transform(process)
        

    
