# -*- encoding: utf-8 -*-
# @Time    :   2024/08/15 23:17:34
# @File    :   trainer.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   train scripts

import os
import logging
import traceback
import evaluate
from .model import Model
from .utils import getClass
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from datasets import disable_caching, enable_caching


logger = logging.getLogger("transformers.trainer")  # Unified for transformers

class ImgTrainer():
    def __init__(self, config:dict, id2label:dict, label2id:dict):
        """init, used to instantiate the yaml file

        Args:
            config (dict): yaml dict
            id2label (dict): map
            label2id (dict): map
        """
        self.config = config
        self.id2label = id2label
        self.label2id = label2id
        self._initDataset()
        self._initModel()
        self._initTrainArgument()
        self._createEvalMetric()
        self._createTrainer()
    
    def _initDataset(self):
        """_summary_
        """
        # use dataset cache
        is_dataset_cached = self.config.get("is_dataset_cached", True)
        if is_dataset_cached:
            enable_caching()
            dataset_cache_dir = self.config.get("dataset_cache_dir", None)
            if dataset_cache_dir is None:
                logger.info("dataset_cache_dir not be set, using default")
            else:
                os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
        else:
            disable_caching()
            
        
        train_cfg = self.config.get("train_dataset", None)
        validation_cfg = self.config.get("validation_dataset", None)
        test_cfg = self.config.get("test_dataset", None)
        
        assert train_cfg is not None, "train_dataset needs to be configured in the config yaml file"
        self.train_dataset = getClass(train_cfg["type"])(train_cfg["args"], self.id2label, self.label2id)
        
        if validation_cfg is None:
            self.validation_dataset = self.train_dataset
            logger.info("validation_dataset not be set, using train_dataset")
        else:
            self.validation_dataset = getClass(validation_cfg["type"])(validation_cfg["args"], self.id2label, self.label2id)
        
        if test_cfg is None:
            self.test_dataset = self.validation_dataset
            logger.info("test_dataset is not be set, using validation_dataset")
        else:
            self.test_dataset = getClass(test_cfg["type"])(test_cfg["args"], self.id2label, self.label2id)

    def _initModel(self):
        """init model
        """
        self.model = Model(self.config["model"], self.id2label, self.label2id)
    
    def _initTrainArgument(self):
        """generate train_args
        """
        args = self.config["train_args"]
        if not args.get("output_dir", None):
            args["output_dir"] = os.path.join("outputs", self.config["name"])
        
        self.train_args = TrainingArguments(**args)
    
    def _createEvalMetric(self):
        """create evaluate
        """
        evaluate_metric = self.config["evaluate_metric"]
        # special process
        if "f1" in evaluate_metric:
            evaluate_metric.remove("f1")
            self.f1_metric = evaluate.load("metrics/f1")
        else:
            self.f1_metric = None
            
        if "recall" in evaluate_metric:
            evaluate_metric.remove("recall")
            self.recall_metric = evaluate.load("metrics/recall")
        else:
            self.recall_metric = None
            
        if "precision" in evaluate_metric:
            evaluate_metric.remove("precision")
            self.precision_metric = evaluate.load("metrics/precision")
        else:
            self.precision_metric = None
        
        if "accuracy" in evaluate_metric:
            evaluate_metric.remove("accuracy")
            self.accuracy_metric = evaluate.load("metrics/accuracy")
        else:
            self.accuracy_metric = None
        
        assert len(evaluate_metric) == 0, f"{evaluate_metric} could not be loaded"


 

    def _createTrainer(self):
        """create Trainer
        """
        
        def eval_metric(eval_predict):
            """calculation accuracy

            Args:
                eval_predict (dict): model output

            Returns:
                dict: accuracy
            """
            predictions, labels = eval_predict
            predictions = predictions.argmax(axis=-1)
            acc = {}
            if self.accuracy_metric is not None:
                acc_ = self.accuracy_metric.compute(predictions=predictions, references=labels)
                acc.update(acc_)
            if self.f1_metric is not None:
                f1 = self.f1_metric.compute(predictions=predictions, references=labels, average="weighted")
                acc.update(f1)
            if self.recall_metric is not None:
                recall = self.recall_metric.compute(predictions=predictions, references=labels, average="weighted")
                acc.update(recall)
            if self.precision_metric is not None:
                precision = self.precision_metric.compute(predictions=predictions, references=labels, average="weighted")
                acc.update(precision)
            return acc
        
        self.trainer = Trainer(
            model = self.model.model,
            args= self.train_args,
            data_collator=DefaultDataCollator(),
            train_dataset=self.train_dataset.data,
            eval_dataset=self.validation_dataset.data,
            compute_metrics=eval_metric
        )
        
    def __call__(self):
        """training and evaluate
        """
        logger.info("training...")
        self.trainer.train()
        logger.info("train finish. evaluating...")
        self.trainer.evaluate(self.test_dataset.data)