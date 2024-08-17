# Image Classification

Requirement:
```
python >= 3.8
```

## Goal
Use huggingface to implement a variety of tasks, and you can replace the model at any time without modifying the code.

## Train Step:
```
1. python -m venv .env
2. source .env/bin/activate
3. pip install -r requirements.txt
4. modify yaml config
5. torchrun main.py (yaml_path) or python main.py
```

## Eval Step:
```
python demo/inference_demo.py
```

## WebUI
```
1. python demo/web_demo.py
2. open link with your browser
```

> Note: during training, only the model file is saved, for the image pre-processing, it is not saved, you need to manually put the pre-processing configuration file into the model file to be used

## FQA
1. open too many file
```
ulimit -n xxx  # increase open file
```
2. How to download a model to train
```
1. open this [link](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending)
2. choose and download a model
3. modify yaml
```


## Support the Author

If you find this project helpful and would like to support the author, you can make a donation using WeChat Pay or Alipay by scanning the QR codes below.

**WeChat Pay:**

![WeChat Pay QR Code](assets/WeChat%20Pay.jpg)

**Alipay:**

![Alipay QR Code](assets/Alipay.jpg)

Thank you for your support!