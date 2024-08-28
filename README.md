# Image Classification

You can track the latest updates by visiting the project's github address：[Image Classification Repository](https://github.com/ciaoyizhen/image_classification)

## update
```
2024.8.19: Support use evaluate in offline
2024.8.23: Supports dynamic loading of data
2024.8.25: Fix processor not be save in model
2024.8.28: Support Data Enhancement
```
## Requirement:

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
> Note: multi-gpu  use  torchrun --nproc_per_node=x main.py your_yaml

## Eval Step:
```
python demo/inference_demo.py
```

## WebUI
```
1. python demo/web_demo.py
2. open link with your browser
```



## FAQ
1. open too many file
```
ulimit -n xxx  # increase open file
```
2. How to download a model to train
```
1. open this (https://huggingface.co/models)
2. choose and download a model
3. modify yaml
```
3. Multi Gpu how to train
```
torchrun --nproc-per-node=x main.py configs/test.yaml

see more in (https://pytorch.org/docs/stable/elastic/run.html)
```


## Support the Author

If you find this project helpful and would like to support the author, you can make a donation using WeChat Pay or Alipay by scanning the QR codes below.

**WeChat Pay:**

<img src="assets/WeChat%20Pay.jpg" alt="WeChat Pay QR Code" width="300"/>

**Alipay:**

<img src="assets/Alipay.jpg" alt="WeChat Pay QR Code" width="300"/>


Thank you for your support!
