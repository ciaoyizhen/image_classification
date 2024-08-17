# -*- encoding: utf-8 -*-
# @Time    :   2024/08/15 22:59:12
# @File    :   main.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   main entrance

import yaml
import typer
from setproctitle import setproctitle
from typing import Annotated
from src import ImgTrainer, loadLabelFile


app = typer.Typer()

def getDefaultConfig():
    while True:
        config_path = typer.prompt("train yaml config path")
        if not config_path.endswith(".yaml"):
            print("config_path must be a yaml file")
        
        return config_path

@app.command("main")
def main(
    *,
    config_path: Annotated[str, typer.Argument(..., help="train yaml config path ", default_factory=getDefaultConfig)]
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    id2label, label2id = loadLabelFile(config)
    setproctitle(config["name"])
    
    trainer = ImgTrainer(config, id2label, label2id)
    trainer()
    
    
if __name__ == "__main__":
    app()