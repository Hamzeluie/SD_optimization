import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append("model")
import yaml
from argparse import Namespace
from models.convert import convert
import shutil


if __name__ == "__main__":  
    # get parameters
    # param_yaml_file = sys.argv[1]
    param_yaml_file = "/home/naserwin/hamze/SD_optimization/params.yaml"
    params = yaml.safe_load(open(param_yaml_file))
    params = params["convert"]
    dataset_name = params["dataset_name"]
    data = {
        "trained_model_path": "", 
        "checkpoint_path": "", 
        "half": None,
        "use_safetensors": None
        }
    args = Namespace(**data)
    # set parameters
    params["trained_model_path"] = os.path.join(params["trained_model_path"], dataset_name)
    params["checkpoint_path"] = os.path.join(params["checkpoint_path"], dataset_name)
    if os.path.isdir(params["checkpoint_path"]):
        shutil.rmtree(params["checkpoint_path"])
    os.makedirs(params["checkpoint_path"], exist_ok=True)
    # convertion proccess
    for check_path in os.listdir(params["trained_model_path"]):
        if check_path.startswith("checkpoint"):
            print(check_path)
            args.trained_model_path = os.path.join(params["trained_model_path"], check_path)
            args.checkpoint_path = os.path.join(params["checkpoint_path"], check_path + ".ckpt")
            convert(args)
    