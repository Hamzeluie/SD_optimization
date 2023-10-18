import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append("model")
import os
from PIL import Image

import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import torch
from argparse import Namespace
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

from dvclive import Live
from diffusers import StableDiffusionPipeline, DiffusionPipeline

  
def save_images(images:np.array, tag_save:str, title:str, live: Live):
    b, _, _, _ = images.shape
    fig = plt.figure(figsize=(b, b))
    dim = int(math.sqrt(b)) + 1
    rows = int(dim)
    cols = int(dim)
    for idx, img in enumerate(images):
        fig.add_subplot(rows, cols, idx + 1)
        plt.imshow(img)
    fig.suptitle(title, fontsize=16)
    live.log_image(f"{tag_save}.jpg", fig)


def read_real_image(img_path, live, img_count=100):
        list_images_path = [i for i in Path(img_path).glob(f"**/*.*")]
        # list_images_path = [i for i in Path(img_path).glob(f"**/*.{img_type}")]
        list_images_path = list_images_path[:img_count]
        real_images = []
        for img_path in list_images_path:
            real_images.append(np.array(Image.open(img_path.as_posix())))
        real_images = np.array(real_images)
        del img_path
        save_images(real_images, f"real", "real_images", live)
        return real_images
          
        
def eval_metrices(pipeline: DiffusionPipeline, args:dict, live: Live, real_images: np.uint8, global_step: int, img_count: int= 100):
    # Read Parameters
    prompt = args.instance_prompt
    # load SD pipeline and generate fake images
    prompts = [prompt]
    fake_images = []
    for i in range(img_count):
        print(i)
        pred = pipeline(prompts, num_images_per_prompt=1, output_type="numpy").images[0]
        pred = [(pred * 255).astype(np.uint8)]
        fake_images.append(pred)
    del pipeline
    torch.cuda.empty_cache()
    fake_images = np.concatenate(fake_images, axis=0)
    # convert pill image to tensor
    fake_img_tensor = torch.from_numpy(fake_images).permute(0, 3, 1, 2)
    real_img_tensor = torch.from_numpy(real_images).permute(0, 3, 1, 2)
    # CLIP score 
    # clip = CLIPScore(model_name_or_path=weight_path)
    # clip.update(fake_img_tensor, prompts)
    # clip_score = clip.compute()
    # inception score(IS) score
    print("Inception score calculation")
    inception = InceptionScore()
    inception.update(fake_img_tensor)
    is_score = inception.compute()
    del inception
    
    # FrechetInceptionDistance(FID) score
    print("FID score calculation")
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_img_tensor, real=True)
    fid.update(fake_img_tensor, real=False)
    fid_score = fid.compute()
    del fid
    
    # KernelInceptionDistance(KID) score
    print("KID score calculation")
    kid = KernelInceptionDistance(subsets= 1, subset_size= img_count)
    kid.update(real_img_tensor, real=True)
    kid.update(fake_img_tensor, real=False)
    kid_score = kid.compute()
    del kid
    # apply plots to dvc live log images
    title = f"FID_score:{round(fid_score.item(), 3)} , IS_score_mean_std:{(round(is_score[0].item(), 3), round(is_score[1].item(), 3))} , KID_score_mean_std:{(round(kid_score[0].item(), 3), round(kid_score[1].item(), 3))}"
    print("Saving facke images")
    save_images(fake_images, f"{global_step}_fake", title, live)
    # apply metrics to dvc live log metrics
    live.log_metric("FID_score", round(fid_score.item(), 3))
    live.log_metric("IS_score_mean", round(is_score[0].item(), 3))
    live.log_metric("IS_score_std", round(is_score[1].item(), 3))
    live.log_metric("KID_score_mean", round(kid_score[0].item(), 3))
    live.log_metric("KID_score_std", round(kid_score[1].item(), 3))
    live.log_metric("CLIP_score", -1)
    del fake_images, fake_img_tensor, real_img_tensor
    print("eval end")
    return {"fid_score": round(fid_score.item(), 3), 
            "IS_score_mean": round(is_score[0].item(), 3),
            "IS_score_std": round(is_score[1].item(), 3),
            "kid_score": round(kid_score[0].item(), 3),
            "KID_score_std": round(kid_score[1].item(), 3),
            "KID_score_mean": round(kid_score[0].item(), 3),
            "CLIP_score": -1}

   
if __name__ == "__main__":  
    # get parameters
    param_yaml_file = sys.argv[1]
    # param_yaml_file = "/home/naserwin/hamze/SD_optimization/params.yaml"
    params = yaml.safe_load(open(param_yaml_file))
    params = params["evaluation"]
    dataset_name = params["dataset_name"]
    args = Namespace(**params["evaluation"])
    img_count = 100
    # set parameers
    args.trained_model_path = os.path.join(args.trained_model_path, dataset_name)
    args.instance_data_dir = os.path.join(args.instance_data_dir, dataset_name)
    args.eval_path = os.path.join(args.eval_path, dataset_name)
    # sort checkpoints base on thire steps
    checkpoints = [int(i.replace("checkpoint-", "")) for i in os.listdir(args.trained_model_path) if i.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints)
    # evaluation proccess
    live = Live(args.eval_path, report="md", resume=True)
    live.log_params (params["train"])
    real_images = read_real_image(args.instance_data_dir, live)
    for ck_path in checkpoints:
        path = os.path.join(args.trained_model_path, f"checkpoint-{ck_path}")
        sd_pipeline = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False).to("cuda")
        eval_metrices(sd_pipeline, args, live, real_images, ck_path, img_count)
        live.next_step()
    live.end()
        

    
    


