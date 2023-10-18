import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
import sys
sys.path.append(os. getcwd())

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from models.convert import convert


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

# logger = get_logger(__name__)

# if is_wandb_available():
#     import wandb

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = []
        self.instance_images_path = []
        self.num_instance_images = []
        self.instance_prompt = []
        self.class_data_root = []
        self.class_images_path = []
        self.num_class_images = []
        self.class_prompt = []
        self._length = 0

        for i in range(len(instance_data_root)):
            self.instance_data_root.append(Path(instance_data_root[i]))
            if not self.instance_data_root[i].exists():
                 raise ValueError("Instance images root doesn't exists.")

            self.instance_images_path.append(list(Path(instance_data_root[i]).iterdir()))
            self.num_instance_images.append(len(self.instance_images_path[i]))
            self.instance_prompt.append(instance_prompt[i])
            self._length += self.num_instance_images[i]

            if class_data_root is not None:
                self.class_data_root.append(Path(class_data_root[i]))
                self.class_data_root[i].mkdir(parents=True, exist_ok=True)
                self.class_images_path.append(list(self.class_data_root[i].iterdir()))
                self.num_class_images.append(len(self.class_images_path))
                if self.num_class_images[i] > self.num_instance_images[i]:
                    self._length -= self.num_instance_images[i]
                    self._length += self.num_class_images[i]
                self.class_prompt.append(class_prompt[i])
            else:
                self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        for i in range(len(self.instance_images_path)):
            instance_image = Image.open(self.instance_images_path[i][index % self.num_instance_images[i]])
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example[f"instance_images_{i}"] = self.image_transforms(instance_image)
            example[f"instance_prompt_ids_{i}"] = self.tokenizer(
                self.instance_prompt[i],
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        if self.class_data_root:
            for i in range(len(self.class_data_root)):
                class_image = Image.open(self.class_images_path[i][index % self.num_class_images[i]])
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                example[f"class_images_{i}"] = self.image_transforms(class_image)
                example[f"class_prompt_ids_{i}"] = self.tokenizer(
                    self.class_prompt[i],
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids

        return example

def collate_fn(num_instances, examples, with_prior_preservation=False):
    input_ids = []
    pixel_values = []

    for i in range(num_instances):
        input_ids += [example[f"instance_prompt_ids_{i}"] for example in examples]
        pixel_values += [example[f"instance_images_{i}"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        for i in range(num_instances):
            input_ids += [example[f"class_prompt_ids_{i}"] for example in examples]
            pixel_values += [example[f"class_images_{i}"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class}")