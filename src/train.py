from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append("model")
from models.stable_deffussion_mlops import TrainMultiSubjectSD
import os
import yaml


def setParameters(params):
        # params = json.load(args)
        data = {
            # Important
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5" if params["pretrained_model_name_or_path"] is None else params["pretrained_model_name_or_path"],
            "resolution": 512 if params["resolution"] is None else params["resolution"],
            "train_batch_size": 1 if params["train_batch_size"] is None else params["train_batch_size"],
            "max_train_steps": 2000 if params["max_train_steps"] is None else params["max_train_steps"],
            "checkpointing_steps": 1000 if params["checkpointing_steps"] is None else params["checkpointing_steps"],
            "num_train_epochs": 1 if params["num_train_epochs"] is None else params["num_train_epochs"],
            "sample_batch_size": 4 if params["sample_batch_size"] is None else params["sample_batch_size"],                     
            "trained_model_path": "multi-subject-model" if params["trained_model_path"] is None else params["trained_model_path"],     
            "instance_prompt": params["instance_prompt"],
            "instance_data_dir": params["instance_data_dir"],
            "dataset_name": params["dataset_name"],
            "eval_index": params["eval_index"],
            "eval_index_max": params["eval_index_max"],
            "eval_path": params["eval_path"],
            
          # Can be changed but default values set.
            "adam_beta1": 0.9 if params["adam_beta1"] is None else params["adam_beta1"],
            "adam_beta2": 0.999 if params["adam_beta2"] is None else params["adam_beta2"],
            "adam_weight_decay": 1e-2 if params["adam_weight_decay"] is None else params["adam_weight_decay"],
            "adam_epsilon": 1e-08 if params["adam_epsilon"] is None else params["adam_epsilon"],
            "max_grad_norm": 1.0 if params["max_grad_norm"] is None else params["max_grad_norm"],
            "learning_rate": 5e-6 if params["learning_rate"] is None else params["learning_rate"],
            # "scale_lr": False if params["scale_lr"] else params["scale_lr"],
            # "lr_scheduler": "constant" if params["lr_scheduler"] is None else params["lr_scheduler"],
            "lr_warmup_steps": 500 if params["lr_warmup_steps"] else params["lr_warmup_steps"],
            "lr_num_cycles": 1 if params["lr_num_cycles"] is None else params["lr_num_cycles"],
            "lr_power": 1.0 if params["lr_power"] is None else params["lr_power"],
            "with_prior_preservation": False if params["with_prior_preservation"] is None else params["with_prior_preservation"],
          
          #   "logging_dir": "logs" if params["logging_dir"] is None else params["logging_dir"],
          #   "report_to": "tensorboard" if params["report_to"] is None else params["report_to"],
          #   "prior_loss_weight": 1.0 if params["prior_loss_weight"] is None else params["prior_loss_weight"],
          #   "num_class_images": 100 if params["num_class_images"] is None else params["num_class_images"],
          #   "center_crop": False if params["center_crop"] is None else params["center_crop"],            
            # "adam_beta1": 0.9 ,
            # "adam_beta2": 0.999,
            # "adam_weight_decay": 1e-2 ,
            # "adam_epsilon": 1e-08 ,
            # "max_grad_norm": 1.0 ,
            # "learning_rate": 5e-6 ,
            "scale_lr": False ,
            "lr_scheduler": "constant" ,
            # "lr_warmup_steps": 500 ,
            # "lr_num_cycles": 1 ,
            # "lr_power": 1.0 ,
            
            "logging_dir": "logs" ,
            "report_to": "tensorboard" ,
                        
            "prior_loss_weight": 1.0 ,
            "num_class_images": 1 ,
            "center_crop": False ,
            "sample_batch_size": 4,
            
            # "num_train_epochs": 1,
            "checkpoints_total_limit": None,
            "gradient_accumulation_steps": 1 ,
            
            # None
            "checkpoint_path": None,
            "allow_tf32": None,
            "mixed_precision": None,
            "enable_xformers_memory_efficient_attention": None,
            "push_to_hub": None,
            "hub_token": None,
            "hub_model_id": None,
            "revision": None,
            "tokenizer_name": None,
            "class_data_dir": params["instance_data_dir"],
            
            "class_prompt": params["instance_prompt"],
            
            "seed": None,
            "train_text_encoder": None,
            "resume_from_checkpoint": False,
            "use_8bit_adam": None,
            "gradient_checkpointing": None,
            "prior_generation_precision": None,
            "local_rank": None,
            "half": None,
            "use_safetensors": None
        }

        if params["instance_data_dir"] is None:
            raise ValueError("Specify `instance_data_dir`")

        if params["instance_prompt"] is None:
            raise ValueError("Specify `instance_prompt`")

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != params["local_rank"]:
            params["local_rank"] = env_local_rank

        # print("***************************************************", params["with_prior_preservation"])
        # if params["with_prior_preservation"]:
        #     if params["class_data_dir"] is None:
        #         raise ValueError("You must specify a data directory for class images.")
        #     if params["class_prompt"] is None:
        #         raise ValueError("You must specify prompt for class images.")
        # else:
        #     # logger is not available yet
        #     if params["class_data_dir"] is not None:
        #         warnings.warn("You need not use class_data_dir without with_prior_preservation.")
        #     if params["class_prompt"] is not None:
        #         warnings.warn("You need not use class_prompt without with_prior_preservation.")
        return data
   
   
if __name__ == "__main__":
    # get parameters
    # param_yaml_file = sys.argv[1]
    param_yaml_file = "/home/naserwin/hamze/SD_optimization/params.yaml"
    params = yaml.safe_load(open(param_yaml_file))
    params = params["train"]
    dataset_name = params["dataset_name"]
    # Set parameters
    trained_model_path = os.path.join(Path(__file__).parents[1].as_posix(),
                                      params["trained_model_path"],
                                      "trained_model", dataset_name)
    os.makedirs(trained_model_path, exist_ok=True)
    params["trained_model_path"] = trained_model_path  
    params["instance_data_dir"] = os.path.join(params["instance_data_dir"], dataset_name)
    trainObject  = TrainMultiSubjectSD()
    data = setParameters(params)
    # train model
    trainObject.train(data)
    