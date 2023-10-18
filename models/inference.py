import sys, os
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

from .convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from diffusers import EulerDiscreteScheduler
# from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

class MultiSubjectInpaint():
    """Stable Diffusion Inpainting Inference"""
    def __init__(self, model_path, save_converted_model) -> None:
        """"Initializing and loading the model
        model_id: [str] path to the checkpoint 
        """
        self.generator = [torch.Generator(device="cuda").manual_seed(-1) for i in range(1)]
        if os.path.isfile(model_path):
            output_path = self.convert(model_path, num_in_channels=4, output_path=save_converted_model)
        else:
            output_path = model_path
        pipe = StableDiffusionInpaintPipeline.from_pretrained(output_path, 
                                                              revision="fp16", 
                                                              torch_dtype=torch.float16,
                                                              safety_checker=None
                                                            )
        self.pipe = pipe.to("cuda")
        # scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.scheduler = scheduler
        print("Multi Subject Inpainting SD is loaded.")

    def generate(self, prompt: str, image, mask, strength_slider:float, CFG_Scale:float, negative_prompt:str, num_inference_steps:int=50):
        ''' inference 
        prompt: input text prompt
        image: PIL type input image 
        mask: PIL type 1 channel input mask. the shape of the image and mask should be the same
        return: PIL type inpainted image 
        '''
        predicted_image = self.pipe(prompt=prompt,
                                    image=image,
                                    mask_image=mask,
                                    generator=self.generator, 
                                    guidance_scale=CFG_Scale,
                                    num_inference_steps=num_inference_steps,
                                    strength=strength_slider,
                                    negative_prompt=negative_prompt
                                    ).images[0]
        # predicted_image.save("predicted_image.png")
        return predicted_image
    def convert(self,
        model_path, 
        original_config_file=None, 
        num_in_channels=None,
        pipeline_type=None, # The pipeline type. One of 'FrozenOpenCLIPEmbedder', 'FrozenCLIPEmbedder', 'PaintByExample'
        image_size=None, #"The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Siffusion v2  Base. Use 768 for Stable Diffusion v2."
        device=None, # Device to use (e.g. cpu, cuda:0, cuda:1, etc.)
        stable_unclip=None, # Set if this is a stable unCLIP model. One of 'txt2img' or 'img2img'.
        stable_unclip_prior=None, # Set if this is a stable unCLIP txt2img model. Selects which prior to use. If `--stable_unclip` is set to `txt2img`, the karlo prior (https://huggingface.co/kakaobrain/karlo-v1-alpha/tree/main/prior) is selected by default."
        clip_stats_path=None, # Path to the clip stats file. Only required if the stable unclip model's config specifies `model.params.noise_aug_config.params.clip_stats_path`.
        controlnet=False, # Set flag if this is a controlnet checkpoint.
        half=False, # Save weights in half precision.
        scheduler_type="pndm", # Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']
        prediction_type="epsilon",
        # Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights
        #     or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield
        #     higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning.
        extract_ema=False, 
        upcast_attention=False, # Whether the attention computation should always be upcasted. This is necessary when running stable diffusion 2.1.
        from_safetensors=False, # If model_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        to_safetensors=False, # Whether to store pipeline in safetensors format or not.
        output_path="result" # Path to the output model.
        ):

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path=model_path,
            original_config_file=original_config_file,
            image_size=image_size,
            prediction_type=prediction_type,
            model_type=pipeline_type,
            extract_ema=extract_ema,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            from_safetensors=from_safetensors,
            device=device,
            stable_unclip=stable_unclip,
            stable_unclip_prior=stable_unclip_prior,
            clip_stats_path=clip_stats_path,
        )

        if half:
            pipe.to(torch_dtype=torch.float16)

        pipe.save_pretrained(output_path, safe_serialization=to_safetensors)
        return output_path


if __name__ == "__main__":
    weight_path = "/home/naserwin/hamze/General_Generative_Defect/multi_subject_SD/plastic_cracks3.ckpt"
    converted_path = "/home/naserwin/hamze/General_Generative_Defect/multi_subject_SD/results"
    model = MultiSubjectInpaint(weight_path, converted_path)

    # input = Image.open("/home/rteam1/faryad/General_Generative_Defect/input.png")
    # msk = Image.open("/home/rteam1/faryad/General_Generative_Defect/mask.png")
    # prmt = "a photo of a ##m@nimu_scrw## defect screw"
    # result = model.generate(prmt, input, msk)
    # result.save("test_result.png")