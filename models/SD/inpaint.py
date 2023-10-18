from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

class Inpaint():
    """Stable Diffusion Inpainting Inference"""
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting") -> None:
        """"Initializing and loading the model
        model_id: [str] path to the checkpoint 
        checkpoint are runwayml/stable-diffusion-v1-5  and 
        runwayml/stable-diffusion-inpainting and 
        LarryAIDraw/v1-5-pruned-emaonly
        """
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, 
                                                              revision="fp16", 
                                                              torch_dtype=torch.
                                                              float32,
                                                              strength=0.75
                                                              )
        self.pipe = pipe.to("cuda")
        print("Inpainting SD is loaded.")

    def generate(self, prompt: str, image, mask, strength_slider:float, CFG_Scale:float):
        ''' inference 
        prompt: input text prompt
        image: PIL type input image 
        mask: PIL type 1 channel input mask. the shape of the image and mask should be the same
        return: PIL type inpainted image 
        '''
        predicted_image = self.pipe(prompt=prompt, 
                                    image=image, 
                                    mask_image=mask, 
                                    guidance_scale=20,
                                    num_inference_steps=150,
                                    strength=strength_slider
                                    ).images[0]
        predicted_image.save("predicted_image.png")
        return predicted_image

if __name__ == "__main__":
    pipe = StableDiffusionInpaintPipeline.from_pretrained("/home/rteam1/faryad/General_Generative_Defect/results/save_converted_model",
                                                          strength=0.75,
                                                          revision="fp16",
                                                          torch_dtype=torch.float32,
                                                        #   safety_checker=None
                                                          )
    pipe = pipe.to("cuda")
    prompt = "a photo of a ##Scratches@mtl## defect screw"
    img = Image.open("/home/rteam1/faryad/General_Generative_Defect/input.png")
    msk = Image.open("/home/rteam1/faryad/General_Generative_Defect/mask.png")
    result = pipe(prompt=prompt, image=img, mask_image=msk).images[0]
    result.save("single_result.png")
