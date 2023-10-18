from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


class Text2Image():
    '''Text to Image Stable Diffusion v2 inferencing'''
    def __init__(self, model_id="stabilityai/stable-diffusion-2") -> None:
        '''Initilizing and loading the model
        model_id: [str] path to the checkpoint 
        '''
        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipe = pipe.to("cuda")
        print("Text to Image SD is loaded.")

    def generate(self, prompt: str):
        ''' inference 
        prompt: input text prompt
        return: PIL type generated image 
        '''
        image = self.pipe(prompt).images[0]
        # image.save("astronaut_rides_horse.png")
        return image
    

