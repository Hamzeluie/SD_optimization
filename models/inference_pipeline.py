from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append("model")
# sys.path.append("SD")
# sys.path.append("utils")
from .SD.txt2img import Text2Image
from .SD.inpaint import Inpaint
from .inference import MultiSubjectInpaint
from .utils.process import InpaintProcess, postprocess

class Pipeline():
    """Main pipe line class"""
    def __init__(self, weight_path, converted_path) -> None:
        '''Initializing and loading all of the models'''
        # self.inpaint_sd = Inpaint()
        self.multi_subject_inpaint = MultiSubjectInpaint(weight_path,converted_path)

    def generate(self,
                 prompt: str,
                 image,
                 mask,
                 padding = 0,
                 blur_len = 9,
                 strength=0.75,
                 CFG_Scale_slider=13,
                 transparency=0.5,
                 num_inference_steps=150,
                 negative_prompt=""):
        ''' inference 
        prompt: input text prompt
        image: PIL type input image 
        mask: PIL type 1 channel input mask. the shape of the image and mask should be the same
        return: PIL type inpainted image without resizing and losing resolution
        '''
        print("Input prompt>> ", prompt)        
        # cropping around the masks
        process = InpaintProcess(image, mask, padding, blur_len)
        img_cropped, msk_cropped, paste_loc = process.crop_padding()
        img_cropped_512x512 = process.resize(img_cropped,[(512,512)]*len(img_cropped))
        msk_cropped_512x512 = process.resize(msk_cropped,[(512,512)]*len(msk_cropped))
        # img_cropped, msk_cropped, msk_centers = crop_512x512(image, mask)
        img512x512_result = []
        for img, msk in zip(img_cropped_512x512, msk_cropped_512x512):
            # img and msk shape is 512x512
            sd_result = self.multi_subject_inpaint.generate(prompt=prompt, 
                                                            image=img, 
                                                            mask=msk, 
                                                            strength_slider=strength, 
                                                            CFG_Scale=CFG_Scale_slider,
                                                            num_inference_steps=num_inference_steps,
                                                            negative_prompt=negative_prompt)
            sd_result = postprocess(img, msk, sd_result, transparency)
            ## for debugging
            # sd_result.save("sd_result.png")
            # img.save("input.png")
            # msk.save("mask.png")
            img512x512_result.append(sd_result)
        # merging cropped regions
        # img_cropped_generated = process.resize(img512x512_result,[(loc[2], loc[3]) for loc in paste_loc])
        image_result = process.merge_cropping(img512x512_result, paste_loc)
        return image_result