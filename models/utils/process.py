from PIL import Image, ImageFilter
import cv2
import numpy as np
from typing import List


class InpaintProcess():
    '''preprocess and postprocess class of inpainting'''
    def __init__(self, 
                 base_image, 
                 base_mask,
                 padding: int,
                 blur_len: int
              ) -> None:
        ''' Initilizing InpaintProcess
        base_image: PIL type original base image
        base_mask: PIL type 1 channel base mask
        padding: crop padding size
        blur_len: blur filter kernel size when merging cropped images with base image
        '''
        self.padding = padding
        self.base_image = base_image
        self.base_mask = base_mask
        # Blurring the mask
        self.blurred_base_mask = base_mask.filter(ImageFilter.BoxBlur(blur_len))
        self.blur_len = blur_len
        self.paste_to = None
        self.image_cropped = []
        self.mask_cropped = []

    def crop_padding(self):
        '''For Cropping image around the mask by considering padding
        return: [image_cropped, mask_cropped, mask_centers]
        '''
        msk_np = np.array(self.base_mask, dtype=np.uint8)
        if len(msk_np.shape) > 2:
            msk_np = cv2.cvtColor(msk_np, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(msk_np, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)
        paste_loc = []
        image_cropped = []
        mask_cropped = []
        W, H = self.base_image.size
        for c in contours:
            if cv2.contourArea(c)< 2:
                continue
            M = cv2.moments(c)
            xc = int(M['m10']/M['m00'])
            yc = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(c) # x, y: up left corner
            w=h=max(h,w)
            x, y = xc - w//2, yc - h//2
            if x - self.padding < 0:
                x = self.padding
            if y - self.padding < 0:
                y = self.padding
            if x + w + self.padding > W:
                w = W - x - self.padding
            if y + h + self.padding > H:
                h = H - y - self.padding
            x = x - self.padding
            y = y - self.padding
            w = w + 2*self.padding
            h = h + 2*self.padding
            paste_loc.append([x, y, w, h])
            image_cropped.append(self.base_image.crop((x, y, x+w, y+h)))
            mask_cropped.append(self.blurred_base_mask.crop((x, y, x+w, y+h)))
        self.paste_to = paste_loc
        self.image_cropped = image_cropped
        self.mask_cropped = mask_cropped
        return image_cropped, mask_cropped, paste_loc
    def resize(self, images_list, new_sizes):
        '''In this function, cropped images are resized to the new_size
        Mostly is used for upscaling images to 512x512 or the input size of 
        the model and down scaling to the original
        '''
        images_new = []
        for img, new_size in zip(images_list, new_sizes): 
            images_new.append(img.resize(new_size))
        return images_new

    def merge_cropping(self, image_cropped: List, paste_to):
        ''' Merging cropped imaged, resize and merge
        imaged_cropped: List of PIL type cropped images
        return merged image
        '''
        image = self.base_image
        for loc, img_crop in zip(paste_to, image_cropped):
            image.paste(img_crop.resize((loc[2],loc[3])), (loc[0], loc[1]))
        return image

def crop_512x512(image, mask):
    '''For Cropping image
    image: PIL type input original image
    mask: PIL type 1 channel mask image
    return: [image_cropped, mask_cropped, mask_centers] --> 
    '''
    msk_np = np.array(mask, dtype=np.uint8)
    if len(msk_np.shape) > 2:
        msk_np = cv2.cvtColor(msk_np, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(msk_np, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_NONE)
    mask_centers = []
    image_cropped = []
    mask_cropped = []
    w, h = image.size
    for i in contours:
        M = cv2.moments(i)
        xc = int(M['m10']/M['m00'])
        yc = int(M['m01']/M['m00'])
        if xc-256 < 0:
            xc = 256
        if yc-256 < 0:
            yc = 256
        if xc+256 > w:
            xc = w - 256
        if yc+256 > h:
            yc = h - 256
        mask_centers.append([xc, yc])
        image_cropped.append(image.crop((xc-256, yc-256, xc+256, yc+256)))
        mask_cropped.append(mask.crop((xc-256, yc-256, xc+256, yc+256)))
    return image_cropped, mask_cropped, mask_centers

def postprocess(image, mask, predicted_image, transparency):
    ''' This function is merging the generated inpainted and the original image
    image: PIL type original image
    mask: PIL type 1 channel mask image
    predicted_image: PIL type generated image
    transparency: float between 0~1. the effect of the generated image in the output result
    predicted_image: PIL type generated image
    '''
    predicted_image_np = np.asarray(predicted_image)
    mask_np = np.asarray(mask)
    mask_np = mask_np.copy()
    image_np = np.asarray(image)
    origin_image = image_np.copy()
    mask_np = mask_np / 255
    # inpainted_np = (1-mask_np)*image_np+mask_np*predicted_image_np
    predicted_image_np = cv2.addWeighted(image_np.astype(np.float64),1-transparency,predicted_image_np.astype(np.float64),transparency,0)
    inpainted_np = (1-mask_np)*origin_image+mask_np*predicted_image_np.astype(np.uint8)

    return Image.fromarray(inpainted_np.astype(np.uint8))


