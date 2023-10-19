import os
from pathlib import Path
import subprocess
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
sys.path.append("model")
import gradio as gr
from models.inference_pipeline import Pipeline
import yaml


# param_yaml_file = sys.argv[1]
param_yaml_file = "/home/naserwin/hamze/SD_optimization/params.yaml"

params = yaml.safe_load(open(param_yaml_file))["app"]

weight_path = params["input_checkpoint_path"]
converted_path = params["output_checkpoint_path"]

pipeline = Pipeline(weight_path=weight_path, converted_path=converted_path)

def write_config(cof_dict):
    cofig = os.path.join(converted_path, "config.yaml")
    with open(cofig, 'w') as file:
        yaml.dump(cof_dict, file)
    file.close()
    
    
def defect_generate(prompt, dict, padding, blur_len, strength_slider, CFG_Scale_slider, transparency, num_inference_steps, inference_batch):
    inference_batch = int(inference_batch)
    init_img =  dict['image'].convert("RGB")
    mask_img = dict['mask'].convert("RGB")
    # image = generator.generate(prompt)
    images = []
    print("************", inference_batch)
    for _ in range(inference_batch):
        images.append(pipeline.generate(prompt,init_img,mask_img, padding, blur_len, strength_slider, CFG_Scale_slider, transparency=transparency, num_inference_steps=num_inference_steps))
    write_config(cof_dict= {"padding": padding, 
                "blur_len": blur_len, 
                "strength_slider": strength_slider,
                "CFG_Scale_slider": CFG_Scale_slider, 
                "transparency": transparency, 
                "num_inference_steps":num_inference_steps})
    return images


def grpu_proccess_kill():
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    utilization = subprocess.check_output(cmd, shell=True)
    utilization = utilization.decode("utf-8").strip().split("\n")
    utilization = [int(x.replace(" %", "")) for x in utilization]
    os.system(f"kill -9 {utilization[-1]}")
    return utilization

css = '''
#image_upload{min-height:800px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 800px}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(
    """
    # General Generative Defect
    Start typing below to see the output.
    """)
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="prompt..")
        greet_btn = gr.Button("Generate").style(full_width=False)
        close_btn = gr.Button("close").style(full_width=False)
    with gr.Row():
        strength_slider = gr.Slider(0, 1, 0.75, label="Denoising strength")
        CFG_Scale_slider = gr.Slider(1, 300, 13, label="CFG Scale")
        transparency = gr.Slider(0, 1, 0.5, label="transparency")
    with gr.Row():
        padding_slider = gr.Slider(0, 256, 32,label="Mask Padding")
        blur_slider = gr.Slider(1, 256, 9,label="Mask Blur")
        num_inference_steps = gr.Slider(1, 300, 150,label="num_inference_steps")
    inference_batch = gr.Number(value=1, minimum=1, maximum=5, label="inference batch")
    input_img = gr.Image(label="Image", elem_id="image_upload",type='pil', tool='sketch').style(height=800)
    output = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[3], rows=[2], object_fit="contain", height="auto")
    
    greet_btn.click(fn=defect_generate, inputs=[prompt, input_img, padding_slider, blur_slider, strength_slider, CFG_Scale_slider, transparency, num_inference_steps, inference_batch], outputs=output, api_name="General Generative Defect", )
    close_btn.click(grpu_proccess_kill)

demo.launch(share=False)