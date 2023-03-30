import torch
from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler

def model_1(prompt_list,model_id=None):
    if model_id == None:
        model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)  
    pipe = pipe.to("cuda")
  #prompt = "a photograph of an astronaut riding a horse"
    for prompt in prompt_list:
        image = pipe(prompt).images[0]
        image_file = prompt + ".png"
        image.save(image_file)
    
def model_2(prompt_list,model_id=None):
    if model_id == None:
        model_id = "stabilityai/stable-diffusion-2-1"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    for prompt in prompt_list:
        image = pipe(prompt).images[0]
        image_file = prompt + ".png"
        image.save(image_file)
