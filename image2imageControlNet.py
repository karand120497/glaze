import cv2
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_utils import ade_palette
#from diffusers.utils import load_image

def canny_image(image):
  image = np.array(image)
  low_threshold = 100
  high_threshold = 200

  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)
  return canny_image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
  
  def control_net(image,prompt_list,mode=None):
  if mode == "canny" or mode == None:
    model_id_controlnet = "lllyasviel/sd-controlnet-canny"
    model_id_diffusion = "runwayml/stable-diffusion-v1-5"
    controlnet = ControlNetModel.from_pretrained(model_id_controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id_diffusion, controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    image_mod = canny_image(image)
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt_list))]      
    output = pipe(prompt_list,image_mod,negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"]*len(prompt_list),generator=generator,num_inference_steps=20)
    image_grid(output.images,1,1)
  elif mode == "pose":
    model_id_pose = "lllyasviel/ControlNet"
    model_id_controlnet = "fusing/stable-diffusion-v1-5-controlnet-openpose"
    model_id_diffusion = "runwayml/stable-diffusion-v1-5"
    model_pose = OpenposeDetector.from_pretrained(model_id_pose)
    poses = model_pose(image)
    controlnet = ControlNetModel.from_pretrained(model_id_controlnet,torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id_diffusion,controlnet=controlnet,torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt_list))]      
    output = pipe(prompt_list,poses,negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"]*len(prompt_list),generator=generator,num_inference_steps=20)
    image_grid(output.images,1,1)
  elif mode == "seg":
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    #image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png").convert('RGB')
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
      outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
      color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    image = Image.fromarray(color_seg)
    controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt_list))] 
    output = pipe(prompt_list,poses,negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"]*len(prompt_list),generator=generator,num_inference_steps=20)
    image_grid(output.images,1,1)
  return output.images

# #main func
# print("Please provide image location :")
# img_link = input()
# image = cv2.imread(img_link)
# prompt = ["Keep the main subject and all foreground elements of the input image intact, but replace the background with a new, visually appealing scene that complements the subject's colors, lighting, and overall composition. Ensure that the new background merges seamlessly with the foreground elements, creating a natural and cohesive final image."]
# gen_image = control_net(image,prompt)
