from transformers import pipeline
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def model_1(img_link):
  image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
  gen_text = image_to_text(img_link)
  return gen_text[0]["generated_text"]
