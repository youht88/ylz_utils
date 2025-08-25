def text2image():
    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if  torch.mps.is_available() else \
             "cpu"
    pipe.to(device)

    #prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    prompt = "A beautiful woman with a red dress and a hat, in a cozy living room."
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    image.save("sdxl-turbo_text2image.png")
    
def image2image():
    from diffusers import AutoPipelineForImage2Image
    from diffusers.utils import load_image
    import torch

    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if  torch.mps.is_available() else \
             "cpu"
    pipe.to(device)

    init_image = load_image("https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png").resize((512, 512))

    prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

    image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
    image.save("sdxl-turbo_image2image.png")

if __name__ == "__main__":
    image2image()