from diffusers import AutoPipelineForText2Image
import torch
import os

class SD:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device="cuda", torch_dtype=torch.float16):
        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)

    def generate_and_save_image(self, prompt, height, width, num_inference_steps, guidance_scale, lora_weights, lora_scale, seed, num_imgs, tag):
        self.pipeline.unfuse_lora()  # Reset any previously applied LoRA modifications
        self.pipeline.unload_lora_weights()  # Ensure the model starts without any LoRA weights

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # lora_weights is a list of paths to LoRA weights and 
        # lora_scale is a corresponding list of scales for fuse_lora's lora_scale parameter
        if lora_weights:
            print("lora weights:", lora_weights)
            print("lora scale", lora_scale)
            for weight_path, scale in zip(lora_weights, lora_scale):
                # Load and apply each LoRA modification in sequence
                self.pipeline.load_lora_weights(weight_path)
                self.pipeline.fuse_lora(scale)

        # Generate images with potentially modified model
        images = self.pipeline(prompt, generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width, num_images_per_prompt=num_imgs).images

        # Define directory for saving images
        clean_prompt = "".join(e for e in prompt if e.isalnum())  # Simplify prompt to alphanumeric for folder naming
        folder_path = os.path.join("generated_images", tag, clean_prompt)  
        os.makedirs(folder_path, exist_ok=True)

        # Save images with index
        for i, image in enumerate(images):
            image_path = os.path.join(folder_path, f"{tag}_{i+1}.png")
            image.save(image_path)
            print(f"Image saved at {image_path}")

        # Clean up after generation to ensure no residual LoRA settings
        self.pipeline.unfuse_lora()
        self.pipeline.unload_lora_weights()

