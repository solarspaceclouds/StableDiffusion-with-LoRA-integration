import argparse
import yaml
from base_model.sd import SD
import torch

def load_config(config_path="config.yaml"):
    """Load pipeline configuration from a YAML file."""
    with open(config_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def initialize_pipeline(model_name, device, torch_dtype):
    """Initialize a Stable Diffusion pipeline based on the specified configuration."""
    model = SD(model_name=model_name, device=device, torch_dtype=torch_dtype)
    return model


def gen_sd_LoRA_images(model, lora_config, prompt, height, width, num_inference_steps, guidance_scale, seed_num, num_images, tag):
    """Generate and save images using the initialized model and LoRA configuration."""
    lora_weights_path = [lora_config['lora_weights_path']]  # Expecting a list for input into the model method
    lora_fuse_scale = [lora_config['fuse_lora_scale']]  # Expecting a list for input into the model method
    
    # Generate and save the image
    model.generate_and_save_image(prompt=prompt, lora_weights=lora_weights_path, lora_scale=lora_fuse_scale, seed=seed_num, num_imgs=num_images, tag=tag, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
def main():
    parser = argparse.ArgumentParser(description="Generate images with specified LoRA configuration.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--lora-name", type=str, default = "easter_egg", required=False, help="Name of the LoRA model to use.")
    parser.add_argument("--base-model", type=str, default = "runwayml/stable-diffusion-v1-5", required=False, help="Name of the LoRA model to use.")
    parser.add_argument("--text-prompt", type=str, default = "easter, A pokemon with blue eyes", required=False, help = "input text prompt to generate image")
    parser.add_argument("--fuse-lora-scale", type=float, default = 1, required=False, help = "level of LoRA application (0.0-1.0)")
    parser.add_argument("--height", type=int, default = 512, required=False, help="desired height of output image (in px)")
    parser.add_argument("--width", type=int, default=512, required=False,help="desired width of output image (in px)")
    parser.add_argument("--num-inference-steps", type=int, default = 15, required=False, help="number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, required=False,help="desired guidance level on following the specified prompt")
    parser.add_argument("--seed-num", type=int, default=0, required=False, help = "seed number for generator to produce reproducible results")
    parser.add_argument("--num-imgs", type=int, default = 4, required=False, help = "number of images desired")
    args = parser.parse_args()


    config = load_config(args.config)
    base_models_config = config['base_model']
    loras_config = config['loras']
    
    selected_lora = next((lora for lora in loras_config if lora['name'] == args.lora_name), None)
    if not selected_lora:
        raise ValueError("Selected LoRA configuration not found.")

    selected_base_model = next((base_sd for base_sd in base_models_config if base_sd['model_name'] == args.base_model), None)
    if not selected_base_model:
        raise ValueError("Selected base model configuration not found.")

    base_model = initialize_pipeline(selected_base_model['model_name'], selected_base_model['device'], getattr(torch, selected_base_model['torch_dtype']))

    text_prompt = args.text_prompt if args.text_prompt else selected_lora['default_prompt']
    
    gen_sd_LoRA_images(base_model, selected_lora, text_prompt, args.height, args.width, args.num_inference_steps, args.guidance_scale, args.seed_num, args.num_imgs, tag=selected_lora['name'])

if __name__ == "__main__":
    main()
    

