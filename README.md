# LoRA Adaptation for Stable Diffusion Using Docker

This README serves as a comprehensive guide for understanding, implementing, and evaluating the LoRA-adapted Stable Diffusion model using Docker. For detailed instructions and insights, refer to the respective sections in the pdf documentation: **Documentation_Stable_Diffusion_LoRA_Project.pdf.**

Check out my Medium article here: 
https://medium.com/@weimaychen2/revolutionizing-digital-creativity-harnessing-lora-adapted-stable-diffusion-within-docker-for-3f0419c6612e

## Overview
This project focuses on adapting the Stable Diffusion model through LoRA integration using Docker. The adaptation aims to enhance the generation of images based on textual prompts. This README provides an extensive guide on utilizing and understanding the adapted model.

## Quick Demo
**landscape.ipynb** and **single_object.ipynb** explore Single LoRa and multiple LoRA integration with Stable Diffusion model

## Files and Folders
### Files
config.yaml: Configuration file for base_model and LoRAs.
Dockerfile: Used to build and run the Docker image.
main.py: Main pipeline for generating images.
landscape.ipynb: Jupyter notebook for demo with the prompt "A beautiful sky".
single_object.ipynb: Jupyter notebook for demo with the prompt "A green pokemon with blue eyes".
Evaluation_img_generation.ipynb: Jupyter notebook for evaluating generated images.


### Folders
base_model: Contains the Stable diffusion + LoRA integration model class.
generated_images: Stores generated images, organized by text prompts.
LoRAs: Contains LoRA model weights in .safetensors format.

## Specific Task(s) within Game Creation
The adapted model can be applied in game creation for:
- Facilitating Asset Creation: Converting text into 2D images, which can further be transformed into 3D assets.
- Concept Ideation: Generating visual themes for story concepts and ideas.
- Stable Diffusion Model and LoRA Integration Process

## General Overview
The main process of generating images involves:

1. Loading pretrained Stable Diffusion model.
2. Loading LoRA model weights into the Stable Diffusion model.
3. Generating images based on input text prompts.

## Extended Detailed Process
The process includes loading LoRA weights, parameter adjustments, and image generation. Optional steps for adjusting LoRA impact are also explained.

## LoRA Integration Results
Integration results for single and multiple LoRAs are discussed. It's noted that including LoRA tags in text prompts generally improves results.

### Examples:
"A green pokemon with blue eyes", pixel LoRA:

![basepixel](https://github.com/solarspaceclouds/StableDiffusion-with-LoRA-integration/assets/65459827/905542d6-ac57-4891-a781-5e8972c5e0bf)

"A beautiful sky", easter and jellyfish forest LoRA

![easter jellyfishforest](https://github.com/solarspaceclouds/StableDiffusion-with-LoRA-integration/assets/65459827/56bd6ce4-bf4c-4f2e-96c9-10e3d369c1ac)

## Parameter Adjustments: StableDifusionPipeline
Key parameters include: text prompt, num_inference_steps, and guidance_scale. 

Optimal balance requires experimentation.

## Running the Code
1. Download LoRAs in **safetensors** format and save them to LoRAs folder in the current directory
2. Instructions for setting up the environment and running the code via examples are provided.
   Refer to config.yaml for the
- List of base_models (currently only 1): runwayml/stable-diffusion-v1-5
- List of LoRA model names: easter_egg, basepixel, jellyfish_forest, wanostyle, moxinstyle
  
  The following example defines all possible available arguments (does not use any default arguments)
  ```
  python3 main.py --text-prompt "An underwater adventure" --lora-name "jellyfish_forest" --fuse-lora-scale 0.9 --height 1024 --width 640 --num-inference-steps 17 --guidance-scale 8.4 --seed-num 13 --num-imgs 3
  ```
4. Docker containerization steps are also outlined.

``` 
docker build -t stable-diffusion-lora .
```
```
docker run --gpus all -p 8080:8080 stable-diffusion-lora
```
```
docker save stable-diffusion-lora:latest > stable-diffusion-lora.tar
```
```
docker load < stable-diffusion-lora.tar
```
```
docker run --gpus all -v $(pwd)/generated_images:/app/generated_images stable-diffusion-lora:latest --text-prompt "An underwater adventure" --lora-name "jellyfish_forest" --fuse-lora-scale 0.9 --height 1024 --width 640 --num-inference-steps 17 --guidance-scale 8.4 --seed-num 13 --num-imgs 3
```

## Evaluation Strategy for the Adapted Model
Comprehensive evaluation strategies, including qualitative (creativity/novelty, visual quality) and quantitative (Inception Score) approaches, are detailed in the documentation and **evaluate_img_generation.ipynb** notebook.

