# LoRA Adaptation for Stable Diffusion Using Docker

This README serves as a comprehensive guide for understanding, implementing, and evaluating the LoRA-adapted Stable Diffusion model using Docker. For detailed instructions and insights, refer to the respective sections in the pdf documentation: Documentation_Stable_Diffusion_LoRA_Project.pdf.

## Overview
This project focuses on adapting the Stable Diffusion model through LoRA integration using Docker. The adaptation aims to enhance the generation of images based on textual prompts. This README provides an extensive guide on utilizing and understanding the adapted model.

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

## Parameter Adjustments: StableDifusionPipeline
key parameters include: text prompt, num_inference_steps, and guidance_scale. 

Optimal balance requires experimentation.

## Running the Code
Instructions for setting up the environment and running the code via examples are provided. Docker containerization steps are also outlined.

## Evaluation Strategy for the Adapted Model
Comprehensive evaluation strategies, including qualitative and quantitative approaches, are detailed. This includes assessing visual quality, creativity, automated metrics, stability, and scalability.

