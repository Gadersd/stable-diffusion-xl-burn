# Stable-Diffusion-XL-Burn

Stable-Diffusion-XL-Burn is a Rust-based project which ports stable diffusion xl into the Rust deep learning framework burn. This repository is licensed under the MIT Licence.

## How To Use

### Step 1: Download the Model and Set Environment Variables

The model files must be in burn's format. Eventually a python script will be provided to convert any SDXL model to burn's format. 
Start by downloading the pre-converted SDXL1.0 model files provided on HuggingFace.

```bash
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.cfg -P ./SDXL1.0/
```

### Step 2: Run the Sample Binary

Invoke the sample binary provided in the rust code. You will need a CUDA GPU with at least 10 GB of VRAM.

```bash
export TORCH_CUDA_VERSION=cu113
# Arguments: <model> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image>
cargo run --release --bin sample SDXL1.0 7.5 30 "An elegant bright red crab." crab
```

This command will generate an image according to the provided prompt, which will be saved as 'crab0.png'.

![An image of an ancient mossy stone](crab0.png)

## License

This project is licensed under MIT license.

We wish you a productive time using this project. Enjoy!
