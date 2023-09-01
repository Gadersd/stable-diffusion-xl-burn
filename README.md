# Stable-Diffusion-XL-Burn

Stable-Diffusion-XL-Burn is a Rust-based project which ports stable diffusion xl into the Rust deep learning framework burn. This repository is licensed under the MIT Licence.

## How To Use

### Step 1: Download the Model and Set Environment Variables

The model files must be in burn's format.
A later section explains how you can convert any SDXL model to burn's format.
Start by downloading the pre-converted SDXL1.0 model files provided on HuggingFace.

```bash
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.cfg -P ./SDXL1.0/

# if you want to use the refiner
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/refiner.bin -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/refiner.cfg -P ./SDXL1.0/
```

### Step 2: Run the Sample Binary

Invoke the sample binary provided in the rust code. You will need a CUDA GPU with at least 10 GB of VRAM.

```bash
export TORCH_CUDA_VERSION=cu113
# Arguments: <model> <refiner(y/n)> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image>
cargo run --release --bin sample SDXL1.0 n 7.5 30 "An elegant bright red crab." crab
```

This command will generate an image according to the provided prompt, which will be saved as 'crab0.png'.

![An image of an ancient mossy stone](crab0.png)

### Converting Model Files to Burn's Format

The scripts in the python directory dump safetensor weights that can be converted to a format burn can load.
Follow the instructions at https://github.com/Stability-AI/generative-models to install Stability AI's python SDXL repo.
Then copy the scripts in the python diretory of this project into the `generative-models` folder and execute

```bash
python3 dump.py
```

A `params` folder will be created containing the dumped weights.
Move the `params` folder to the root folder of this project and run

```bash
cargo run --release --bin convert params
mkdir SDXL
mv *.bin SDXL
```
Copy the .cfg files into the `SDXL` folder which can be downloaded with wget as shown in the download section.
Now the models can be sampled as demonstrated above.

## License

This project is licensed under MIT license.

We wish you a productive time using this project. Enjoy!
