# Stable-Diffusion-XL-Burn

Stable-Diffusion-XL-Burn is a Rust-based project which ports stable diffusion xl into the Rust deep learning framework burn. This repository is licensed under the MIT Licence.

## How To Use

### Step 1: Download the Model and Set Environment Variables

The model files must be in burn's format.
A later section explains how you can convert any SDXL model to burn's format.
Start by downloading the pre-converted SDXL1.0 model files provided on HuggingFace.

```bash
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.mpk -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.mpk -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.mpk -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.cfg -P ./SDXL1.0/

# if you want to use the refiner
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/refiner.mpk -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/refiner.cfg -P ./SDXL1.0/
```

### Step 2: Run the Sample Binary

Invoke the sample binary provided in the rust code. You will need a CUDA GPU with at least 10 GB of VRAM.

```bash
cargo run --release --bin sample -- --model-dir SDXL1.0 --output-dir ./ --prompt "An elegant bright red crab."
```

This command will generate an image according to the provided prompt, which will be saved as '0.png'.

![An image of a crab](crab0.png)

One can also perform inpainting. Use a large number of steps to achieve the best coherence.

```bash
cargo run --release --bin sample \
  -- --model-dir SDXL1.0 \
     --output-dir ./inpainted \
     --prompt "An crab below a bright blue shining ocean." \
     --reference-img 0.png \
     --n-diffusion-steps 100 \
     --crop-left 0 \
     --crop-right 1024 \
     --crop-top 0 \
     --crop-bottom 200
```

![An image of a crab with the ocean visible](inpainted0.png)

Execute the following to see all available options.

```bash
cargo run --release --bin sample -- --help
```

### Converting Model Files to Burn's Format

Download the Python generative-models repo and install the dependencies.

```bash
git clone https://github.com/Stability-AI/generative-models.git
cd generative-models

# download weights
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true -P ./checkpoints/
wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors?download=true -P ./checkpoints/

source .pt2/bin/activate
git checkout 477d8b9 # only some older versions of the repo are compatible
python3 -m venv .pt2
pip3 install -r requirements/pt2.txt
pip3 install numpy
pip3 install torch
pip3 install pytorch_lightning
pip3 install omegaconf
pip3 install safetensors
pip3 install kornia
pip3 install open_clip_torch
pip3 install einops
pip3 install transformers
pip3 install scipy
pip3 install invisible-watermark
```

Now copy the scripts in the python directory of this project into the `generative-models` directory and execute

```bash
python3 dump.py
```

Wait until it finishes and then move the newly created `params` folder into this project's directory. Then execute

```bash
cargo run --release --bin convert params
mkdir SDXL && mv *.mpk SDXL/
```

Download the configuration files with

```bash
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/diffuser.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/embedder.cfg -P ./SDXL1.0/
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/latent_decoder.cfg -P ./SDXL1.0/

# if you want to use the refiner
wget https://huggingface.co/Gadersd/stable-diffusion-xl-burn/resolve/main/SDXL1.0/refiner.cfg -P ./SDXL1.0/
```

Now run the following to generate an image

```bash
export TORCH_CUDA_VERSION=cu113
# Arguments: <model> <refiner(y/n)> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image>
cargo run --release --bin sample SDXL n 7.5 30 "fireworks" celebration
```

## License

This project is licensed under MIT license.

We wish you a productive time using this project. Enjoy!
