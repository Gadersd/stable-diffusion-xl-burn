pub mod load;

use burn::{
    config::Config, 
    module::{Module, Param},
    tensor::{
        backend::Backend,
        Tensor,
        Int, 
        Float, 
        BasicOps, 
        Data, 
        Distribution, 
    },
};

use num_traits::ToPrimitive;

use super::autoencoder::{Autoencoder, AutoencoderConfig};
use super::unet::{UNet, UNetConfig, conditioning_embedding};
use super::clip::{CLIP, CLIPConfig};
use crate::token::{Tokenizer, clip::SimpleTokenizer, open_clip::OpenClipTokenizer};

/*#[derive(Config)]
pub struct StableDiffusionConfig {

}

impl StableDiffusionConfig {
    pub fn init<B: Backend>(&self) -> StableDiffusion<B> {
        let n_steps = 1000;
        let alpha_cumulative_products = offset_cosine_schedule_cumprod::<B>(n_steps).into();

        let autoencoder = AutoencoderConfig::new().init();
        let diffusion = UNetConfig::new(2816, 4, 4, 320, 64, 2048).init();
        let clip = CLIPConfig::new(49408, 768, 12, 77, 12, false).init();

        StableDiffusion {
            n_steps, 
            alpha_cumulative_products, 
            autoencoder, 
            diffusion, 
            clip, 
        }
    }
}

#[derive(Module, Debug)]
pub struct StableDiffusion<B: Backend> {
    n_steps: usize, 
    alpha_cumulative_products: Param<Tensor<B, 1>>, 
    autoencoder: Autoencoder<B>, 
    diffusion: UNet<B>, 
    clip: CLIP<B>, 
}

impl<B: Backend> StableDiffusion<B> {
    pub fn sample_image(&self, context: Tensor<B, 3>, unconditional_context: Tensor<B, 2>, unconditional_guidance_scale: f64, n_steps: usize) -> Vec<Vec<u8>> {
        let [n_batch, _, _] = context.dims();

        let latent = self.sample_latent(context, unconditional_context, unconditional_guidance_scale, n_steps);
        self.latent_to_image(latent)
    }

    pub fn latent_to_image(&self, latent: Tensor<B, 4>) -> Vec<Vec<u8>> {
        let [n_batch, _, _, _] = latent.dims();
        let image = self.autoencoder.decode_latent(latent * (1.0 / 0.18215));

        let n_channel = 3;
        let height = 512;
        let width = 512;
        let num_elements_per_image = n_channel * height * width;

        // correct size and scale and reorder to 
        let image = (image + 1.0) / 2.0;
        let image = image
            .reshape([n_batch, n_channel, height, width])
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .mul_scalar(255.0);

        let flattened: Vec<_> = image.
            into_data().
            value;

        (0..n_batch).into_iter().map(|b| {
            let start = b * num_elements_per_image;
            let end = start + num_elements_per_image;

            flattened[start..end].into_iter().map(|v| v.to_f64().unwrap().min(255.0).max(0.0).to_u8().unwrap()).collect()
        }).collect()
    }

    pub fn sample_latent(&self, context: Tensor<B, 3>, unconditional_context: Tensor<B, 2>, unconditional_guidance_scale: f64, n_steps: usize) -> Tensor<B, 4> {
        let device = context.device();

        let step_size = self.n_steps / n_steps;

        let [n_batches, _, _] = context.dims();

        let gen_noise = || {
            Tensor::random([n_batches, 4, 128, 128], Distribution::Normal(0.0, 1.0)).to_device(&device)
        };

        let sigma = 0.0; // Use deterministic diffusion

        let mut latent = gen_noise();

        for t in (0..self.n_steps).rev().step_by(step_size) {
            let current_alpha: f64 = self.alpha_cumulative_products.val().slice([t..t + 1]).into_scalar().to_f64().unwrap();
            let prev_alpha: f64 = if t >= step_size {
                let i = t - step_size;
                self.alpha_cumulative_products.val().slice([i..i + 1]).into_scalar().to_f64().unwrap()
            } else {
                1.0
            };

            let sqrt_noise = (1.0 - current_alpha).sqrt();

            let timestep = Tensor::from_ints([t as i32]).to_device(&device);
            let pred_noise = self.forward_diffuser(latent.clone(), timestep, context.clone(), unconditional_context.clone(), unconditional_guidance_scale);
            let predx0 = (latent - pred_noise.clone() * sqrt_noise) / current_alpha.sqrt();
            let dir_latent = pred_noise * (1.0 - prev_alpha - sigma * sigma).sqrt();

            let prev_latent = predx0 * prev_alpha.sqrt() + dir_latent + gen_noise() * sigma;
            latent = prev_latent;
        }

        latent
    }

    fn forward_diffuser(&self, latent: Tensor<B, 4>, timestep: Tensor<B, 1, Int>, context: Tensor<B, 3>, unconditional_context: Tensor<B, 2>, unconditional_guidance_scale: f64) -> Tensor<B, 4> {
        let [n_batch, _, _, _] = latent.dims();
        //let latent = latent.repeat(0, 2);

        let unconditional_latent = self.diffusion.forward(
            latent.clone(), 
            timestep.clone(), 
            unconditional_context.unsqueeze().repeat(0, n_batch), 
            Tensor::zeros([1, 1]), 
        );

        let conditional_latent = self.diffusion.forward(
            latent, 
            timestep, 
            context, 
            Tensor::zeros([1, 1]), 
        );

        /*let latent = self.diffusion.forward(
            latent.repeat(0, 2), 
            timestep.repeat(0, 2), 
            Tensor::cat(vec![unconditional_context.unsqueeze::<3>(), context], 0)
        );

        let unconditional_latent = latent.clone().slice([0..n_batch]);
        let conditional_latent = latent.slice([n_batch..2 * n_batch]);*/

        unconditional_latent.clone() + (conditional_latent - unconditional_latent) * unconditional_guidance_scale
    }

    pub fn unconditional_context(&self, tokenizer: &SimpleTokenizer) -> Tensor<B, 2> {
        self.context(tokenizer, "").squeeze(0)
    }

    pub fn context(&self, tokenizer: &SimpleTokenizer, text: &str) -> Tensor<B, 3> {
        let device = &self.clip.devices()[0];
        let text = format!("<|startoftext|>{}<|endoftext|>", text);
        let tokenized: Vec<_> = tokenizer.encode(&text).into_iter().map(|v| v as i32).collect();

        self.clip.forward_hidden(Tensor::from_ints(&tokenized[..]).to_device(device).unsqueeze(), 11)
    }
}*/


#[derive(Config, Debug)]
pub struct LatentDecoderConfig {}

impl LatentDecoderConfig {
    pub fn init<B: Backend>(&self) -> LatentDecoder<B> {
        let autoencoder = AutoencoderConfig::new().init();
        LatentDecoder {
            autoencoder, 
        }
    }
}


#[derive(Module, Debug)]
pub struct LatentDecoder<B: Backend> {
    autoencoder: Autoencoder<B>, 
}

impl<B: Backend> LatentDecoder<B> {
    pub fn latent_to_image(&self, latent: Tensor<B, 4>) -> Vec<Vec<u8>> {
        let [n_batch, _, _, _] = latent.dims();
        let image = self.autoencoder.decode_latent(latent * (1.0 / 0.13025));

        let n_channel = 3;
        let height = 1024;
        let width = 1024;
        let num_elements_per_image = n_channel * height * width;

        // correct size and scale and reorder to 
        let image = (image + 1.0) / 2.0;
        let image = image
            .reshape([n_batch, n_channel, height, width])
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .mul_scalar(255.0);

        let flattened: Vec<_> = image.
            into_data().
            value;

        (0..n_batch).into_iter().map(|b| {
            let start = b * num_elements_per_image;
            let end = start + num_elements_per_image;

            flattened[start..end].into_iter().map(|v| v.to_f64().unwrap().min(255.0).max(0.0).to_u8().unwrap()).collect()
        }).collect()
    }
}
        


#[derive(Config, Debug)]
pub struct DiffuserConfig {}

impl DiffuserConfig {
    pub fn init<B: Backend>(&self) -> Diffuser<B> {
        let n_steps = 1000;
        let alpha_cumulative_products = Tensor::zeros([1]).into(); //offset_cosine_schedule_cumprod::<B>(n_steps).into();
        let diffusion = UNetConfig::new(2816, 4, 4, 320, 64, 2048).init();

        Diffuser {
            n_steps, 
            alpha_cumulative_products, 
            diffusion, 
        }
    }
}

#[derive(Module, Debug)]
pub struct Diffuser<B: Backend> {
    n_steps: usize, 
    pub alpha_cumulative_products: Param<Tensor<B, 1>>, 
    pub diffusion: UNet<B>, 
}

impl<B: Backend> Diffuser<B> {
    pub fn sample_latent(&self, conditioning: Conditioning<B>, unconditional_guidance_scale: f64, n_steps: usize) -> Tensor<B, 4> {
        let device = conditioning.context.device();

        let step_size = self.n_steps / n_steps;

        let [n_batches, _, _] = conditioning.context.dims();

        let gen_noise = || {
            Tensor::random([n_batches, 4, 128, 128], Distribution::Normal(0.0, 1.0)).to_device(&device)
        };

        let sigma = 0.0; // Use deterministic diffusion

        let mut latent = gen_noise();

        for t in (0..self.n_steps).rev().step_by(step_size) {
            let current_alpha: f64 = self.alpha_cumulative_products.val().slice([t..t + 1]).into_scalar().to_f64().unwrap();
            let prev_alpha: f64 = if t >= step_size {
                let i = t - step_size;
                self.alpha_cumulative_products.val().slice([i..i + 1]).into_scalar().to_f64().unwrap()
            } else {
                1.0
            };

            let sqrt_noise = (1.0 - current_alpha).sqrt();

            let timestep = Tensor::from_ints([t as i32]).to_device(&device);
            let pred_noise = self.forward_diffuser(latent.clone(), timestep, conditioning.clone(), unconditional_guidance_scale);
            let predx0 = (latent - pred_noise.clone() * sqrt_noise) / current_alpha.sqrt();
            let dir_latent = pred_noise * (1.0 - prev_alpha - sigma * sigma).sqrt();

            let prev_latent = predx0 * prev_alpha.sqrt() + dir_latent + gen_noise() * sigma;
            latent = prev_latent;
        }

        latent
    }

    fn forward_diffuser(&self, latent: Tensor<B, 4>, timestep: Tensor<B, 1, Int>, conditioning: Conditioning<B>, unconditional_guidance_scale: f64) -> Tensor<B, 4> {
        let [n_batch, _, _, _] = latent.dims();
        //let latent = latent.repeat(0, 2);

        let unconditional_latent = self.diffusion.forward(
            latent.clone(), 
            timestep.clone(), 
            conditioning.unconditional_context.unsqueeze().repeat(0, n_batch), 
            conditioning.unconditional_channel_context.unsqueeze().repeat(0, n_batch), 
        );

        let conditional_latent = self.diffusion.forward(
            latent, 
            timestep, 
            conditioning.context, 
            conditioning.channel_context, 
        );

        /*let latent = self.diffusion.forward(
            latent.repeat(0, 2), 
            timestep.repeat(0, 2), 
            Tensor::cat(vec![unconditional_context.unsqueeze::<3>(), context], 0)
        );

        let unconditional_latent = latent.clone().slice([0..n_batch]);
        let conditional_latent = latent.slice([n_batch..2 * n_batch]);*/

        unconditional_latent.clone() + (conditional_latent - unconditional_latent) * unconditional_guidance_scale
    }
}


#[derive(Clone, Debug)]
pub struct Conditioning<B: Backend> {
    pub unconditional_context: Tensor<B, 2>, 
    pub context: Tensor<B, 3>, 
    pub unconditional_channel_context: Tensor<B, 1>, 
    pub channel_context: Tensor<B, 2>, 
}


#[derive(Config, Debug)]
pub struct EmbedderConfig {}

impl EmbedderConfig {
    pub fn init<B: Backend>(&self) -> Embedder<B> {
        let clip = CLIPConfig::new(49408, 768, 768, 12, 77, 12, true).init();
        let open_clip = CLIPConfig::new(49408, 1024, 1024, 16, 77, 24, false).init();

        let clip_tokenizer = SimpleTokenizer::new().unwrap();
        let open_clip_tokenizer = OpenClipTokenizer::new().unwrap();

        Embedder {
            clip, 
            open_clip, 
            clip_tokenizer, 
            open_clip_tokenizer, 
        }
    }
}

#[derive(Module, Debug)]
pub struct Embedder<B: Backend> {
    clip: CLIP<B>, 
    open_clip: CLIP<B>, 
    clip_tokenizer: SimpleTokenizer, 
    open_clip_tokenizer: OpenClipTokenizer, 
}

impl<B: Backend> Embedder<B> {
    pub fn text_to_conditioning(&self, text: &str, size: Tensor<B, 2, Int>, crop: Tensor<B, 2, Int>, ar: Tensor<B, 2, Int>) -> Conditioning<B> {
        let (unconditional_context, unconditional_channel_context) = self.unconditional_context(size.clone(), crop.clone(), ar.clone());
        let (context, channel_context) = self.context(text, size, crop, ar);

        Conditioning {
            unconditional_context, 
            context, 
            unconditional_channel_context, 
            channel_context, 
        }
    }

    fn unconditional_context(&self, size: Tensor<B, 2, Int>, crop: Tensor<B, 2, Int>, ar: Tensor<B, 2, Int>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let clip_context = text_to_context_clip("", &self.clip, &self.clip_tokenizer);
        let (open_clip_context, pooled_text_embed) = text_to_context_open_clip("", &self.open_clip, &self.open_clip_tokenizer);

        (
            Tensor::cat(vec![clip_context, open_clip_context], 2).squeeze(0), 
            conditioning_embedding(pooled_text_embed, 256, size, crop, ar).squeeze(0), 
        )
    }

    fn context(&self, text: &str, size: Tensor<B, 2, Int>, crop: Tensor<B, 2, Int>, ar: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let clip_context = text_to_context_clip(text, &self.clip, &self.clip_tokenizer);
        let (open_clip_context, pooled_text_embed) = text_to_context_open_clip(text, &self.open_clip, &self.open_clip_tokenizer);

        (
            Tensor::cat(vec![clip_context, open_clip_context], 2), 
            conditioning_embedding(pooled_text_embed, 256, size, crop, ar), 
        )
    }
}


pub fn text_to_context_clip<B: Backend, T: Tokenizer>(text: &str, clip: &CLIP<B>, tokenizer: &T) -> Tensor<B, 3> {
    let device = &clip.devices()[0];

    let tokens = tokenize_text(text, tokenizer, clip.max_sequence_length(), device);

    let n_layers = clip.num_layers();
    clip.forward_hidden(tokens, n_layers - 1) // penultimate layer
}

pub fn text_to_context_open_clip<B: Backend, T: Tokenizer>(text: &str, clip: &CLIP<B>, tokenizer: &T) -> (Tensor<B, 3>, Tensor<B, 2>) {
    let device = &clip.devices()[0];

    let tokens = tokenize_text(text, tokenizer, clip.max_sequence_length(), device);

    let n_layers = clip.num_layers();
    clip.forward_hidden_pooled(tokens, n_layers - 1) // penultimate layer
}

pub fn tokenize_text<B: Backend, T: Tokenizer>(text: &str, tokenizer: &T, seq_len: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    let mut tokenized: Vec<_> = tokenizer
        .encode(text, true, true)
        .into_iter()
        .map(|v| v as i32)
        .collect();

    tokenized.resize(seq_len, tokenizer.padding_token() as i32);

    Tensor::from_ints(&tokenized[..]).to_device(device).unsqueeze()
}





















use crate::helper::to_float;
use std::f64::consts::PI;

fn cosine_schedule<B: Backend>(n_steps: usize) -> Tensor<B, 1> {
    to_float(Tensor::arange(1..n_steps + 1))
        .mul_scalar(PI * 0.5 / n_steps as f64)
        .cos()
}

fn offset_cosine_schedule<B: Backend>(n_steps: usize, device: &B::Device) -> Tensor<B, 1> {
    let min_signal_rate: f64 = 0.02;
    let max_signal_rate: f64 = 0.95;
    let start_angle = max_signal_rate.acos();
    let end_angle = min_signal_rate.acos();

    let times = Tensor::arange_device(1..n_steps + 1, device);

    let diffusion_angles = to_float(times) * ( (end_angle - start_angle) / n_steps as f64) + start_angle;
    diffusion_angles.cos()
}

pub fn offset_cosine_schedule_cumprod<B: Backend>(n_steps: usize, device: &B::Device) -> Tensor<B, 1> {
    offset_cosine_schedule::<B>(n_steps, device).powf(2.0)
}