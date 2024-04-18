pub mod load;

use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, BasicOps, Data, Distribution, Float, Int, Tensor},
};
use burn::prelude::*;
use burn::tensor::ElementConversion;

use num_traits::ToPrimitive;

use super::autoencoder::{Autoencoder, AutoencoderConfig};
use super::clip::{CLIPConfig, CLIP};
use super::unet::{conditioning_embedding, UNet, UNetConfig};
use crate::backend::Backend as MyBackend;
use crate::token::{clip::ClipTokenizer, open_clip::OpenClipTokenizer, Tokenizer};

/*#[derive(Config)]
pub struct StableDiffusionConfig {

}

impl StableDiffusionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StableDiffusion<B> {
        let n_steps = 1000;
        let alpha_cumulative_products = offset_cosine_schedule_cumprod::<B>(n_steps).into();

        let autoencoder = AutoencoderConfig::new().init(device);
        let diffusion = UNetConfig::new(2816, 4, 4, 320, 64, 2048).init(device);
        let clip = CLIPConfig::new(49408, 768, 12, 77, 12, false).init(device);

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

    pub fn unconditional_context(&self, tokenizer: &ClipTokenizer) -> Tensor<B, 2> {
        self.context(tokenizer, "").squeeze(0)
    }

    pub fn context(&self, tokenizer: &ClipTokenizer, text: &str) -> Tensor<B, 3> {
        let device = &self.clip.devices()[0];
        let text = format!("<|startoftext|>{}<|endoftext|>", text);
        let tokenized: Vec<_> = tokenizer.encode(&text).into_iter().map(|v| v as i32).collect();

        self.clip.forward_hidden(Tensor::from_ints(&tokenized[..]).to_device(device).unsqueeze(), 11)
    }
}*/

pub struct RawImages {
    pub buffer: Vec<Vec<u8>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Config, Debug)]
pub struct LatentDecoderConfig {
    scale_factor: f64,
}

impl LatentDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LatentDecoder<B> {
        let autoencoder = AutoencoderConfig::new().init(device);
        let scale_factor = self.scale_factor;

        LatentDecoder {
            autoencoder,
            scale_factor,
        }
    }
}

#[derive(Module, Debug)]
pub struct LatentDecoder<B: Backend> {
    autoencoder: Autoencoder<B>,
    scale_factor: f64,
}

impl<B: MyBackend> LatentDecoder<B> {
    pub fn latent_to_image(&self, latent: Tensor<B, 4>) -> RawImages {
        let [n_batch, _, latent_height, latent_width] = latent.dims();
        let image = self.decode_latent(latent);

        let n_channel = 3;
        let height = latent_height * 8;
        let width = latent_width * 8;
        let num_elements_per_image = n_channel * height * width;

        // correct size and scale and reorder to
        let image = (image + 1.0) / 2.0;
        let image = image
            .reshape([n_batch, n_channel, height, width])
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .mul_scalar(255.0);

        let flattened: Vec<_> = image.into_data().value;

        let buffer = (0..n_batch)
            .into_iter()
            .map(|b| {
                let start = b * num_elements_per_image;
                let end = start + num_elements_per_image;

                flattened[start..end]
                    .into_iter()
                    .map(|v| v.to_f64().unwrap().min(255.0).max(0.0).to_u8().unwrap())
                    .collect()
            })
            .collect();

        RawImages {
            buffer: buffer,
            width: width,
            height: height,
        }
    }

    pub fn image_to_latent(&self, images: &RawImages, device: &B::Device) -> Tensor<B, 4> {
        let n_images = images.buffer.len();
        let n_channel = 3;

        let data = images.buffer.iter().map(|v| v.iter().map(|v| v.elem())).flatten().collect();
        let shape = [n_images, images.height, images.width, n_channel];

        // transform elements to between -1 and 1 and change dims to [n_batch, n_channel, height, width]
        let pre_latent = Tensor::from_data(Data::new(data, shape.into()), device)
            .div_scalar(255.0)
            .swap_dims(2, 3)
            .swap_dims(1, 2)
            .mul_scalar(2.0)
            .sub_scalar(1.0);

        self.encode_image(pre_latent)
    }

    pub fn encode_image(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.autoencoder
            .encode_image(x)
            .mul_scalar(self.scale_factor)
    }

    pub fn decode_latent(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.autoencoder
            .decode_latent(x * (1.0 / self.scale_factor) /* * (1.0 / 0.13025)*/)
    }
}

#[derive(Config, Debug)]
pub struct DiffuserConfig {
    adm_in_channels: usize,
    model_channels: usize,
    channel_mults: Vec<usize>,
    num_head_channels: usize,
    transformer_depths: Vec<usize>,
    context_dim: usize,
    is_refiner: bool,
}

impl DiffuserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Diffuser<B> {
        let n_steps = 1000;
        let alpha_cumulative_products = Param::from_tensor(Tensor::zeros([1], device)); //offset_cosine_schedule_cumprod::<B>(n_steps).into();
                                                                   //let diffusion = UNetConfig::new(2816, 4, 4, 320, 64, 2048).init(device);
        let diffusion = UNetConfig::new(
            self.adm_in_channels,
            4,
            4,
            self.model_channels,
            self.channel_mults.clone(),
            self.num_head_channels,
            self.transformer_depths.clone(),
            self.context_dim,
        )
        .init(device);

        let is_refiner = self.is_refiner;

        Diffuser {
            n_steps,
            alpha_cumulative_products,
            diffusion,
            is_refiner,
        }
    }
}

#[derive(Module, Debug)]
pub struct Diffuser<B: Backend> {
    n_steps: usize,
    pub alpha_cumulative_products: Param<Tensor<B, 1>>,
    pub diffusion: UNet<B>,
    is_refiner: bool,
}

impl<B: MyBackend> Diffuser<B> {
    pub fn sample_latent(
        &self,
        conditioning: Conditioning<B>,
        unconditional_guidance_scale: f64,
        n_steps: usize,
    ) -> Tensor<B, 4> {
        let latent = Self::gen_noise(&conditioning);

        self.diffuse_latent(
            latent,
            conditioning,
            0,
            n_steps,
            unconditional_guidance_scale,
        )
    }

    pub fn sample_latent_with_inpainting(
        &self,
        conditioning: Conditioning<B>,
        unconditional_guidance_scale: f64,
        n_steps: usize,
        reference: Tensor<B, 4>, 
        mask: Tensor<B, 4, Bool>, 
    ) -> Tensor<B, 4> {
        let latent = Self::gen_noise(&conditioning);

        self.diffuse_latent_with_inpainting(
            latent,
            conditioning,
            0,
            n_steps,
            unconditional_guidance_scale,
            reference, 
            mask, 
        )
    }

    pub fn refine_latent(
        &self,
        latent: Tensor<B, 4>,
        conditioning: Conditioning<B>,
        unconditional_guidance_scale: f64,
        step_start: usize,
        n_steps: usize,
    ) -> Tensor<B, 4> {
        let t = self.n_steps - step_start; // last step is beginning of diffusion
        let start_alpha: f64 = self.get_alpha(t);

        let noised_latent = latent * start_alpha.sqrt()
            + Self::gen_noise(&conditioning) * (1.0 - start_alpha).sqrt();

        self.diffuse_latent(
            noised_latent,
            conditioning,
            step_start,
            n_steps,
            unconditional_guidance_scale,
        )
    }

    fn gen_noise(conditioning: &Conditioning<B>) -> Tensor<B, 4> {
        let device = conditioning.context_full.device();
        let [n_batches, _, _] = conditioning.context_full.dims();
        let [height, width] = conditioning.resolution;

        Tensor::random(
            [n_batches, 4, height / 8, width / 8],
            Distribution::Normal(0.0, 1.0),
            &device
        )
    }

    fn diffuse_latent(
        &self,
        mut latent: Tensor<B, 4>,
        conditioning: Conditioning<B>,
        step_start: usize,
        n_steps: usize,
        unconditional_guidance_scale: f64,
    ) -> Tensor<B, 4> {
        let device = latent.device();

        let step_size = self.n_steps / n_steps;

        let sigma = 0.0; // Use deterministic diffusion

        let step_start = self.n_steps - step_start;

        for t in (0..step_start).rev().step_by(step_size) {
            let current_alpha: f64 = self.get_alpha(t);
            let prev_alpha: f64 = if t >= step_size {
                self.get_alpha(t - step_size)
            } else {
                1.0
            };

            let sqrt_noise = (1.0 - current_alpha).sqrt();

            let timestep = Tensor::from_ints([t as i32], &device);
            let pred_noise = self.forward_diffuser(
                latent.clone(),
                timestep,
                conditioning.clone(),
                unconditional_guidance_scale,
            );
            let predx0 = (latent - pred_noise.clone() * sqrt_noise) / current_alpha.sqrt();
            let dir_latent = pred_noise * (1.0 - prev_alpha - sigma * sigma).sqrt();

            let prev_latent =
                predx0 * prev_alpha.sqrt() + dir_latent + Self::gen_noise(&conditioning) * sigma;
            latent = prev_latent;
        }

        latent
    }

    fn diffuse_latent_with_inpainting(
        &self,
        mut latent: Tensor<B, 4>,
        conditioning: Conditioning<B>,
        step_start: usize,
        n_steps: usize,
        unconditional_guidance_scale: f64,
        reference: Tensor<B, 4>, 
        mask: Tensor<B, 4, Bool>, 
    ) -> Tensor<B, 4> {
        let device = latent.device();

        let step_size = self.n_steps / n_steps;

        let sigma = 0.0; // Use deterministic diffusion

        let step_start = self.n_steps - step_start;

        for t in (0..step_start).rev().step_by(step_size) {
            let current_alpha: f64 = self.get_alpha(t);
            let prev_alpha: f64 = if t >= step_size {
                self.get_alpha(t - step_size)
            } else {
                1.0
            };

            let sqrt_noise = (1.0 - current_alpha).sqrt();

            // combine with reference for inpainting
            let noised_reference = reference.clone() * current_alpha.sqrt()
                + Self::gen_noise(&conditioning) * sqrt_noise;
            latent = noised_reference.mask_where(mask.clone(), latent);

            let timestep = Tensor::from_ints([t as i32], &device);
            let pred_noise = self.forward_diffuser(
                latent.clone(),
                timestep,
                conditioning.clone(),
                unconditional_guidance_scale,
            );
            let predx0 = (latent - pred_noise.clone() * sqrt_noise) / current_alpha.sqrt();
            let dir_latent = pred_noise * (1.0 - prev_alpha - sigma * sigma).sqrt();

            let prev_latent =
                predx0 * prev_alpha.sqrt() + dir_latent + Self::gen_noise(&conditioning) * sigma;
            latent = prev_latent;
        }

        latent
    }

    fn get_alpha(&self, i: usize) -> f64 {
        self.alpha_cumulative_products
            .val()
            .slice([i..i + 1])
            .into_scalar()
            .to_f64()
            .unwrap()
    }

    fn forward_diffuser(
        &self,
        latent: Tensor<B, 4>,
        timestep: Tensor<B, 1, Int>,
        conditioning: Conditioning<B>,
        unconditional_guidance_scale: f64,
    ) -> Tensor<B, 4> {
        let [n_batch, _, _, _] = latent.dims();
        let full_context_dim = conditioning.unconditional_context_full.dims()[0];
        let open_clip_context_dim = conditioning.unconditional_context_open_clip.dims()[0];

        // grab the right contexts depending if refiner or not
        let (unconditional_context, context, unconditional_channel_context, channel_context) =
            if !self.is_refiner {
                (
                    conditioning.unconditional_context_full,
                    conditioning.context_full,
                    conditioning.unconditional_channel_context,
                    conditioning.channel_context,
                )
            } else {
                (
                    conditioning.unconditional_context_open_clip,
                    conditioning.context_open_clip,
                    conditioning.unconditional_channel_context_refiner,
                    conditioning.channel_context_refiner,
                )
            };

        let conditional_latent =
            self.diffusion
                .forward(latent.clone(), timestep.clone(), context, channel_context);

        // don't use guidance scaling for refiner
        if self.is_refiner {
            return conditional_latent;
        }

        let unconditional_latent = self.diffusion.forward(
            latent,
            timestep,
            unconditional_context.unsqueeze().repeat(0, n_batch),
            unconditional_channel_context.unsqueeze().repeat(0, n_batch),
        );

        unconditional_latent.clone()
            + (conditional_latent - unconditional_latent) * unconditional_guidance_scale
    }
}

#[derive(Clone, Debug)]
pub struct Conditioning<B: Backend> {
    pub unconditional_context_full: Tensor<B, 2>,
    pub unconditional_context_open_clip: Tensor<B, 2>,
    pub context_full: Tensor<B, 3>,
    pub context_open_clip: Tensor<B, 3>,
    pub unconditional_channel_context: Tensor<B, 1>,
    pub unconditional_channel_context_refiner: Tensor<B, 1>,
    pub channel_context: Tensor<B, 2>,
    pub channel_context_refiner: Tensor<B, 2>,
    pub resolution: [usize; 2], // (height, width)
}

use crate::backend_converter::BackendConverter;

impl<B: Backend> Conditioning<B> {
    pub fn convert<B2: Backend, BC: BackendConverter<B2>>(
        self,
        converter: BC,
        device: &B2::Device,
    ) -> Conditioning<B2> {
        Conditioning {
            unconditional_context_full: converter.convert(self.unconditional_context_full, device),
            unconditional_context_open_clip: converter
                .convert(self.unconditional_context_open_clip, device),
            context_full: converter.convert(self.context_full, device),
            context_open_clip: converter.convert(self.context_open_clip, device),
            unconditional_channel_context: converter
                .convert(self.unconditional_channel_context, device),
            unconditional_channel_context_refiner: converter
                .convert(self.unconditional_channel_context_refiner, device),
            channel_context: converter.convert(self.channel_context, device),
            channel_context_refiner: converter.convert(self.channel_context_refiner, device),
            resolution: self.resolution,
        }
    }
}

/// These are the resolutions (height, width) Stable Diffusion XL was trained on.
pub const RESOLUTIONS: [[i32; 2]; 40] = [
    [512, 2048],
    [512, 1984],
    [512, 1920],
    [512, 1856],
    [576, 1792],
    [576, 1728],
    [576, 1664],
    [640, 1600],
    [640, 1536],
    [704, 1472],
    [704, 1408],
    [704, 1344],
    [768, 1344],
    [768, 1280],
    [832, 1216],
    [832, 1152],
    [896, 1152],
    [896, 1088],
    [960, 1088],
    [960, 1024],
    [1024, 1024],
    [1024, 960],
    [1088, 960],
    [1088, 896],
    [1152, 896],
    [1152, 832],
    [1216, 832],
    [1280, 768],
    [1344, 768],
    [1408, 704],
    [1472, 704],
    [1536, 640],
    [1600, 640],
    [1664, 576],
    [1728, 576],
    [1792, 576],
    [1856, 512],
    [1920, 512],
    [1984, 512],
    [2048, 512],
];

#[derive(Config, Debug)]
pub struct EmbedderConfig {
    clip_config: CLIPConfig,
    open_clip_config: CLIPConfig,
}

impl EmbedderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Embedder<B> {
        /*let clip = CLIPConfig::new(49408, 768, 768, 12, 77, 12, true).init(device);
        let open_clip = CLIPConfig::new(49408, 1024, 1024, 16, 77, 24, false).init(device);*/

        let clip = self.clip_config.init(device);
        let open_clip = self.open_clip_config.init(device);

        let clip_tokenizer = ClipTokenizer::new().unwrap();
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
    clip_tokenizer: ClipTokenizer,
    open_clip_tokenizer: OpenClipTokenizer,
}

impl<B: MyBackend> Embedder<B> {
    pub fn text_to_conditioning(
        &self,
        text: &str,
        size: Tensor<B, 2, Int>,
        crop: Tensor<B, 2, Int>,
        ar: Tensor<B, 1, Int>,
    ) -> Conditioning<B> {
        let [n_batch, _] = size.dims();
        let ar_data = ar.clone().into_data();
        let resolution = [
            ar_data.value[0].to_usize().unwrap(),
            ar_data.value[1].to_usize().unwrap(),
        ];
        let batched_ar = ar.unsqueeze().repeat(0, n_batch);

        let (
            unconditional_context_full,
            unconditional_context_open_clip,
            unconditional_channel_context,
            unconditional_channel_context_refiner,
        ) = self.unconditional_context(size.clone(), crop.clone(), batched_ar.clone());
        let (context_full, context_open_clip, channel_context, channel_context_refiner) =
            self.context(text, size, crop, batched_ar);

        Conditioning {
            unconditional_context_full,
            unconditional_context_open_clip,
            context_full,
            context_open_clip,
            unconditional_channel_context,
            unconditional_channel_context_refiner,
            channel_context,
            channel_context_refiner,
            resolution,
        }
    }

    fn unconditional_context(
        &self,
        size: Tensor<B, 2, Int>,
        crop: Tensor<B, 2, Int>,
        ar: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>, Tensor<B, 1>) {
        let clip_context = text_to_context_clip("", &self.clip, &self.clip_tokenizer);
        let (open_clip_context, pooled_text_embed) =
            text_to_context_open_clip("", &self.open_clip, &self.open_clip_tokenizer);

        let [n_batch, _] = ar.dims();
        let aesthetic_scores = Tensor::from_ints([6], &size.device())
            .repeat(0, n_batch)
            .unsqueeze();

        (
            Tensor::cat(vec![clip_context, open_clip_context.clone()], 2).squeeze(0),
            open_clip_context.squeeze(0),
            conditioning_embedding(
                pooled_text_embed.clone(),
                256,
                size.clone(),
                crop.clone(),
                ar,
            )
            .squeeze(0),
            conditioning_embedding(pooled_text_embed, 256, size, crop, aesthetic_scores).squeeze(0),
        )
    }

    fn context(
        &self,
        text: &str,
        size: Tensor<B, 2, Int>,
        crop: Tensor<B, 2, Int>,
        ar: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
        let clip_context = text_to_context_clip(text, &self.clip, &self.clip_tokenizer);
        let (open_clip_context, pooled_text_embed) =
            text_to_context_open_clip(text, &self.open_clip, &self.open_clip_tokenizer);

        let [n_batch, _] = ar.dims();
        let aesthetic_scores = Tensor::from_ints([6], &size.device())
            .repeat(0, n_batch)
            .unsqueeze();

        (
            Tensor::cat(vec![clip_context, open_clip_context.clone()], 2),
            open_clip_context,
            conditioning_embedding(
                pooled_text_embed.clone(),
                256,
                size.clone(),
                crop.clone(),
                ar,
            ),
            conditioning_embedding(pooled_text_embed, 256, size, crop, aesthetic_scores),
        )
    }
}

pub fn text_to_context_clip<B: MyBackend, T: Tokenizer>(
    text: &str,
    clip: &CLIP<B>,
    tokenizer: &T,
) -> Tensor<B, 3> {
    let device = &clip.devices()[0];

    let tokens = tokenize_text(text, tokenizer, clip.max_sequence_length(), device);

    let n_layers = clip.num_layers();
    clip.forward_hidden(tokens, n_layers - 1) // penultimate layer
}

pub fn text_to_context_open_clip<B: MyBackend, T: Tokenizer>(
    text: &str,
    clip: &CLIP<B>,
    tokenizer: &T,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    let device = &clip.devices()[0];

    let tokens = tokenize_text(text, tokenizer, clip.max_sequence_length(), device);

    let n_layers = clip.num_layers();
    clip.forward_hidden_pooled(tokens, n_layers - 1) // penultimate layer
}

pub fn tokenize_text<B: Backend, T: Tokenizer>(
    text: &str,
    tokenizer: &T,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let mut tokenized: Vec<_> = tokenizer
        .encode(text, true, true)
        .into_iter()
        .map(|v| v as i32)
        .collect();

    tokenized.resize(seq_len, tokenizer.padding_token() as i32);

    Tensor::from_ints(&tokenized[..], device)
        .unsqueeze()
}

use std::f64::consts::PI;

fn cosine_schedule<B: Backend>(n_steps: usize, device: &B::Device) -> Tensor<B, 1> {
    Tensor::arange(1..n_steps as i64 + 1, device)
        .float()
        .mul_scalar(PI * 0.5 / n_steps as f64)
        .cos()
}

fn offset_cosine_schedule<B: Backend>(n_steps: usize, device: &B::Device) -> Tensor<B, 1> {
    let min_signal_rate: f64 = 0.02;
    let max_signal_rate: f64 = 0.95;
    let start_angle = max_signal_rate.acos();
    let end_angle = min_signal_rate.acos();

    let times = Tensor::arange(1..n_steps as i64 + 1, device).float();

    let diffusion_angles = times * ((end_angle - start_angle) / n_steps as f64) + start_angle;
    diffusion_angles.cos()
}

pub fn offset_cosine_schedule_cumprod<B: Backend>(
    n_steps: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    offset_cosine_schedule::<B>(n_steps, device).powf_scalar(2.0)
}
