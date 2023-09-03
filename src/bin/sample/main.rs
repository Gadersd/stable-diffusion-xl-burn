use std::env;
use std::error::Error;
use std::process;

use stablediffusion::model::autoencoder::{load::load_decoder, Decoder, DecoderConfig};
use stablediffusion::model::autoencoder::{load::load_encoder, Encoder, EncoderConfig};
use stablediffusion::model::clip::{load::load_clip_text_transformer, CLIPConfig, CLIP};
use stablediffusion::model::stablediffusion::{
    load::*, offset_cosine_schedule_cumprod, Diffuser, DiffuserConfig, Embedder, EmbedderConfig,
    LatentDecoder, LatentDecoderConfig, RESOLUTIONS,
};
use stablediffusion::model::unet::{load::load_unet, UNet, UNetConfig};

use stablediffusion::backend::Backend;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{self, Tensor},
};

use burn_tch::{TchBackend, TchDevice};

use burn::record::{self, BinFileRecorder, HalfPrecisionSettings, Recorder};

fn load_embedder_model<B: Backend>(model_name: &str) -> Result<Embedder<B>, Box<dyn Error>> {
    let config = EmbedderConfig::load(&format!("{}.cfg", model_name))?;
    let record = BinFileRecorder::<HalfPrecisionSettings>::new().load(model_name.into())?;

    Ok(config.init().load_record(record))
}

fn load_diffuser_model<B: Backend>(model_name: &str) -> Result<Diffuser<B>, Box<dyn Error>> {
    let config = DiffuserConfig::load(&format!("{}.cfg", model_name))?;
    let record = BinFileRecorder::<HalfPrecisionSettings>::new().load(model_name.into())?;

    Ok(config.init().load_record(record))
}

fn load_latent_decoder_model<B: Backend>(
    model_name: &str,
) -> Result<LatentDecoder<B>, Box<dyn Error>> {
    let config = LatentDecoderConfig::load(&format!("{}.cfg", model_name))?;
    let record = BinFileRecorder::<HalfPrecisionSettings>::new().load(model_name.into())?;

    Ok(config.init().load_record(record))
}

use stablediffusion::helper::to_float;

fn arb_tensor<B: Backend, const D: usize>(dims: [usize; D]) -> Tensor<B, D> {
    let prod = dims.iter().cloned().product();
    to_float(Tensor::arange(0..prod)).sin().reshape(dims)
}

use stablediffusion::token::{clip::ClipTokenizer, open_clip::OpenClipTokenizer, Tokenizer};

use burn::tensor::ElementConversion;
use num_traits::cast::ToPrimitive;
use stablediffusion::model::stablediffusion::Conditioning;

use stablediffusion::backend_converter::*;

fn main() {
    type Backend = TchBackend<f32>;
    type Backend_f16 = TchBackend<tensor::f16>;

    let device = TchDevice::Cuda(0);

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("Usage: {} <model_name> <refiner(y/n)> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image_name>", args[0]);
        process::exit(1);
    }

    let model_name = &args[1];
    let use_refiner = match args[2].as_str() {
        "y" => true,
        _ => false,
    };
    let unconditional_guidance_scale: f64 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid unconditional guidance scale.");
        process::exit(1);
    });
    let n_steps: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid number of diffusion steps.");
        process::exit(1);
    });
    let prompt = &args[5];
    let output_image_name = &args[6];

    let conditioning = {
        println!("Loading embedder...");
        let embedder: Embedder<Backend> =
            load_embedder_model(&format!("{}/embedder", model_name)).unwrap();
        let embedder = embedder.to_device(&device);

        let resolution = [1024, 1024]; //RESOLUTIONS[8];

        let size = Tensor::from_ints(resolution).to_device(&device).unsqueeze();
        let crop = Tensor::from_ints([0, 0]).to_device(&device).unsqueeze();
        let ar = Tensor::from_ints(resolution).to_device(&device).unsqueeze();

        println!("Running embedder...");
        embedder.text_to_conditioning(prompt, size, crop, ar)
    };

    let conditioning: Conditioning<Backend_f16> =
        conditioning.convert(DefaultBackendConverter::new(), &device);

    let latent = {
        println!("Loading diffuser...");
        let diffuser: Diffuser<Backend_f16> =
            load_diffuser_model(&format!("{}/diffuser", model_name)).unwrap();
        let diffuser = diffuser.to_device(&device);

        println!("Running diffuser...");
        diffuser.sample_latent(conditioning.clone(), unconditional_guidance_scale, n_steps)
    };

    let latent = if use_refiner {
        println!("Loading refiner...");
        let diffuser: Diffuser<Backend_f16> =
            load_diffuser_model(&format!("{}/refiner", model_name)).unwrap();
        let diffuser = diffuser.to_device(&device);

        println!("Running refiner...");
        diffuser.refine_latent(
            latent,
            conditioning,
            unconditional_guidance_scale,
            800,
            n_steps,
        )
    } else {
        latent
    };

    let latent: Tensor<Backend, 4> = DefaultBackendConverter::new().convert(latent, &device);

    let images = {
        println!("Loading latent decoder...");
        let latent_decoder: LatentDecoder<Backend> =
            load_latent_decoder_model(&format!("{}/latent_decoder", model_name)).unwrap();
        let latent_decoder = latent_decoder.to_device(&device);

        println!("Running decoder...");
        latent_decoder.latent_to_image(latent)
    };

    println!("Saving images...");
    save_images(
        &images.buffer,
        output_image_name,
        images.width as u32,
        images.height as u32,
    )
    .unwrap();
    println!("Done.");

    return;
}

use image::{self, ColorType::Rgb8, ImageResult};

fn save_images(images: &Vec<Vec<u8>>, basepath: &str, width: u32, height: u32) -> ImageResult<()> {
    for (index, img_data) in images.iter().enumerate() {
        let path = format!("{}{}.png", basepath, index);
        image::save_buffer(path, &img_data[..], width, height, Rgb8)?;
    }

    Ok(())
}

// save red test image
fn save_test_image() -> ImageResult<()> {
    let width = 256;
    let height = 256;
    let raw: Vec<_> = (0..width * height)
        .into_iter()
        .flat_map(|i| {
            let row = i / width;
            let red = (255.0 * row as f64 / height as f64) as u8;

            [red, 0, 0]
        })
        .collect();

    image::save_buffer("red.png", &raw[..], width, height, Rgb8)
}
