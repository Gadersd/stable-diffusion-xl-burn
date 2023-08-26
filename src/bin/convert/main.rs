use std::env;
use std::error::Error;
use std::process;

use stablediffusion::model::autoencoder::{load::load_decoder, Decoder, DecoderConfig};
use stablediffusion::model::autoencoder::{load::load_encoder, Encoder, EncoderConfig};
use stablediffusion::model::clip::{load::load_clip_text_transformer, CLIPConfig, CLIP};
use stablediffusion::model::stablediffusion::{load::*, Diffuser, Embedder, LatentDecoder};
use stablediffusion::model::unet::{load::load_unet, UNet, UNetConfig};

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{self, backend::Backend, Tensor},
};

use burn_tch::{TchBackend, TchDevice};

use burn::record::{self, BinFileRecorder, HalfPrecisionSettings, Recorder};

fn convert_embedder_dump_to_model<B: Backend>(
    dump_path: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading dump...");
    let model: Embedder<B> = load_embedder(dump_path, device)?;

    println!("Saving model...");
    save_model_file(model, model_name)?;

    Ok(())
}

fn convert_latent_decoder_dump_to_model<B: Backend>(
    dump_path: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading dump...");
    let model: LatentDecoder<B> = load_latent_decoder(dump_path, device)?;

    println!("Saving model...");
    save_model_file(model, model_name)?;

    Ok(())
}

fn convert_diffuser_dump_to_model<B: Backend>(
    dump_path: &str,
    model_name: &str,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    println!("Loading dump...");
    let model: Diffuser<B> = load_diffuser(dump_path, device)?;

    println!("Saving model...");
    save_model_file(model, model_name)?;

    Ok(())
}

fn save_model_file<B: Backend, M: Module<B>>(
    model: M,
    name: &str,
) -> Result<(), record::RecorderError> {
    BinFileRecorder::<HalfPrecisionSettings>::new().record(model.into_record(), name.into())
}

use std::env;

fn main() {
    let params = match env::args().nth(1) {
        Some(folder) => folder,
        None => {
            eprintln!("Error: no weight dump folder name provided.");
            std::process::exit(1);
        }
    };

    type Backend = TchBackend<f32>;
    let device = TchDevice::Cpu;

    println!("Saving embedder...");
    match convert_embedder_dump_to_model::<Backend>(&params, "embedder", &device) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error converting embedder: {}", e);
            std::process::exit(1);
        }
    }

    println!("Saving diffuser...");
    match convert_diffuser_dump_to_model::<Backend>(&params, "diffuser", &device) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error converting diffuser: {}", e);
            std::process::exit(1);
        }
    }

    println!("Saving latent decoder...");
    match convert_latent_decoder_dump_to_model::<Backend>(&params, "latent_decoder", &device) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error converting latent decoder: {}", e);
            std::process::exit(1);
        }
    }

    println!("Conversion completed.");
}
