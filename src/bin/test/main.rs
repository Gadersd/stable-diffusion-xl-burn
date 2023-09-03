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

/*fn test_tiny_clip<B: Backend>(device: &B::Device) {
    println!("Loading Tiny Clip");
    let encoder: CLIP<B> = load_clip_text_transformer("params", device, false).unwrap();

    let tokenized: Vec<_> = vec![3, 1];
    println!("Tokens = {:?}", tokenized);

    let tokens = Tensor::from_ints(&tokenized[..]).unsqueeze();
    let output = encoder.forward(tokens);
    println!("Output: {:?}", output.into_data());
}*/

/*fn test_tiny_open_clip<B: Backend>(device: &B::Device) {
    println!("Loading Tiny Open Clip");
    let encoder: CLIP<B> = load_clip_text_transformer("params", device, true).unwrap();

    let tokenized: Vec<_> = vec![3, 1];
    println!("Tokens = {:?}", tokenized);

    let tokens = Tensor::from_ints(&tokenized[..]).unsqueeze();
    let output = encoder.forward(tokens);
    println!("Output: {:?}", output.into_data());
}*/

fn test_clip<B: Backend>(device: &B::Device) {
    println!("Loading Clip");
    let encoder: CLIP<B> = load_clip_text_transformer("params", device, false).unwrap();

    let tokenizer = ClipTokenizer::new().unwrap();

    let text = "Hello world! asdf!!!!asdf";
    println!("Sampling with text: {}", text);

    let mut tokenized: Vec<_> = tokenizer
        .encode(text, true, true)
        .into_iter()
        .map(|v| v as i32)
        .collect();
    tokenized.resize(77, tokenizer.padding_token() as i32);
    println!("Tokens = {:?}", tokenized);

    let tokens = Tensor::from_ints(&tokenized[..]).unsqueeze();
    let output = encoder.forward_hidden(tokens, 11);
    println!("Output: {:?}", output.into_data());
}

fn test_open_clip<B: Backend>(device: &B::Device) {
    println!("Loading Open Clip");
    let encoder: CLIP<B> = load_clip_text_transformer("params", device, true).unwrap();

    let tokenizer = OpenClipTokenizer::new().unwrap();

    let text = "Hello world! asdf!!!!asdf";
    println!("Sampling with text: {}", text);

    let mut tokenized: Vec<_> = tokenizer
        .encode(text, true, true)
        .into_iter()
        .map(|v| v as i32)
        .collect();
    tokenized.resize(77, tokenizer.padding_token() as i32);
    println!("Tokens = {:?}", tokenized);

    let tokens = Tensor::from_ints(&tokenized[..]).unsqueeze();
    let n_layers = encoder.num_layers();
    let (output, pooled) = encoder.forward_hidden_pooled(tokens, n_layers - 1); // penultimate layer
    println!("Output: {:?}\n\n", output.into_data());
    println!("Pooled: {:?}\n\n", pooled.into_data());
}

fn test_tiny_unet<B: Backend>(device: &B::Device) {
    println!("Loading unet");
    let unet: UNet<B> = load_unet("params", device).unwrap();

    println!("Sampling...");
    let x = arb_tensor([1, 4, 4, 4]); //Tensor::zeros([1, 4, 4, 4]);
    let context = arb_tensor([1, 1, 20]); //Tensor::zeros([1, 1, 20]);
    let y = arb_tensor([1, 8]); //Tensor::zeros([1, 8]);
    let t = Tensor::from_ints([1]).unsqueeze();
    let output = unet.forward(x, t, context, y);

    println!("Output: {:?}", output.into_data());
}

fn test_tiny_encoder<B: Backend>(device: &B::Device) {
    println!("Loading Encoder");
    let encoder: Encoder<B> = load_encoder("params", device).unwrap();

    println!("Sampling...");
    let x = arb_tensor([1, 3, 16, 16]);
    let output = encoder.forward(x);

    println!("Output: {:?}", output.into_data());
}

fn test_tiny_decoder<B: Backend>(device: &B::Device) {
    println!("Loading Decoder");
    let decoder: Decoder<B> = load_decoder("params", device).unwrap();

    println!("Sampling...");
    let x = arb_tensor([1, 4, 4, 4]);
    let output = decoder.forward(x);

    println!("Output: {:?}", output.into_data());
}

use burn::tensor::ElementConversion;
use num_traits::cast::ToPrimitive;
use stablediffusion::model::stablediffusion::Conditioning;

use stablediffusion::backend_converter::*;

fn main() {
    //type Backend = NdArrayBackend<f32>;
    //let device = NdArrayDevice::Cpu;

    type Backend = TchBackend<f32>;
    type Backend_f16 = TchBackend<tensor::f16>;

    let cpu_device = TchDevice::Cpu;
    let device = /*TchDevice::Cpu;*/ TchDevice::Cuda(0);

    //test_clip::<Backend>(&device);
    //test_tiny_open_clip::<Backend>(&device);
    //test_open_clip::<Backend>(&device);

    let text = "A beautiful photo of a seaside bluff.";

    let conditioning = {
        println!("Loading embedder...");
        let embedder: Embedder<Backend> = load_embedder_model("embedder").unwrap();
        let embedder = embedder.to_device(&device);

        let resolution = RESOLUTIONS[8];

        let size = Tensor::from_ints(resolution).to_device(&device).unsqueeze();
        let crop = Tensor::from_ints([0, 0]).to_device(&device).unsqueeze();
        let ar = Tensor::from_ints(resolution).to_device(&device).unsqueeze();

        println!("Running embedder...");
        embedder.text_to_conditioning(text, size, crop, ar)
    };

    let conditioning: Conditioning<Backend_f16> =
        conditioning.convert(DefaultBackendConverter::new(), &device);

    let latent = {
        println!("Loading diffuser...");
        let diffuser: Diffuser<Backend_f16> = load_diffuser_model("diffuser").unwrap();
        let diffuser = diffuser.to_device(&device);

        let unconditional_guidance_scale = 7.5;
        let n_steps = 30;

        println!("Running diffuser...");
        diffuser.sample_latent(conditioning, unconditional_guidance_scale, n_steps)
    };

    let latent: Tensor<Backend, 4> = DefaultBackendConverter::new().convert(latent, &device);

    let images = {
        println!("Loading latent decoder...");
        let latent_decoder: LatentDecoder<Backend> =
            load_latent_decoder_model("latent_decoder").unwrap();
        let latent_decoder = latent_decoder.to_device(&device);

        println!("Running decoder...");
        latent_decoder.latent_to_image(latent)
    };

    println!("Saving images...");
    save_images(
        &images.buffer,
        "img",
        images.width as u32,
        images.height as u32,
    )
    .unwrap();
    println!("Done.");

    return;

    /*let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <dump_path> <model_name>", args[0]);
        process::exit(1);
    }

    let dump_path = &args[1];
    let model_name = &args[2];

    if let Err(e) = convert_dump_to_model::<Backend>(dump_path, model_name, &device) {
        eprintln!("Failed to convert dump to model: {:?}", e);
        process::exit(1);
    }

    println!("Successfully converted {} to {}", dump_path, model_name);*/
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
