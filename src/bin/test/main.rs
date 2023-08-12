use std::env;
use std::process;
use std::error::Error;

use stablediffusion::model::unet::{UNet, UNetConfig, load::load_unet};
use stablediffusion::model::autoencoder::{Decoder, DecoderConfig, load::load_decoder};
use stablediffusion::model::autoencoder::{Encoder, EncoderConfig, load::load_encoder};
use stablediffusion::model::clip::{CLIP, CLIPConfig, load::load_clip_text_transformer};
use stablediffusion::model::stablediffusion::{Embedder, Diffuser, LatentDecoder, load::*};

use burn::{
    config::Config, 
    module::{Module, Param},
    nn,
    tensor::{
        self, 
        backend::Backend,
        Tensor,
    },
};

//use burn_ndarray::{NdArrayBackend, NdArrayDevice};
use burn_tch::{TchBackend, TchDevice};

use burn::record::{self, Recorder, NamedMpkGzFileRecorder, HalfPrecisionSettings};

fn load_embedder_model<B: Backend>(model_name: &str) -> Result<Embedder<B>, record::RecorderError> {
    NamedMpkGzFileRecorder::<HalfPrecisionSettings>::new()
    .load(model_name.into())
    .map(|record| EmbedderConfig::new().init().load_record(record))
}

fn load_diffuser_model<B: Backend>(model_name: &str) -> Result<Diffuser<B>, record::RecorderError> {
    NamedMpkGzFileRecorder::<HalfPrecisionSettings>::new()
    .load(model_name.into())
    .map(|record| DiffuserConfig::new().init().load_record(record))
}

fn load_latent_decoder_model<B: Backend>(model_name: &str) -> Result<LatentDecoder<B>, record::RecorderError> {
    NamedMpkGzFileRecorder::<HalfPrecisionSettings>::new()
    .load(model_name.into())
    .map(|record| LatentDecoderConfig::new().init().load_record(record))
}

use stablediffusion::helper::to_float;

fn arb_tensor<B: Backend, const D: usize>(dims: [usize; D]) -> Tensor<B, D> {
    let prod = dims.iter().cloned().product();
    to_float(Tensor::arange(0..prod)).sin().reshape(dims)
}

use stablediffusion::token::{Tokenizer, clip::SimpleTokenizer, open_clip::OpenClipTokenizer};

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

    let tokenizer = SimpleTokenizer::new().unwrap();

    let text = "Hello world! asdf!!!!asdf";
    println!("Sampling with text: {}", text);

    let mut tokenized: Vec<_> = tokenizer.encode(text, true, true).into_iter().map(|v| v as i32).collect();
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

    let mut tokenized: Vec<_> = tokenizer.encode(text, true, true).into_iter().map(|v| v as i32).collect();
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

fn main() {
    //type Backend = NdArrayBackend<f32>;
    //let device = NdArrayDevice::Cpu;

    type Backend = TchBackend<f32>;
    let cpu_device = TchDevice::Cpu;
    let device = TchDevice::Cpu; //TchDevice::Cuda(0);

    //test_clip::<Backend>(&device);
    //test_tiny_open_clip::<Backend>(&device);
    //test_open_clip::<Backend>(&device);

    let text = "A monkey slapping its ass.";

    let conditioning = {
        let embedder: Embedder<Backend> = load_embedder_model("embedder").unwrap();

        let size = Tensor::from_ints([1024, 1024]).to_device(&device).unsqueeze();
        let crop = Tensor::from_ints([0, 0]).to_device(&device).unsqueeze();
        let ar = Tensor::from_ints([1024, 1024]).to_device(&device).unsqueeze();

        embedder.text_to_conditioning(text, size, crop, ar)
    };

    println!("Conditioning: {:?}", conditioning.context.clone().slice([0..1, 0..1, 0..100]).into_data());
    println!("Conditioning: {:?}", conditioning.unconditional_context.clone().slice([0..1, 0..100]).into_data());
    println!("Conditioning: {:?}", conditioning.channel_context.clone().slice([0..1, 0..100]).into_data());
    println!("Conditioning: {:?}", conditioning.unconditional_channel_context.clone().slice([0..100]).into_data());

    let latent = {
        let diffuser = load_diffuser_model("diffuser").unwrap();

        let unconditional_guidance_scale = 7.5;
        let n_steps = 1;

        diffuser.sample_latent(conditioning, unconditional_guidance_scale, n_steps)
    };

    println!("Latent: {:?}", latent.clone().slice([0..1, 0..1, 0..100, 0..100]).into_data());

    /*let images = {
        let latent_decoder = load_latent_model("latent_decoder").unwrap();

        latent_decoder.latent_to_image(latent)
    };

    save_images(&images, "img", 1024, 1024).unwrap();*/

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


use image::{self, ImageResult, ColorType::Rgb8};

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
    let raw: Vec<_> = (0..width * height).into_iter().flat_map(|i| {
        let row = i / width;
        let red = (255.0 * row as f64 / height as f64) as u8;

        [red, 0, 0]
    }).collect();

    image::save_buffer("red.png", &raw[..], width, height, Rgb8)
}