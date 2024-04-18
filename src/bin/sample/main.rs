use std::env;
use std::error::Error;
use std::process;

use stablediffusion::model::autoencoder::{load::load_decoder, Decoder, DecoderConfig};
use stablediffusion::model::autoencoder::{load::load_encoder, Encoder, EncoderConfig};
use stablediffusion::model::clip::{load::load_clip_text_transformer, CLIPConfig, CLIP};
use stablediffusion::model::stablediffusion::{
    load::*, offset_cosine_schedule_cumprod, Diffuser, DiffuserConfig, Embedder, EmbedderConfig,
    LatentDecoder, LatentDecoderConfig, RESOLUTIONS,
    RawImages, 
};
use stablediffusion::model::unet::{load::load_unet, UNet, UNetConfig};

use stablediffusion::backend::Backend;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{self, Tensor},
};

use burn_tch::{LibTorch, LibTorchDevice};

use burn::record::{self, NamedMpkFileRecorder, HalfPrecisionSettings, Recorder};

fn load_embedder_model<B: Backend>(model_dir: &str, device: &B::Device) -> Result<Embedder<B>, Box<dyn Error>> {
    let config = EmbedderConfig::load(&format!("{}.cfg", model_dir))?;
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;

    Ok(config.init(device).load_record(record))
}

fn load_diffuser_model<B: Backend>(model_dir: &str, device: &B::Device) -> Result<Diffuser<B>, Box<dyn Error>> {
    let config = DiffuserConfig::load(&format!("{}.cfg", model_dir))?;
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;
    //let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;

    Ok(config.init(device).load_record(record))
}

fn load_latent_decoder_model<B: Backend>(
    model_dir: &str,
    device: &B::Device
) -> Result<LatentDecoder<B>, Box<dyn Error>> {
    let config = LatentDecoderConfig::load(&format!("{}.cfg", model_dir))?;
    let record = NamedMpkFileRecorder::<HalfPrecisionSettings>::new().load(model_dir.into(), device)?;

    Ok(config.init(device).load_record(record))
}

fn arb_tensor<B: Backend, const D: usize>(dims: [usize; D], device: &B::Device) -> Tensor<B, D> {
    let prod: usize = dims.iter().cloned().product();
    Tensor::arange(0..prod as i64, device).float().sin().reshape(dims)
}

use stablediffusion::token::{clip::ClipTokenizer, open_clip::OpenClipTokenizer, Tokenizer};

use burn::tensor::ElementConversion;
use num_traits::cast::ToPrimitive;
use stablediffusion::model::stablediffusion::Conditioning;

use stablediffusion::backend_converter::*;

use burn::tensor::Bool;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opts {
    /// Directory of the model weights
    #[structopt(parse(from_os_str), short = "md", long)]
    model_dir: PathBuf, 

    /// Use the refiner model?
    #[structopt(short = "ref", long)]
    use_refiner: bool, 

    /// Path of the reference image for inpainting
    #[structopt(parse(from_os_str), short = "rd", long)]
    reference_img: Option<PathBuf>, 

    /// Left-most pixel of the crop window
    #[structopt(long)]
    crop_left: Option<usize>,
    
    /// Right-most pixel of the crop window
    #[structopt(long)]
    crop_right: Option<usize>,

    /// Top-most pixel of the crop window
    #[structopt(long)]
    crop_top: Option<usize>,

    /// Bottom-most pixel of the crop window
    #[structopt(long)]
    crop_bottom: Option<usize>,

    /// Crop outside or inside the specified crop window?
    #[structopt(long)]
    crop_out: bool, 

    /// Controls the strength of the adherence to the prompt
    #[structopt(short = "gs", long, default_value = "7.5")]
    unconditional_guidance_scale: f64,

    /// Number of diffusion iterations used for generating the image
    #[structopt(short = "steps", long, default_value = "30")]
    n_diffusion_steps: usize,

    #[structopt(short = "pr", long)]
    prompt: String,

    /// Directory of the image outputs
    #[structopt(parse(from_os_str), short = "od", long)]
    output_dir: PathBuf,
}

type TorchBackend = LibTorch<f32>;
type Backend_f16 = LibTorch<tensor::f16>;

struct InpaintingTensors {
    orig_dims: (usize, usize), 
    reference_latent: Tensor<Backend_f16, 4>, 
    mask: Tensor<Backend_f16, 4, Bool>, 
}

fn main() {
    let device = LibTorchDevice::Cuda(0);

    let opts = Opts::from_args();

    let inpainting_info = opts.reference_img.map(|ref_dir| {
        let imgs = load_images(&[ref_dir.to_str().unwrap().into()]).unwrap();

        let crop_left = opts.crop_left.unwrap_or(0);
        let crop_right = opts.crop_right.unwrap_or(imgs.width);
        let crop_top = opts.crop_top.unwrap_or(0);
        let crop_bottom = opts.crop_bottom.unwrap_or(imgs.height);

        assert!(crop_right <= imgs.width && crop_bottom <= imgs.height && crop_left < crop_right || crop_top < crop_bottom, "Invalid crop parameters.");

        // compute latent
        println!("Loading latent encoder...");
        let latent_decoder: LatentDecoder<TorchBackend> =
            load_latent_decoder_model(&format!("{}/latent_decoder", opts.model_dir.to_str().unwrap()), &device).unwrap();

        println!("Running encoder...");

        let latent = latent_decoder.image_to_latent(&imgs, &device);
        let latent = DefaultBackendConverter::new().convert(latent, &device);

        // get converted pixels idxs
        let [_, _, height, width] = latent.dims();
        let scale = imgs.height / height;
        let crop_left = crop_left / scale;
        let crop_right = crop_right / scale;
        let crop_top = crop_top / scale;
        let crop_bottom = crop_bottom / scale;

        // compute mask
        let crop_width = crop_right - crop_left;
        let crop_height = crop_bottom - crop_top;

        let pad_left = crop_left;
        let pad_right = width - crop_right;

        let pad_top = crop_top;
        let pad_bottom = height - crop_bottom;

        let mask = Tensor::<Backend_f16, 2>::ones([crop_height, crop_width], &device)
            .pad( (pad_left, pad_right, pad_top, pad_bottom), 0.0.elem() )
            .bool()
            .unsqueeze::<4>()
            .expand([1, 4, height, width]);
        let mask = if opts.crop_out {
            mask.bool_not()
        } else {
            mask
        };

        InpaintingTensors {
            orig_dims: (imgs.width, imgs.height), 
            reference_latent: latent, 
            mask: mask.unsqueeze::<4>(), 
        }
    });

    /*let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!("Usage: {} <model_dir> <refiner(y/n)> <unconditional_guidance_scale> <n_diffusion_steps> <prompt> <output_image_name>", args[0]);
        process::exit(1);
    }*/


    /*let unconditional_guidance_scale: f64 = args[3].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid unconditional guidance scale.");
        process::exit(1);
    });
    let n_steps: usize = args[4].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid number of diffusion steps.");
        process::exit(1);
    });
    let prompt = &args[5];
    let output_image_name = &args[6];*/

    let conditioning = {
        println!("Loading embedder...");
        let embedder: Embedder<TorchBackend> =
            load_embedder_model(&format!("{}/embedder", opts.model_dir.to_str().unwrap()), &device).unwrap();

        let resolution = if let Some(inpainting_info) = inpainting_info.as_ref() {
            [inpainting_info.orig_dims.1 as i32, inpainting_info.orig_dims.0 as i32]
        } else {
            [1024, 1024]
        }; //RESOLUTIONS[8];

        let size = Tensor::from_ints(resolution, &device).unsqueeze();
        let crop = Tensor::from_ints([0, 0], &device).unsqueeze();
        let ar = Tensor::from_ints(resolution, &device).unsqueeze();

        println!("Running embedder...");
        embedder.text_to_conditioning(&opts.prompt, size, crop, ar)
    };

    let conditioning: Conditioning<Backend_f16> =
        conditioning.convert(DefaultBackendConverter::new(), &device);

    let latent = {
        println!("Loading diffuser...");
        let diffuser: Diffuser<Backend_f16> =
            load_diffuser_model(&format!("{}/diffuser", opts.model_dir.to_str().unwrap()), &device).unwrap();

        if let Some(inpainting_info) = inpainting_info {
            diffuser.sample_latent_with_inpainting(conditioning.clone(), opts.unconditional_guidance_scale, opts.n_diffusion_steps, inpainting_info.reference_latent, inpainting_info.mask)
        } else {
            println!("Running diffuser...");
            diffuser.sample_latent(conditioning.clone(), opts.unconditional_guidance_scale, opts.n_diffusion_steps)
        }
    };

    let latent = if opts.use_refiner {
        println!("Loading refiner...");
        let diffuser: Diffuser<Backend_f16> =
            load_diffuser_model(&format!("{}/refiner", opts.model_dir.to_str().unwrap()), &device).unwrap();

        println!("Running refiner...");
        diffuser.refine_latent(
            latent,
            conditioning,
            opts.unconditional_guidance_scale,
            800,
            opts.n_diffusion_steps,
        )
    } else {
        latent
    };

    let latent: Tensor<TorchBackend, 4> = DefaultBackendConverter::new().convert(latent, &device);

    let images = {
        println!("Loading latent decoder...");
        let latent_decoder: LatentDecoder<TorchBackend> =
            load_latent_decoder_model(&format!("{}/latent_decoder", opts.model_dir.to_str().unwrap()), &device).unwrap();

        println!("Running decoder...");
        latent_decoder.latent_to_image(latent)
    };

    println!("Saving images...");
    save_images(
        &images.buffer,
        opts.output_dir.to_str().unwrap(),
        images.width as u32,
        images.height as u32,
    )
    .unwrap();
    println!("Done.");

    return;
}

use image::{self, ColorType::Rgb8, RgbImage, ImageError, ImageResult};
use image::io::Reader as ImageReader;

fn load_images(filenames: &[String]) -> Result<RawImages, ImgLoadError> {
    let images = filenames
        .into_iter()
        .map(|filename| load_image(&filename))
        .collect::<ImageResult<Vec<RgbImage>>>()?;

    let (width, height) = images.first().map(|img| img.dimensions()).ok_or(ImgLoadError::NoImages)?;

    if !images.iter().map(|img| img.dimensions()).all(|d| d == (width, height) ) {
        return Err(ImgLoadError::DifferentDimensions);
    }

    let image_buffers: Vec<Vec<u8>> = images
        .into_iter()
        .map(|image| image.into_vec())
        .collect();

    Ok(
        RawImages {
            buffer: image_buffers, 
            width: width as usize, 
            height: height as usize, 
        }
    )
}

#[derive(Debug)]
enum ImgLoadError {
    DifferentDimensions, 
    NoImages, 
    ImageError(ImageError), 
}

impl From<ImageError> for ImgLoadError {
    fn from(err: ImageError) -> Self {
        ImgLoadError::ImageError(err)
    }
}

fn load_image(filename: &str) -> ImageResult<RgbImage> {
    Ok(
        ImageReader::open(filename)?.decode()?.to_rgb8()
    )
}

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
