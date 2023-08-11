use super::LayerNorm;
use crate::model::load::*;

use std::error::Error;

use burn::{
    config::Config, 
    module::{Module, Param},
    nn,
    tensor::{
        backend::Backend,
        Tensor,
    },
};

pub fn load_layer_norm<B: Backend>(path: &str, device: &B::Device) -> Result<LayerNorm<B>, Box<dyn Error>> {
    let eps = load_f32::<B>("eps", path, device)?.into();

    let gamma = load_tensor::<B, 1>("weight", path, device)?.into();
    let beta = load_tensor::<B, 1>("bias", path, device)?.into();

    Ok( 
        LayerNorm { 
            gamma, 
            beta, 
            eps, 
        } 
    )
}