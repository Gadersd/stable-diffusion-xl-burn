use burn::{
    tensor::{backend::Backend, Data, ElementConversion, Tensor},
};

use std::marker::PhantomData;

pub trait BackendConverter<B2: Backend> {
    fn convert<B: Backend, const D: usize>(&self, x: Tensor<B, D>, device: &B2::Device) -> Tensor<B2, D>;
}

pub struct DefaultBackendConverter<B: Backend> {
    _dummy: PhantomData<B>, 
}

impl<B: Backend> DefaultBackendConverter<B> {
    pub fn new() -> Self {
        Self {
            _dummy: PhantomData
        }
    }
}

impl<B2: Backend> BackendConverter<B2> for DefaultBackendConverter<B2> {
    fn convert<B: Backend, const D: usize>(&self, x: Tensor<B, D>, device: &B2::Device) -> Tensor<B2, D> {
        let data = x.into_data();

        let data = Data::new(
            data.value.into_iter().map(|v| v.elem()).collect(),
            data.shape,
        );
    
        Tensor::from_data_device(data, device)
    }
}
