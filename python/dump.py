import math

import numpy as np
import torch as th
import torch.nn as nn


from sgm.inference.api import (
    model_specs,
    SamplingParams,
    SamplingPipeline,
    Sampler,
    ModelArchitecture,
)


def arb_tensor(dims):
    values = th.arange(th.prod(th.tensor(dims)).item()).sin()
    return values.reshape(dims)


import clip as clipsave
import unet as unetsave
import autoencoder as autoencodersave
import save

from sgm.modules.diffusionmodules.discretizer import LegacyDDPMDiscretization

def alphas_cumprod():
    dis = LegacyDDPMDiscretization()
    return th.tensor(dis.alphas_cumprod)

def dump_base(model):
    ac = alphas_cumprod()
    save.save_scalar(ac.shape[0], 'n_steps', "params")
    save.save_tensor(ac, "alphas_cumprod", "params")

    clipsave.save_clip_text_transformer(model.conditioner.embedders[0].transformer.text_model, "params/clip")
    clipsave.save_open_clip_text_transformer(model.conditioner.embedders[1].model, "params/open_clip")
    del model.conditioner

    unetsave.save_unet_model(model.model.diffusion_model, "params/diffuser_base", refiner=False)
    del model.model.diffusion_model

    autoencodersave.save_autoencoder(model.first_stage_model, "params/autoencoder")
    del model.first_stage_model

def dump_refiner(model):
    ac = alphas_cumprod()
    save.save_scalar(ac.shape[0], 'n_steps', "params")
    save.save_tensor(ac, "alphas_cumprod", "params")

    unetsave.save_unet_model(model.model.diffusion_model, "params/diffuser_refiner", refiner=True)
    del model.model.diffusion_model

if __name__ == "__main__":
    base_pipeline = SamplingPipeline(ModelArchitecture.SDXL_V1_BASE, device = "cpu")
    dump(base_pipeline.model)

    refiner_pipeline = SamplingPipeline(ModelArchitecture.SDXL_V1_REFINER, device = "cpu")
    dump_refiner(refiner_pipeline.model)

    print("done.")
