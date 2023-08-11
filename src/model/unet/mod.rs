pub mod load;

use burn::{
    config::Config, 
    module::{Module, Param},
    nn::{self, PaddingConfig2d, GELU, conv::{Conv2d, Conv2dConfig}},
    tensor::{
        backend::Backend,
        activation::softmax, 
        module::embedding, 
        Tensor,
        Distribution, 
        Int, 
    },
};

use super::silu::*;
use super::groupnorm::*;
use crate::helper::to_float;
use crate::model::layernorm::{LayerNorm, LayerNormConfig};
use super::attention::qkv_attention;


pub fn timestep_embedding<B: Backend>(timesteps: Tensor<B, 1, Int>, dim: usize, max_period: usize) -> Tensor<B, 2> {
    let [n_batch] = timesteps.dims();

    let half = dim / 2;
    let freqs = ( to_float(Tensor::arange_device(0..half, &timesteps.device())) * (-(max_period as f64).ln() / half as f64 ) ).exp();
    let args = to_float(timesteps).unsqueeze::<2>().transpose().repeat(1, half) * freqs.unsqueeze();
    Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
}


pub fn conditioning_embedding<B: Backend>(pooled_text_enc: Tensor<B, 2>, dim: usize, size: Tensor<B, 2, Int>, crop: Tensor<B, 2, Int>, ar: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let [n_batch, _] = pooled_text_enc.dims();

    let cat = Tensor::cat(vec![size, crop, ar], 1);
    let [n_batch, w] = cat.dims();

    let embed = timestep_embedding(cat.reshape([n_batch * w]), dim, 10000).reshape([n_batch, w * dim]);

    Tensor::cat(vec![pooled_text_enc, embed], 1)
}


#[derive(Config)]
pub struct UNetConfig {
    adm_in_channels: usize, 
    in_channels: usize, 
    out_channels: usize, 
    model_channels: usize, 
    n_head_channels: usize, 
    context_dim: usize, 
}

impl UNetConfig {
    pub fn init<B: Backend>(&self) -> UNet<B> {
        assert!(self.model_channels % self.n_head_channels == 0, "The number of head channels must evenly divide the model channels.");

        let time_embed_dim = self.model_channels * 4;

        let lin1_time_embed = nn::LinearConfig::new(self.model_channels, time_embed_dim).init();
        let silu_time_embed = SILU::new();
        let lin2_time_embed = nn::LinearConfig::new(time_embed_dim, time_embed_dim).init();

        let lin1_label_embed = nn::LinearConfig::new(self.adm_in_channels, time_embed_dim).init();
        let silu_label_embed = SILU::new();
        let lin2_label_embed = nn::LinearConfig::new(time_embed_dim, time_embed_dim).init();

        let model_channels = self.model_channels;

        /*## Resblock: ch320, t_embed_dim1280, out_channels=320, dims=2, use_scale_shift_norm=False
        ##Appended layer
        ## Resblock: ch320, t_embed_dim1280, out_channels=320, dims=2, use_scale_shift_norm=False
        ##Appended layer
        ##Downsample: ch320, conv_resampleTrue, dims2, out_channels=4
        ##Appended Layer
        ## Resblock: ch320, t_embed_dim1280, out_channels=640, dims=2, use_scale_shift_norm=False
        ##SpatialTransformer: ch=640, num_heads10, dim_head:64, depth2, context_dim2048, attn_typesoftmax-xformers
        ##Appended layer
        ## Resblock: ch640, t_embed_dim1280, out_channels=640, dims=2, use_scale_shift_norm=False
        ##SpatialTransformer: ch=640, num_heads10, dim_head:64, depth2, context_dim2048, attn_typesoftmax-xformers
        ##Appended layer
        ##Downsample: ch640, conv_resampleTrue, dims2, out_channels=4
        ##Appended Layer
        ## Resblock: ch640, t_embed_dim1280, out_channels=1280, dims=2, use_scale_shift_norm=False
        ##SpatialTransformer: ch=1280, num_heads20, dim_head:64, depth10, context_dim2048, attn_typesoftmax-xformers
        ##Appended layer
        ## Resblock: ch1280, t_embed_dim1280, out_channels=1280, dims=2, use_scale_shift_norm=False
        ##SpatialTransformer: ch=1280, num_heads20, dim_head:64, depth10, context_dim2048, attn_typesoftmax-xformers
        ##Appended layer*/

        let n_head = |channels| {
            channels / self.n_head_channels
        };

        let input_blocks = UNetInputBlocks {
            conv: Conv2dConfig::new([self.in_channels, self.model_channels], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1)).init(),
            r1: ResBlockConfig::new(self.model_channels, time_embed_dim, self.model_channels).init(), 
            r2: ResBlockConfig::new(self.model_channels, time_embed_dim, self.model_channels).init(),
            d1: DownsampleConfig::new(self.model_channels).init(), 
            rt1: ResTransformerConfig::new(self.model_channels, time_embed_dim, 2 * self.model_channels, self.context_dim, n_head(2 * self.model_channels), 2).init(), 
            rt2: ResTransformerConfig::new(2 * self.model_channels, time_embed_dim, 2 * self.model_channels, self.context_dim, n_head(2 * self.model_channels), 2).init(), 
            d2: DownsampleConfig::new(2 * self.model_channels).init(), 
            rt3: ResTransformerConfig::new(2 * self.model_channels, time_embed_dim, 4 * self.model_channels, self.context_dim, n_head(4 * self.model_channels), 10).init(), 
            rt4: ResTransformerConfig::new(4 * self.model_channels, time_embed_dim, 4 * self.model_channels, self.context_dim, n_head(4 * self.model_channels), 10).init(), 
        };

        /*let input_blocks = UNetInputBlocks {
            conv: Conv2dConfig::new([self.in_channels, self.model_channels], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1)).init(),
            rt1: ResTransformerConfig::new(320, 1280, 320, 768, 8).init(),
            rt2: ResTransformerConfig::new(320, 1280, 320, 768, 8).init(),
            d1: DownsampleConfig::new(320).init(),
            rt3: ResTransformerConfig::new(320, 1280, 640, 768, 8).init(),
            rt4: ResTransformerConfig::new(640, 1280, 640, 768, 8).init(),
            d2: DownsampleConfig::new(640).init(),
            rt5: ResTransformerConfig::new(640, 1280, 1280, 768, 8).init(),
            rt6: ResTransformerConfig::new(1280, 1280, 1280, 768, 8).init(),
            d3: DownsampleConfig::new(1280).init(),
            r1: ResBlockConfig::new(1280, 1280, 1280).init(),
            r2: ResBlockConfig::new(1280, 1280, 1280).init(),
        };*/
        
        let middle_block = ResTransformerResConfig::new(
            4 * self.model_channels, 
            4 * self.model_channels, 
            4 * self.model_channels, 
            self.context_dim, 
            n_head(4 * self.model_channels), 
            10
        ).init();

        let output_blocks = UNetOutputBlocks {
            rt1: ResTransformerConfig::new(8 * self.model_channels, time_embed_dim, 4 * self.model_channels, self.context_dim, n_head(4 * self.model_channels), 10).init(), 
            rt2: ResTransformerConfig::new(8 * self.model_channels, time_embed_dim, 4 * self.model_channels, self.context_dim, n_head(4 * self.model_channels), 10).init(), 
            rtu1: ResTransformerUpsampleConfig::new(6 * self.model_channels, time_embed_dim, 4 * self.model_channels, self.context_dim, n_head(4 * self.model_channels), 10).init(), 
            rt3: ResTransformerConfig::new(6 * self.model_channels, time_embed_dim, 2 * self.model_channels, self.context_dim, n_head(2 * self.model_channels), 2).init(), 
            rt4: ResTransformerConfig::new(4 * self.model_channels, time_embed_dim, 2 * self.model_channels, self.context_dim, n_head(2 * self.model_channels), 2).init(), 
            rtu2: ResTransformerUpsampleConfig::new(3 * self.model_channels, time_embed_dim, 2 * self.model_channels, self.context_dim, n_head(2 * self.model_channels), 2).init(),
            r1: ResBlockConfig::new(3 * self.model_channels, time_embed_dim, self.model_channels).init(), 
            r2: ResBlockConfig::new(2 * self.model_channels, time_embed_dim, self.model_channels).init(), 
            r3: ResBlockConfig::new(2 * self.model_channels, time_embed_dim, self.model_channels).init(), 
        };

        /*let output_blocks = UNetOutputBlocks {
            r1: ResBlockConfig::new(2560, 1280, 1280).init(),
            r2: ResBlockConfig::new(2560, 1280, 1280).init(),
            ru: ResUpSampleConfig::new(2560, 1280, 1280).init(),
            rt1: ResTransformerConfig::new(2560, 1280, 1280, 768, 8).init(),
            rt2: ResTransformerConfig::new(2560, 1280, 1280, 768, 8).init(),
            rtu1: ResTransformerUpsampleConfig::new(1920, 1280, 1280, 768, 8).init(),
            rt3: ResTransformerConfig::new(1920, 1280, 640, 768, 8).init(),
            rt4: ResTransformerConfig::new(1280, 1280, 640, 768, 8).init(),
            rtu2: ResTransformerUpsampleConfig::new(960, 1280, 640, 768, 8).init(),
            rt5: ResTransformerConfig::new(960, 1280, 320, 768, 8).init(),
            rt6: ResTransformerConfig::new(640, 1280, 320, 768, 8).init(),
            rt7: ResTransformerConfig::new(640, 1280, 320, 768, 8).init(),
        };*/

        let norm_out = GroupNormConfig::new(32, self.model_channels).init();
        let silu_out = SILU::new();
        let conv_out = Conv2dConfig::new([self.model_channels, self.out_channels], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1)).init();

        UNet {
            model_channels, 
            lin1_time_embed, 
            silu_time_embed, 
            lin2_time_embed, 
            lin1_label_embed, 
            silu_label_embed, 
            lin2_label_embed, 
            input_blocks, 
            middle_block, 
            output_blocks, 
            norm_out, 
            silu_out, 
            conv_out, 
        }
    }
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    model_channels: usize, 
    lin1_time_embed: nn::Linear<B>, 
    silu_time_embed: SILU, 
    lin2_time_embed: nn::Linear<B>, 
    lin1_label_embed: nn::Linear<B>, 
    silu_label_embed: SILU, 
    lin2_label_embed: nn::Linear<B>, 
    input_blocks: UNetInputBlocks<B>, 
    middle_block: ResTransformerRes<B>, 
    output_blocks: UNetOutputBlocks<B>, 
    norm_out: GroupNorm<B>, 
    silu_out: SILU, 
    conv_out: Conv2d<B>, 
}

impl<B: Backend> UNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>, timesteps: Tensor<B, 1, Int>, context: Tensor<B, 3>, label: Tensor<B, 2>) -> Tensor<B, 4> {
        // embed the timestep
        let t_emb = timestep_embedding(timesteps, self.model_channels, 10000);
        let t_emb = self.lin1_time_embed.forward(t_emb);
        let t_emb = self.silu_time_embed.forward(t_emb);
        let t_emb = self.lin2_time_embed.forward(t_emb);

        // embed the labels
        let label_emb = self.lin1_label_embed.forward(label);
        let label_emb = self.silu_label_embed.forward(label_emb);
        let label_emb = self.lin2_label_embed.forward(label_emb);

        let emb = t_emb + label_emb;

        let mut saved_inputs = Vec::new();
        let mut x = x;

        // input blocks
        for block in self.input_blocks.as_array() {
            x = block.forward(x, emb.clone(), context.clone());
            saved_inputs.push(x.clone())
        }

        // middle block
        x = self.middle_block.forward(x, emb.clone(), context.clone());

        // output blocks
        for block in self.output_blocks.as_array() {
            x = Tensor::cat(vec![x, saved_inputs.pop().unwrap()], 1);
            x = block.forward(x, emb.clone(), context.clone());
        }

        let x = self.norm_out.forward(x);
        let x = self.silu_out.forward(x);
        let x = self.conv_out.forward(x);
        x
    }
}



#[derive(Module, Debug)]
pub struct UNetInputBlocks<B: Backend> {
    conv: Conv2d<B>, 
    r1: ResBlock<B>,
    r2: ResBlock<B>,
    d1: Downsample<B>, 
    rt1: ResTransformer<B>, 
    rt2: ResTransformer<B>, 
    d2: Downsample<B>, 
    rt3: ResTransformer<B>, 
    rt4: ResTransformer<B>, 
}

impl<B: Backend> UNetInputBlocks<B> {
    fn as_array(&self) -> [&dyn UNetBlock<B>; 9] {
        [
            &self.conv, 
            &self.r1, 
            &self.r2,
            &self.d1,
            &self.rt1, 
            &self.rt2,
            &self.d2,
            &self.rt3,
            &self.rt4,
        ]
    }
}

#[derive(Module, Debug)]
pub struct UNetOutputBlocks<B: Backend> {
    rt1: ResTransformer<B>,
    rt2: ResTransformer<B>,
    rtu1: ResTransformerUpsample<B>,
    rt3: ResTransformer<B>,
    rt4: ResTransformer<B>,
    rtu2: ResTransformerUpsample<B>,
    r1: ResBlock<B>,
    r2: ResBlock<B>,
    r3: ResBlock<B>,
}

impl<B: Backend> UNetOutputBlocks<B> {
    fn as_array(&self) -> [&dyn UNetBlock<B>; 9] {
        [
            &self.rt1,
            &self.rt2,
            &self.rtu1,
            &self.rt3,
            &self.rt4,
            &self.rtu2,
            &self.r1,
            &self.r2,
            &self.r3,
        ]
    }
}





trait UNetBlock<B: Backend> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4>;
}

#[derive(Config)]
pub struct ResTransformerConfig {
    n_channels_in: usize, 
    n_channels_embed: usize, 
    n_channels_out: usize, 
    n_context_state: usize, 
    n_head: usize, 
    n_transformer_blocks: usize, 
}

impl ResTransformerConfig {
    fn init<B: Backend>(&self) -> ResTransformer<B> {
        let res = ResBlockConfig::new(self.n_channels_in, self.n_channels_embed, self.n_channels_out).init();
        let transformer = SpatialTransformerConfig::new(
            self.n_channels_out, 
            self.n_context_state, 
            self.n_head, 
            self.n_transformer_blocks
        ).init();

        ResTransformer {
            res, 
            transformer, 
        }
    }
}

#[derive(Module, Debug)]
pub struct ResTransformer<B: Backend> {
    res: ResBlock<B>, 
    transformer: SpatialTransformer<B>, 
}

impl<B: Backend> UNetBlock<B> for ResTransformer<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res.forward(x, emb);
        let x = self.transformer.forward(x, context);
        x
    }
}

#[derive(Config)]
pub struct ResUpSampleConfig {
    n_channels_in: usize, 
    n_channels_embed: usize, 
    n_channels_out: usize, 
}

impl ResUpSampleConfig {
    fn init<B: Backend>(&self) -> ResUpSample<B> {
        let res = ResBlockConfig::new(self.n_channels_in, self.n_channels_embed, self.n_channels_out).init();
        let upsample = UpsampleConfig::new(self.n_channels_out).init();

        ResUpSample {
            res, 
            upsample, 
        }
    }
}

#[derive(Module, Debug)]
pub struct ResUpSample<B: Backend> {
    res: ResBlock<B>, 
    upsample: Upsample<B>, 
}

impl<B: Backend> UNetBlock<B> for ResUpSample<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res.forward(x, emb);
        let x = self.upsample.forward(x);
        x
    }
}

#[derive(Config)]
pub struct ResTransformerUpsampleConfig {
    n_channels_in: usize, 
    n_channels_embed: usize, 
    n_channels_out: usize, 
    n_context_state: usize, 
    n_head: usize, 
    n_transformer_blocks: usize, 
}

impl ResTransformerUpsampleConfig {
    fn init<B: Backend>(&self) -> ResTransformerUpsample<B> {
        let res = ResBlockConfig::new(self.n_channels_in, self.n_channels_embed, self.n_channels_out).init();
        let transformer = SpatialTransformerConfig::new(
            self.n_channels_out, 
            self.n_context_state, 
            self.n_head, 
            self.n_transformer_blocks
        ).init();
        let upsample = UpsampleConfig::new(self.n_channels_out).init();

        ResTransformerUpsample {
            res, 
            transformer, 
            upsample, 
        }
    }
}

#[derive(Module, Debug)]
pub struct ResTransformerUpsample<B: Backend> {
    res: ResBlock<B>, 
    transformer: SpatialTransformer<B>, 
    upsample: Upsample<B>, 
}

impl<B: Backend> UNetBlock<B> for ResTransformerUpsample<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res.forward(x, emb);
        let x = self.transformer.forward(x, context);
        let x = self.upsample.forward(x);
        x
    }
}

#[derive(Config)]
pub struct ResTransformerResConfig {
    n_channels_in: usize, 
    n_channels_embed: usize, 
    n_channels_out: usize, 
    n_context_state: usize, 
    n_head: usize, 
    n_transformer_blocks: usize, 
}

impl ResTransformerResConfig {
    fn init<B: Backend>(&self) -> ResTransformerRes<B> {
        let res1 = ResBlockConfig::new(self.n_channels_in, self.n_channels_embed, self.n_channels_out).init();
        let transformer = SpatialTransformerConfig::new(
            self.n_channels_out, 
            self.n_context_state, 
            self.n_head, 
            self.n_transformer_blocks
        ).init();
        let res2 = ResBlockConfig::new(self.n_channels_in, self.n_channels_embed, self.n_channels_out).init();

        ResTransformerRes {
            res1, 
            transformer, 
            res2, 
        }
    }
}

#[derive(Module, Debug)]
pub struct ResTransformerRes<B: Backend> {
    res1: ResBlock<B>, 
    transformer: SpatialTransformer<B>, 
    res2: ResBlock<B>, 
}

impl<B: Backend> UNetBlock<B> for ResTransformerRes<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res1.forward(x, emb.clone());
        let x = self.transformer.forward(x, context);
        let x = self.res2.forward(x, emb);
        x
    }
}



#[derive(Config)]
pub struct UpsampleConfig {
    n_channels: usize, 
}

impl UpsampleConfig {
    fn init<B: Backend>(&self) -> Upsample<B> {
        let conv = Conv2dConfig::new([self.n_channels, self.n_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        Upsample {
            conv, 
        }
    }
}

#[derive(Module, Debug)]
pub struct Upsample<B: Backend> {
    conv: Conv2d<B>, 
}

impl<B: Backend> Upsample<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n_batch, n_channel, height, width] = x.dims();
        let x = x
                .reshape([n_batch, n_channel, height, 1, width, 1])
                .repeat(3, 2)
                .repeat(5, 2)
                .reshape([n_batch, n_channel, 2 * height, 2 * width]);
        self.conv.forward(x)
    }
}

impl<B: Backend> UNetBlock<B> for Upsample<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        self.forward(x)
    }
}

#[derive(Config)]
pub struct DownsampleConfig {
    n_channels: usize, 
}

impl DownsampleConfig {
    fn init<B: Backend>(&self) -> Conv2d<B> {
        Conv2dConfig::new([self.n_channels, self.n_channels], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init()
    }
}

type Downsample<B> = Conv2d<B>;

impl<B: Backend> UNetBlock<B> for Conv2d<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        self.forward(x)
    }
}




#[derive(Config)]
pub struct SpatialTransformerConfig {
    n_channels: usize, 
    n_context_state: usize, 
    n_head: usize, 
    n_blocks: usize, 
}

impl SpatialTransformerConfig {
    fn init<B: Backend>(&self) -> SpatialTransformer<B> {
        let norm = GroupNormConfig::new(32, self.n_channels).init();
        let proj_in = nn::LinearConfig::new(self.n_channels, self.n_channels).init(); //Conv2dConfig::new([self.n_channels, self.n_channels], [1, 1]).init();
        let blocks = (0..self.n_blocks)
            .into_iter()
            .map(|_| TransformerBlockConfig::new(self.n_channels, self.n_context_state, self.n_head).init())
            .collect();
        let proj_out = nn::LinearConfig::new(self.n_channels, self.n_channels).init(); //Conv2dConfig::new([self.n_channels, self.n_channels], [1, 1]).init();

        SpatialTransformer {
            norm, 
            proj_in, 
            blocks, 
            proj_out, 
        }
    }
}

#[derive(Module, Debug)]
pub struct SpatialTransformer<B: Backend> {
    norm: GroupNorm<B>, 
    proj_in: nn::Linear<B>, 
    blocks: Vec<TransformerBlock<B>>, 
    proj_out: nn::Linear<B>, 
}

impl<B: Backend> SpatialTransformer<B> {
    fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let [n_batch, n_channel, height, width] = x.dims();

        let x_in = x.clone();

        let x = self.norm.forward(x);
        let x = x.reshape([n_batch, n_channel, height * width]).swap_dims(1, 2);
        let x = self.proj_in.forward(x);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, context.clone());
        }

        let x = self.proj_out.forward(x)
            .swap_dims(1, 2)
            .reshape([n_batch, n_channel, height, width]);

        x_in + x
    }
}








#[derive(Config)]
pub struct TransformerBlockConfig {
    n_state: usize, 
    n_context_state: usize, 
    n_head: usize, 
}

impl TransformerBlockConfig {
    fn init<B: Backend>(&self) -> TransformerBlock<B> {
        let norm1 = LayerNormConfig::new(self.n_state).init();
        let attn1 = MultiHeadAttentionConfig::new(self.n_state, self.n_state, self.n_head).init();
        let norm2 = LayerNormConfig::new(self.n_state).init();
        let attn2 = MultiHeadAttentionConfig::new(self.n_state, self.n_context_state, self.n_head).init();
        let norm3 = LayerNormConfig::new(self.n_state).init();
        let mlp = MLPConfig::new(self.n_state, 4).init();

        TransformerBlock {
            norm1, 
            attn1, 
            norm2, 
            attn2, 
            norm3, 
            mlp, 
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: LayerNorm<B>, 
    attn1: MultiHeadAttention<B>, 
    norm2: LayerNorm<B>, 
    attn2: MultiHeadAttention<B>, 
    norm3: LayerNorm<B>, 
    mlp: MLP<B>, 
}

impl<B: Backend> TransformerBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn1.forward( self.norm1.forward(x), None);
        let x = x.clone() + self.attn2.forward( self.norm2.forward(x), Some(context));
        x.clone() + self.mlp.forward( self.norm3.forward(x) )
    }
}


#[derive(Config)]
pub struct MLPConfig {
    n_state: usize, 
    mult: usize, 
}

impl MLPConfig {
    pub fn init<B: Backend>(&self) -> MLP<B> {
        let n_state_hidden = self.n_state * self.mult;
        let geglu = GEGLUConfig::new(self.n_state, n_state_hidden).init();
        let lin = nn::LinearConfig::new(n_state_hidden, self.n_state).init();

        MLP {
            geglu, 
            lin, 
        }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    geglu: GEGLU<B>, 
    lin: nn::Linear<B>, 
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.lin.forward( self.geglu.forward(x) )
    }
}


#[derive(Config)]
pub struct GEGLUConfig {
    n_state_in: usize, 
    n_state_out: usize, 
}

impl GEGLUConfig {
    fn init<B: Backend>(&self) -> GEGLU<B> {
        let proj = nn::LinearConfig::new(self.n_state_in, 2 * self.n_state_out).init();
        let gelu = GELU::new();

        GEGLU {
            proj, 
            gelu, 
        }
    }
}

#[derive(Module, Debug)]
pub struct GEGLU<B: Backend> {
    proj: nn::Linear<B>, 
    gelu: GELU, 
}

impl<B: Backend> GEGLU<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let projected = self.proj.forward(x);
        let [n_batch, n_ctx, n_state] = projected.dims();

        let n_state_out = n_state / 2;

        let x = projected.clone().slice([0..n_batch, 0..n_ctx, 0..n_state_out]);
        let gate = projected.slice([0..n_batch, 0..n_ctx, n_state_out..n_state]);

        x * self.gelu.forward(gate)
    }
}





#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    n_state: usize, 
    n_context_state: usize, 
    n_head: usize, 
}

impl MultiHeadAttentionConfig {
    fn init<B: Backend>(&self) -> MultiHeadAttention<B> {
        assert!(self.n_state % self.n_head == 0, "State size {} must be a multiple of head size {}", self.n_state, self.n_head);

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).with_bias(false).init();
        let key = nn::LinearConfig::new(self.n_context_state, self.n_state).with_bias(false).init();
        let value = nn::LinearConfig::new(self.n_context_state, self.n_state).with_bias(false).init();
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init();

        MultiHeadAttention { 
            n_head, 
            query, 
            key, 
            value, 
            out 
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    n_head: usize, 
    query: nn::Linear<B>, 
    key: nn::Linear<B>, 
    value: nn::Linear<B>, 
    out: nn::Linear<B>, 
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, context: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let xa = context.unwrap_or_else(|| x.clone());

        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);

        let wv = qkv_attention(q, k, v, None, self.n_head);

        self.out.forward(wv)
    }
}
















#[derive(Config)]
pub struct ResBlockConfig {
    n_channels_in: usize, 
    n_channels_embed: usize, 
    n_channels_out: usize, 
}


impl ResBlockConfig {
    fn init<B: Backend>(&self) -> ResBlock<B> {
        let norm_in = GroupNormConfig::new(32, self.n_channels_in).init();
        let silu_in = SILU::new();
        let conv_in = Conv2dConfig::new([self.n_channels_in, self.n_channels_out], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1)).init();

        let silu_embed = SILU::new();
        let lin_embed = nn::LinearConfig::new(self.n_channels_embed, self.n_channels_out).init();

        let norm_out = GroupNormConfig::new(32, self.n_channels_out).init();
        let silu_out = SILU::new();
        let conv_out = Conv2dConfig::new([self.n_channels_out, self.n_channels_out], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1)).init();

        let skip_connection = if self.n_channels_in != self.n_channels_out {
            Some( Conv2dConfig::new([self.n_channels_in, self.n_channels_out], [1, 1]).init() )
        } else {
            None
        };

        ResBlock {
            norm_in, 
            silu_in, 
            conv_in, 
            silu_embed, 
            lin_embed, 
            norm_out, 
            silu_out, 
            conv_out, 
            skip_connection, 
        }
    }
}


#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    norm_in: GroupNorm<B>, 
    silu_in: SILU, 
    conv_in: Conv2d<B>, 
    silu_embed: SILU, 
    lin_embed: nn::Linear<B>, 
    norm_out: GroupNorm<B>, 
    silu_out: SILU, 
    conv_out: Conv2d<B>, 
    skip_connection: Option<Conv2d<B>>, 
}

impl<B: Backend> ResBlock<B> {
    fn forward(&self, x: Tensor<B, 4>, embed: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.norm_in.forward(x.clone());
        let h = self.silu_in.forward(h);
        let h = self.conv_in.forward(h);

        let embed_out = self.silu_embed.forward(embed);
        let embed_out = self.lin_embed.forward(embed_out);
        
        let [n_batch_embed, n_state_embed] = embed_out.dims();
        let h = h + embed_out.reshape([n_batch_embed, n_state_embed, 1, 1]);

        let h = self.norm_out.forward(h);
        let h = self.silu_out.forward(h);
        let h = self.conv_out.forward(h);

        let out = if let Some(skipc) = self.skip_connection.as_ref() {
            skipc.forward(x) + h
        } else {
            x + h
        };

        out
    }
}

impl<B: Backend> UNetBlock<B> for ResBlock<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        self.forward(x, emb)
    }
}


