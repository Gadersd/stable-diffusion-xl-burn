pub mod load;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d, GELU,
    },
    tensor::{activation::softmax, module::embedding, Distribution, Int, Tensor},
};

use burn::tensor::backend::Backend;

use super::groupnorm::*;
use super::silu::*;
use crate::backend::Backend as MyBackend;
use crate::helper::to_float;
use crate::model::layernorm::{LayerNorm, LayerNormConfig};

pub fn timestep_embedding<B: Backend>(
    timesteps: Tensor<B, 1, Int>,
    dim: usize,
    max_period: usize,
) -> Tensor<B, 2> {
    let [n_batch] = timesteps.dims();

    let half = dim / 2;
    let freqs = (to_float(Tensor::arange_device(0..half, &timesteps.device()))
        * (-(max_period as f64).ln() / half as f64))
        .exp();
    let args = to_float(timesteps)
        .unsqueeze::<2>()
        .transpose()
        .repeat(1, half)
        * freqs.unsqueeze();
    Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
}

pub fn conditioning_embedding<B: Backend>(
    pooled_text_enc: Tensor<B, 2>,
    dim: usize,
    size: Tensor<B, 2, Int>,
    crop: Tensor<B, 2, Int>,
    ar: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [n_batch, _] = pooled_text_enc.dims();

    let cat = Tensor::cat(vec![size, crop, ar], 1);
    let [n_batch, w] = cat.dims();

    let embed =
        timestep_embedding(cat.reshape([n_batch * w]), dim, 10000).reshape([n_batch, w * dim]);

    Tensor::cat(vec![pooled_text_enc, embed], 1)
}

#[derive(Config)]
pub struct UNetConfig {
    adm_in_channels: usize,
    in_channels: usize,
    out_channels: usize,
    model_channels: usize,
    channel_mults: Vec<usize>,
    n_head_channels: usize,
    transformer_depths: Vec<usize>,
    context_dim: usize,
}

impl UNetConfig {
    pub fn init<B: MyBackend>(&self) -> UNet<B> {
        assert!(
            self.model_channels % self.n_head_channels == 0,
            "The number of head channels must evenly divide the model channels."
        );

        let n_levels = self.channel_mults.len();

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

        let n_head = |channels| channels / self.n_head_channels;

        let mut input_blocks = Vec::new();
        input_blocks.push(UNetBlocks::Conv(
            Conv2dConfig::new([self.in_channels, self.model_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),
        ));
        for level in 0..n_levels {
            let channels_in = self.channel_mults[level.saturating_sub(1)] * self.model_channels;
            let channels_out = self.channel_mults[level] * self.model_channels;

            let (r1, r2) = if level != 1 && level != 2 {
                let r1 = UNetBlocks::Res(
                    ResBlockConfig::new(channels_in, time_embed_dim, channels_out).init(),
                );

                let r2 = UNetBlocks::Res(
                    ResBlockConfig::new(channels_out, time_embed_dim, channels_out).init(),
                );

                (r1, r2)
            } else {
                let n_head = n_head(channels_out);
                let depth = self.transformer_depths[level];

                let rt1 = UNetBlocks::ResT(
                    ResTransformerConfig::new(
                        channels_in,
                        time_embed_dim,
                        channels_out,
                        self.context_dim,
                        n_head,
                        depth,
                    )
                    .init(),
                );

                let rt2 = UNetBlocks::ResT(
                    ResTransformerConfig::new(
                        channels_out,
                        time_embed_dim,
                        channels_out,
                        self.context_dim,
                        n_head,
                        depth,
                    )
                    .init(),
                );

                (rt1, rt2)
            };

            input_blocks.extend([r1, r2]);

            // no downsampling on last block
            if level != n_levels - 1 {
                let d = DownsampleConfig::new(channels_out).init();
                input_blocks.push(UNetBlocks::Down(d));
            }
        }

        /*let input_blocks = UNetInputBlocks {
            conv: Conv2dConfig::new([self.in_channels, self.model_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(),
            r1: ResBlockConfig::new(self.model_channels, time_embed_dim, self.model_channels)
                .init(),
            r2: ResBlockConfig::new(self.model_channels, time_embed_dim, self.model_channels)
                .init(),
            d1: DownsampleConfig::new(self.model_channels).init(),
            rt1: ResTransformerConfig::new(
                self.model_channels,
                time_embed_dim,
                2 * self.model_channels,
                self.context_dim,
                n_head(2 * self.model_channels),
                2,
            )
            .init(),
            rt2: ResTransformerConfig::new(
                2 * self.model_channels,
                time_embed_dim,
                2 * self.model_channels,
                self.context_dim,
                n_head(2 * self.model_channels),
                2,
            )
            .init(),
            d2: DownsampleConfig::new(2 * self.model_channels).init(),
            rt3: ResTransformerConfig::new(
                2 * self.model_channels,
                time_embed_dim,
                4 * self.model_channels,
                self.context_dim,
                n_head(4 * self.model_channels),
                10,
            )
            .init(),
            rt4: ResTransformerConfig::new(
                4 * self.model_channels,
                time_embed_dim,
                4 * self.model_channels,
                self.context_dim,
                n_head(4 * self.model_channels),
                10,
            )
            .init(),
        };*/

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

        let channels_in_middle = self.channel_mults.last().unwrap() * self.model_channels;
        let &depth_middle = self.transformer_depths.last().unwrap();
        let middle_block = ResTransformerResConfig::new(
            channels_in_middle,
            channels_in_middle,
            channels_in_middle,
            self.context_dim,
            n_head(channels_in_middle),
            depth_middle,
        )
        .init();

        let mut output_blocks = Vec::new();
        for level in (0..n_levels).into_iter().rev() {
            let next_level = if level != n_levels - 1 {
                level + 1
            } else {
                level
            };
            let channels_out = self.channel_mults[level] * self.model_channels;

            let channels_in1 = self.channel_mults[next_level] * self.model_channels + channels_out;
            let channels_in2 = 2 * channels_out;
            let channels_in3 =
                channels_out + self.channel_mults[level.saturating_sub(1)] * self.model_channels;

            let (r1, r2, r3) = if level != 1 && level != 2 {
                let r1 = UNetBlocks::Res(
                    ResBlockConfig::new(channels_in1, time_embed_dim, channels_out).init(),
                );

                let r2 = UNetBlocks::Res(
                    ResBlockConfig::new(channels_in2, time_embed_dim, channels_out).init(),
                );

                let r3 = if level != 0 {
                    UNetBlocks::ResU(
                        ResUpsampleConfig::new(channels_in3, time_embed_dim, channels_out).init(),
                    )
                } else {
                    UNetBlocks::Res(
                        ResBlockConfig::new(channels_in3, time_embed_dim, channels_out).init(),
                    )
                };

                (r1, r2, r3)
            } else {
                let n_head = n_head(channels_out);
                let depth = self.transformer_depths[level];

                let rt1 = UNetBlocks::ResT(
                    ResTransformerConfig::new(
                        channels_in1,
                        time_embed_dim,
                        channels_out,
                        self.context_dim,
                        n_head,
                        depth,
                    )
                    .init(),
                );

                let rt2 = UNetBlocks::ResT(
                    ResTransformerConfig::new(
                        channels_in2,
                        time_embed_dim,
                        channels_out,
                        self.context_dim,
                        n_head,
                        depth,
                    )
                    .init(),
                );

                let rtu = UNetBlocks::ResTU(
                    ResTransformerUpsampleConfig::new(
                        channels_in3,
                        time_embed_dim,
                        channels_out,
                        self.context_dim,
                        n_head,
                        depth,
                    )
                    .init(),
                );

                (rt1, rt2, rtu)
            };

            output_blocks.extend([r1, r2, r3]);
        }

        /*let output_blocks = UNetOutputBlocks {
            rt1: ResTransformerConfig::new(
                8 * self.model_channels,
                time_embed_dim,
                4 * self.model_channels,
                self.context_dim,
                n_head(4 * self.model_channels),
                10,
            )
            .init(),
            rt2: ResTransformerConfig::new(
                8 * self.model_channels,
                time_embed_dim,
                4 * self.model_channels,
                self.context_dim,
                n_head(4 * self.model_channels),
                10,
            )
            .init(),
            rtu1: ResTransformerUpsampleConfig::new(
                6 * self.model_channels,
                time_embed_dim,
                4 * self.model_channels,
                self.context_dim,
                n_head(4 * self.model_channels),
                10,
            )
            .init(),
            rt3: ResTransformerConfig::new(
                6 * self.model_channels,
                time_embed_dim,
                2 * self.model_channels,
                self.context_dim,
                n_head(2 * self.model_channels),
                2,
            )
            .init(),
            rt4: ResTransformerConfig::new(
                4 * self.model_channels,
                time_embed_dim,
                2 * self.model_channels,
                self.context_dim,
                n_head(2 * self.model_channels),
                2,
            )
            .init(),
            rtu2: ResTransformerUpsampleConfig::new(
                3 * self.model_channels,
                time_embed_dim,
                2 * self.model_channels,
                self.context_dim,
                n_head(2 * self.model_channels),
                2,
            )
            .init(),
            r1: ResBlockConfig::new(3 * self.model_channels, time_embed_dim, self.model_channels)
                .init(),
            r2: ResBlockConfig::new(2 * self.model_channels, time_embed_dim, self.model_channels)
                .init(),
            r3: ResBlockConfig::new(2 * self.model_channels, time_embed_dim, self.model_channels)
                .init(),
        };*/

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
        let conv_out = Conv2dConfig::new([self.model_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

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
pub struct UNet<B: MyBackend> {
    model_channels: usize,
    lin1_time_embed: nn::Linear<B>,
    silu_time_embed: SILU,
    lin2_time_embed: nn::Linear<B>,
    lin1_label_embed: nn::Linear<B>,
    silu_label_embed: SILU,
    lin2_label_embed: nn::Linear<B>,
    input_blocks: Vec<UNetBlocks<B>>,
    middle_block: ResTransformerRes<B>,
    output_blocks: Vec<UNetBlocks<B>>,
    norm_out: GroupNorm<B>,
    silu_out: SILU,
    conv_out: Conv2d<B>,
}

impl<B: MyBackend> UNet<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: Tensor<B, 1, Int>,
        context: Tensor<B, 3>,
        label: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
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
        for block in &self.input_blocks {
            x = block.as_ref().forward(x, emb.clone(), context.clone());
            saved_inputs.push(x.clone())
        }

        // middle block
        x = self.middle_block.forward(x, emb.clone(), context.clone());

        // output blocks
        for block in &self.output_blocks {
            x = Tensor::cat(vec![x, saved_inputs.pop().unwrap()], 1);
            x = block.as_ref().forward(x, emb.clone(), context.clone());
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

#[derive(Clone, Debug)]
pub enum UNetBlocks<B: Backend> {
    Conv(Conv2d<B>),
    Res(ResBlock<B>),
    Down(Downsample<B>),
    ResT(ResTransformer<B>),
    ResTU(ResTransformerUpsample<B>),
    ResU(ResUpsample<B>),
}

/*#[derive(Clone, Debug)]
enum UNetBlockInnerModule<B: ADBackend> {
    Conv(<Conv2d<B> as Module<B>>::InnerModule),
    Res(<ResBlock<B> as Module<B>>::InnerModule),
    Down(<Downsample<B> as Module<B>>::InnerModule),
    ResT(<ResTransformer<B> as Module<B>>::InnerModule),
    ResTU(<ResTransformerUpsample<B> as Module<B>>::InnerModule),
    ResU(<ResUpsample<B> as Module<B>>::InnerModule),
}*/

use burn::module::ADModule;
use burn::tensor::backend::ADBackend;

/*impl<B: ADBackend> ADModule<B> for UNetBlocks<B> {
    type InnerModule = UNetBlocks<<B as ADBackend>::InnerBackend>;

    // Required method
    fn valid(&self) -> Self::InnerModule {
        match self {
            UNetBlocks::Conv(b) => UNetBlocks::Conv(b.valid()),
            UNetBlocks::Res(b) => UNetBlocks::Res(b.valid()),
            UNetBlocks::Down(b) => UNetBlocks::Down(b.valid()),
            UNetBlocks::ResT(b) => UNetBlocks::ResT(b.valid()),
            UNetBlocks::ResTU(b) => UNetBlocks::ResTU(b.valid()),
            UNetBlocks::ResU(b) => UNetBlocks::ResU(b.valid()),
        }
    }
}*/

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub enum UNetBlockRecord<B: MyBackend> {
    Conv(<Conv2d<B> as Module<B>>::Record),
    Res(<ResBlock<B> as Module<B>>::Record),
    Down(<Downsample<B> as Module<B>>::Record),
    ResT(<ResTransformer<B> as Module<B>>::Record),
    ResTU(<ResTransformerUpsample<B> as Module<B>>::Record),
    ResU(<ResUpsample<B> as Module<B>>::Record),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub enum UNetBlockRecordItem<B: MyBackend, S: PrecisionSettings> {
    Conv(<<Conv2d<B> as Module<B>>::Record as Record>::Item<S>),
    Res(<<ResBlock<B> as Module<B>>::Record as Record>::Item<S>),
    Down(<<Downsample<B> as Module<B>>::Record as Record>::Item<S>),
    ResT(<<ResTransformer<B> as Module<B>>::Record as Record>::Item<S>),
    ResTU(<<ResTransformerUpsample<B> as Module<B>>::Record as Record>::Item<S>),
    ResU(<<ResUpsample<B> as Module<B>>::Record as Record>::Item<S>),
}

use burn::module::ModuleVisitor;
use burn::record::PrecisionSettings;
use burn::record::Record;

impl<B: MyBackend> Record for UNetBlockRecord<B> {
    type Item<S: PrecisionSettings> = UNetBlockRecordItem<B, S>;

    // Required methods
    fn into_item<S>(self) -> Self::Item<S>
    where
        S: PrecisionSettings,
    {
        match self {
            UNetBlockRecord::Conv(b) => UNetBlockRecordItem::Conv(b.into_item()),
            UNetBlockRecord::Res(b) => UNetBlockRecordItem::Res(b.into_item()),
            UNetBlockRecord::Down(b) => UNetBlockRecordItem::Down(b.into_item()),
            UNetBlockRecord::ResT(b) => UNetBlockRecordItem::ResT(b.into_item()),
            UNetBlockRecord::ResTU(b) => UNetBlockRecordItem::ResTU(b.into_item()),
            UNetBlockRecord::ResU(b) => UNetBlockRecordItem::ResU(b.into_item()),
        }
    }
    fn from_item<S>(item: Self::Item<S>) -> Self
    where
        S: PrecisionSettings,
    {
        match item {
            UNetBlockRecordItem::Conv(b) => {
                UNetBlockRecord::Conv(<Conv2d<B> as Module<B>>::Record::from_item(b))
            }
            UNetBlockRecordItem::Res(b) => {
                UNetBlockRecord::Res(<ResBlock<B> as Module<B>>::Record::from_item(b))
            }
            UNetBlockRecordItem::Down(b) => {
                UNetBlockRecord::Down(<Downsample<B> as Module<B>>::Record::from_item(b))
            }
            UNetBlockRecordItem::ResT(b) => {
                UNetBlockRecord::ResT(<ResTransformer<B> as Module<B>>::Record::from_item(b))
            }
            UNetBlockRecordItem::ResTU(b) => UNetBlockRecord::ResTU(
                <ResTransformerUpsample<B> as Module<B>>::Record::from_item(b),
            ),
            UNetBlockRecordItem::ResU(b) => {
                UNetBlockRecord::ResU(<ResUpsample<B> as Module<B>>::Record::from_item(b))
            }
        }
    }
}

use burn::module::ModuleMapper;

impl<B: MyBackend> Module<B> for UNetBlocks<B> {
    type Record = UNetBlockRecord<B>;

    // Required methods
    fn visit<V>(&self, visitor: &mut V)
    where
        V: ModuleVisitor<B>,
    {
        match self {
            UNetBlocks::Conv(b) => {
                b.visit(visitor);
            }
            UNetBlocks::Res(b) => {
                b.visit(visitor);
            }
            UNetBlocks::Down(b) => {
                b.visit(visitor);
            }
            UNetBlocks::ResT(b) => {
                b.visit(visitor);
            }
            UNetBlocks::ResTU(b) => {
                b.visit(visitor);
            }
            UNetBlocks::ResU(b) => {
                b.visit(visitor);
            }
        };
    }
    fn map<M>(self, mapper: &mut M) -> Self
    where
        M: ModuleMapper<B>,
    {
        match self {
            UNetBlocks::Conv(b) => UNetBlocks::Conv(b.map(mapper)),
            UNetBlocks::Res(b) => UNetBlocks::Res(b.map(mapper)),
            UNetBlocks::Down(b) => UNetBlocks::Down(b.map(mapper)),
            UNetBlocks::ResT(b) => UNetBlocks::ResT(b.map(mapper)),
            UNetBlocks::ResTU(b) => UNetBlocks::ResTU(b.map(mapper)),
            UNetBlocks::ResU(b) => UNetBlocks::ResU(b.map(mapper)),
        }
    }
    fn load_record(self, record: Self::Record) -> Self {
        match self {
            UNetBlocks::Conv(b) => {
                let record = if let UNetBlockRecord::Conv(r) = record {
                    Some(r)
                } else {
                    None
                };
                UNetBlocks::Conv(b.load_record(record.unwrap()))
            }
            UNetBlocks::Res(b) => {
                let record = if let UNetBlockRecord::Res(r) = record {
                    Some(r)
                } else {
                    None
                };
                UNetBlocks::Res(b.load_record(record.unwrap()))
            }
            UNetBlocks::Down(b) => {
                let record = if let UNetBlockRecord::Down(r) = record {
                    Some(r)
                } else {
                    None
                };
                UNetBlocks::Down(b.load_record(record.unwrap()))
            }
            UNetBlocks::ResT(b) => {
                let record = if let UNetBlockRecord::ResT(r) = record {
                    Some(r)
                } else {
                    None
                };
                UNetBlocks::ResT(b.load_record(record.unwrap()))
            }

            UNetBlocks::ResTU(b) => {
                let record = if let UNetBlockRecord::ResTU(r) = record {
                    Some(r)
                } else {
                    None
                };
                UNetBlocks::ResTU(b.load_record(record.unwrap()))
            }

            UNetBlocks::ResU(b) => {
                let record = if let UNetBlockRecord::ResU(r) = record {
                    Some(r)
                } else {
                    None
                };
                UNetBlocks::ResU(b.load_record(record.unwrap()))
            }
        }
    }

    fn into_record(self) -> Self::Record {
        match self {
            UNetBlocks::Conv(b) => UNetBlockRecord::Conv(b.into_record()),
            UNetBlocks::Res(b) => UNetBlockRecord::Res(b.into_record()),
            UNetBlocks::Down(b) => UNetBlockRecord::Down(b.into_record()),
            UNetBlocks::ResT(b) => UNetBlockRecord::ResT(b.into_record()),
            UNetBlocks::ResTU(b) => UNetBlockRecord::ResTU(b.into_record()),
            UNetBlocks::ResU(b) => UNetBlockRecord::ResU(b.into_record()),
        }
    }

    fn devices(&self) -> Vec<<B as Backend>::Device> {
        match self {
            UNetBlocks::Conv(b) => b.devices(),
            UNetBlocks::Res(b) => b.devices(),
            UNetBlocks::Down(b) => b.devices(),
            UNetBlocks::ResT(b) => b.devices(),
            UNetBlocks::ResTU(b) => b.devices(),
            UNetBlocks::ResU(b) => b.devices(),
        }
    }

    fn fork(self, device: &<B as Backend>::Device) -> Self {
        match self {
            UNetBlocks::Conv(b) => UNetBlocks::Conv(b.fork(device)),
            UNetBlocks::Res(b) => UNetBlocks::Res(b.fork(device)),
            UNetBlocks::Down(b) => UNetBlocks::Down(b.fork(device)),
            UNetBlocks::ResT(b) => UNetBlocks::ResT(b.fork(device)),
            UNetBlocks::ResTU(b) => UNetBlocks::ResTU(b.fork(device)),
            UNetBlocks::ResU(b) => UNetBlocks::ResU(b.fork(device)),
        }
    }
    fn to_device(self, device: &<B as Backend>::Device) -> Self {
        match self {
            UNetBlocks::Conv(b) => UNetBlocks::Conv(b.to_device(device)),
            UNetBlocks::Res(b) => UNetBlocks::Res(b.to_device(device)),
            UNetBlocks::Down(b) => UNetBlocks::Down(b.to_device(device)),
            UNetBlocks::ResT(b) => UNetBlocks::ResT(b.to_device(device)),
            UNetBlocks::ResTU(b) => UNetBlocks::ResTU(b.to_device(device)),
            UNetBlocks::ResU(b) => UNetBlocks::ResU(b.to_device(device)),
        }
    }

    fn no_grad(self) -> Self {
        match self {
            UNetBlocks::Conv(b) => UNetBlocks::Conv(b.no_grad()),
            UNetBlocks::Res(b) => UNetBlocks::Res(b.no_grad()),
            UNetBlocks::Down(b) => UNetBlocks::Down(b.no_grad()),
            UNetBlocks::ResT(b) => UNetBlocks::ResT(b.no_grad()),
            UNetBlocks::ResTU(b) => UNetBlocks::ResTU(b.no_grad()),
            UNetBlocks::ResU(b) => UNetBlocks::ResU(b.no_grad()),
        }
    }
    fn num_params(&self) -> usize {
        match self {
            UNetBlocks::Conv(b) => b.num_params(),
            UNetBlocks::Res(b) => b.num_params(),
            UNetBlocks::Down(b) => b.num_params(),
            UNetBlocks::ResT(b) => b.num_params(),
            UNetBlocks::ResTU(b) => b.num_params(),
            UNetBlocks::ResU(b) => b.num_params(),
        }
    }
}

impl<B: MyBackend> UNetBlocks<B> {
    fn as_ref(&self) -> &dyn UNetBlock<B> {
        match self {
            UNetBlocks::Conv(b) => b,
            UNetBlocks::Res(b) => b,
            UNetBlocks::Down(b) => b,
            UNetBlocks::ResT(b) => b,
            UNetBlocks::ResTU(b) => b,
            UNetBlocks::ResU(b) => b,
        }
    }
}

trait UNetBlock<B: Backend> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4>;
}

#[derive(Config, Debug)]
pub struct ResTransformerConfig {
    n_channels_in: usize,
    n_channels_embed: usize,
    n_channels_out: usize,
    n_context_state: usize,
    n_head: usize,
    n_transformer_blocks: usize,
}

impl ResTransformerConfig {
    fn init<B: MyBackend>(&self) -> ResTransformer<B> {
        let res = ResBlockConfig::new(
            self.n_channels_in,
            self.n_channels_embed,
            self.n_channels_out,
        )
        .init();
        let transformer = SpatialTransformerConfig::new(
            self.n_channels_out,
            self.n_context_state,
            self.n_head,
            self.n_transformer_blocks,
        )
        .init();

        ResTransformer { res, transformer }
    }
}

#[derive(Module, Debug)]
pub struct ResTransformer<B: MyBackend> {
    res: ResBlock<B>,
    transformer: SpatialTransformer<B>,
}

impl<B: MyBackend> UNetBlock<B> for ResTransformer<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res.forward(x, emb);
        let x = self.transformer.forward(x, context);
        x
    }
}

#[derive(Config, Debug)]
pub struct ResUpsampleConfig {
    n_channels_in: usize,
    n_channels_embed: usize,
    n_channels_out: usize,
}

impl ResUpsampleConfig {
    fn init<B: Backend>(&self) -> ResUpsample<B> {
        let res = ResBlockConfig::new(
            self.n_channels_in,
            self.n_channels_embed,
            self.n_channels_out,
        )
        .init();

        let upsample = UpsampleConfig::new(self.n_channels_out).init();

        ResUpsample { res, upsample }
    }
}

#[derive(Module, Debug)]
pub struct ResUpsample<B: Backend> {
    res: ResBlock<B>,
    upsample: Upsample<B>,
}

impl<B: Backend> UNetBlock<B> for ResUpsample<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res.forward(x, emb);
        let x = self.upsample.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct ResTransformerUpsampleConfig {
    n_channels_in: usize,
    n_channels_embed: usize,
    n_channels_out: usize,
    n_context_state: usize,
    n_head: usize,
    n_transformer_blocks: usize,
}

impl ResTransformerUpsampleConfig {
    fn init<B: MyBackend>(&self) -> ResTransformerUpsample<B> {
        let res = ResBlockConfig::new(
            self.n_channels_in,
            self.n_channels_embed,
            self.n_channels_out,
        )
        .init();
        let transformer = SpatialTransformerConfig::new(
            self.n_channels_out,
            self.n_context_state,
            self.n_head,
            self.n_transformer_blocks,
        )
        .init();
        let upsample = UpsampleConfig::new(self.n_channels_out).init();

        ResTransformerUpsample {
            res,
            transformer,
            upsample,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResTransformerUpsample<B: MyBackend> {
    res: ResBlock<B>,
    transformer: SpatialTransformer<B>,
    upsample: Upsample<B>,
}

impl<B: MyBackend> UNetBlock<B> for ResTransformerUpsample<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res.forward(x, emb);
        let x = self.transformer.forward(x, context);
        let x = self.upsample.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct ResTransformerResConfig {
    n_channels_in: usize,
    n_channels_embed: usize,
    n_channels_out: usize,
    n_context_state: usize,
    n_head: usize,
    n_transformer_blocks: usize,
}

impl ResTransformerResConfig {
    fn init<B: MyBackend>(&self) -> ResTransformerRes<B> {
        let res1 = ResBlockConfig::new(
            self.n_channels_in,
            self.n_channels_embed,
            self.n_channels_out,
        )
        .init();
        let transformer = SpatialTransformerConfig::new(
            self.n_channels_out,
            self.n_context_state,
            self.n_head,
            self.n_transformer_blocks,
        )
        .init();
        let res2 = ResBlockConfig::new(
            self.n_channels_in,
            self.n_channels_embed,
            self.n_channels_out,
        )
        .init();

        ResTransformerRes {
            res1,
            transformer,
            res2,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResTransformerRes<B: MyBackend> {
    res1: ResBlock<B>,
    transformer: SpatialTransformer<B>,
    res2: ResBlock<B>,
}

impl<B: MyBackend> UNetBlock<B> for ResTransformerRes<B> {
    fn forward(&self, x: Tensor<B, 4>, emb: Tensor<B, 2>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let x = self.res1.forward(x, emb.clone());
        let x = self.transformer.forward(x, context);
        let x = self.res2.forward(x, emb);
        x
    }
}

#[derive(Config, Debug)]
pub struct UpsampleConfig {
    n_channels: usize,
}

impl UpsampleConfig {
    fn init<B: Backend>(&self) -> Upsample<B> {
        let conv = Conv2dConfig::new([self.n_channels, self.n_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        Upsample { conv }
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

#[derive(Config, Debug)]
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

#[derive(Config, Debug)]
pub struct SpatialTransformerConfig {
    n_channels: usize,
    n_context_state: usize,
    n_head: usize,
    n_blocks: usize,
}

impl SpatialTransformerConfig {
    fn init<B: MyBackend>(&self) -> SpatialTransformer<B> {
        let norm = GroupNormConfig::new(32, self.n_channels).init();
        let proj_in = nn::LinearConfig::new(self.n_channels, self.n_channels).init(); //Conv2dConfig::new([self.n_channels, self.n_channels], [1, 1]).init();
        let blocks = (0..self.n_blocks)
            .into_iter()
            .map(|_| {
                TransformerBlockConfig::new(self.n_channels, self.n_context_state, self.n_head)
                    .init()
            })
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
pub struct SpatialTransformer<B: MyBackend> {
    norm: GroupNorm<B>,
    proj_in: nn::Linear<B>,
    blocks: Vec<TransformerBlock<B>>,
    proj_out: nn::Linear<B>,
}

impl<B: MyBackend> SpatialTransformer<B> {
    fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let [n_batch, n_channel, height, width] = x.dims();

        let x_in = x.clone();

        let x = self.norm.forward(x);
        let x = x
            .reshape([n_batch, n_channel, height * width])
            .swap_dims(1, 2);
        let x = self.proj_in.forward(x);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, context.clone());
        }

        let x = self
            .proj_out
            .forward(x)
            .swap_dims(1, 2)
            .reshape([n_batch, n_channel, height, width]);

        x_in + x
    }
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    n_state: usize,
    n_context_state: usize,
    n_head: usize,
}

impl TransformerBlockConfig {
    fn init<B: MyBackend>(&self) -> TransformerBlock<B> {
        let norm1 = LayerNormConfig::new(self.n_state).init();
        let attn1 = MultiHeadAttentionConfig::new(self.n_state, self.n_state, self.n_head).init();
        let norm2 = LayerNormConfig::new(self.n_state).init();
        let attn2 =
            MultiHeadAttentionConfig::new(self.n_state, self.n_context_state, self.n_head).init();
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
pub struct TransformerBlock<B: MyBackend> {
    norm1: LayerNorm<B>,
    attn1: MultiHeadAttention<B>,
    norm2: LayerNorm<B>,
    attn2: MultiHeadAttention<B>,
    norm3: LayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: MyBackend> TransformerBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn1.forward(self.norm1.forward(x), None);
        let x = x.clone() + self.attn2.forward(self.norm2.forward(x), Some(context));
        x.clone() + self.mlp.forward(self.norm3.forward(x))
    }
}

#[derive(Config, Debug)]
pub struct MLPConfig {
    n_state: usize,
    mult: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self) -> MLP<B> {
        let n_state_hidden = self.n_state * self.mult;
        let geglu = GEGLUConfig::new(self.n_state, n_state_hidden).init();
        let lin = nn::LinearConfig::new(n_state_hidden, self.n_state).init();

        MLP { geglu, lin }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    geglu: GEGLU<B>,
    lin: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.lin.forward(self.geglu.forward(x))
    }
}

#[derive(Config, Debug)]
pub struct GEGLUConfig {
    n_state_in: usize,
    n_state_out: usize,
}

impl GEGLUConfig {
    fn init<B: Backend>(&self) -> GEGLU<B> {
        let proj = nn::LinearConfig::new(self.n_state_in, 2 * self.n_state_out).init();
        let gelu = GELU::new();

        GEGLU { proj, gelu }
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

        let x = projected
            .clone()
            .slice([0..n_batch, 0..n_ctx, 0..n_state_out]);
        let gate = projected.slice([0..n_batch, 0..n_ctx, n_state_out..n_state]);

        x * self.gelu.forward(gate)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    n_state: usize,
    n_context_state: usize,
    n_head: usize,
}

impl MultiHeadAttentionConfig {
    fn init<B: MyBackend>(&self) -> MultiHeadAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init();
        let key = nn::LinearConfig::new(self.n_context_state, self.n_state)
            .with_bias(false)
            .init();
        let value = nn::LinearConfig::new(self.n_context_state, self.n_state)
            .with_bias(false)
            .init();
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init();

        MultiHeadAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: MyBackend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
}

impl<B: MyBackend> MultiHeadAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, context: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let xa = context.unwrap_or_else(|| x.clone());

        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);

        let wv = Tensor::from_primitive(B::qkv_attention(q.into_primitive(), k.into_primitive(), v.into_primitive(), None, self.n_head));

        self.out.forward(wv)
    }
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    n_channels_in: usize,
    n_channels_embed: usize,
    n_channels_out: usize,
}

impl ResBlockConfig {
    fn init<B: Backend>(&self) -> ResBlock<B> {
        let norm_in = GroupNormConfig::new(32, self.n_channels_in).init();
        let silu_in = SILU::new();
        let conv_in = Conv2dConfig::new([self.n_channels_in, self.n_channels_out], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        let silu_embed = SILU::new();
        let lin_embed = nn::LinearConfig::new(self.n_channels_embed, self.n_channels_out).init();

        let norm_out = GroupNormConfig::new(32, self.n_channels_out).init();
        let silu_out = SILU::new();
        let conv_out = Conv2dConfig::new([self.n_channels_out, self.n_channels_out], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        let skip_connection = if self.n_channels_in != self.n_channels_out {
            Some(Conv2dConfig::new([self.n_channels_in, self.n_channels_out], [1, 1]).init())
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
