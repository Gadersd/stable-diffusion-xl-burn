import pathlib
import save
from save import *

from torch import nn

def save_clipmlp(clip_mlp, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(clip_mlp.fc1, pathlib.Path(path, 'fc1'))
    save_linear(clip_mlp.fc2, pathlib.Path(path, 'fc2'))

def save_clip_attention(clip_attention, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(clip_attention.k_proj, pathlib.Path(path, 'key'))
    save_linear(clip_attention.v_proj, pathlib.Path(path, 'value'))
    save_linear(clip_attention.q_proj, pathlib.Path(path, 'query'))
    save_linear(clip_attention.out_proj, pathlib.Path(path, 'out'))
    save_scalar(clip_attention.num_heads, 'n_head', path)

def save_clip_encoder_layer(clip_encoder_layer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_clip_attention(clip_encoder_layer.self_attn, pathlib.Path(path, 'attn'))
    save_layer_norm(clip_encoder_layer.layer_norm1, pathlib.Path(path, 'attn_ln'))
    save_clipmlp(clip_encoder_layer.mlp, pathlib.Path(path, 'mlp'))
    save_layer_norm(clip_encoder_layer.layer_norm2, pathlib.Path(path, 'mlp_ln'))

def save_clip_encoder(clip_encoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(clip_encoder.layers):
        save_clip_encoder_layer(layer, pathlib.Path(path, f'blocks/{i}'))
    save_scalar(len(clip_encoder.layers), "n_layer", path)

def save_clip_text_embeddings(clip_text_embeddings, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_embedding(clip_text_embeddings.token_embedding, pathlib.Path(path, 'token_embedding'))
    save_embedding(clip_text_embeddings.position_embedding, pathlib.Path(path, 'position_embedding'))

def save_clip_text_transformer(clip_text_transformer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        save_clip_text_embeddings(clip_text_transformer.embeddings, path)
        save_clip_encoder(clip_text_transformer.encoder, path)
        save_layer_norm(clip_text_transformer.final_layer_norm, pathlib.Path(path, 'layer_norm'))
        if hasattr(clip_text_transformer, 'text_projection') and clip_text_transformer.text_projection is not None:
            save_tensor(clip_text_transformer.text_projection, 'text_projection', path)




def save_open_clip_attention(clip_attention, path):
    # Extract original parameters
    in_proj_weight = clip_attention.in_proj_weight
    in_proj_bias = clip_attention.in_proj_bias
    emb_dim = in_proj_weight.shape[1]  # get the emb_dim using weight tensor's shape

    # Split the weight and bias
    q_weight, k_weight, v_weight = torch.split(in_proj_weight, [emb_dim, emb_dim, emb_dim])
    q_bias, k_bias, v_bias = torch.split(in_proj_bias, [emb_dim, emb_dim, emb_dim])

    # Set the extracted weights and biases to corresponding Linear modules
    clip_attention.q_proj = nn.Linear(emb_dim, emb_dim, bias=True)
    clip_attention.k_proj = nn.Linear(emb_dim, emb_dim, bias=True)
    clip_attention.v_proj = nn.Linear(emb_dim, emb_dim, bias=True)

    clip_attention.q_proj.weight.data.copy_(q_weight)
    clip_attention.q_proj.bias.data.copy_(q_bias)
    
    clip_attention.k_proj.weight.data.copy_(k_weight)
    clip_attention.k_proj.bias.data.copy_(k_bias)

    clip_attention.v_proj.weight.data.copy_(v_weight)
    clip_attention.v_proj.bias.data.copy_(v_bias)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(clip_attention.k_proj, pathlib.Path(path, 'key'))
    save_linear(clip_attention.v_proj, pathlib.Path(path, 'value'))
    save_linear(clip_attention.q_proj, pathlib.Path(path, 'query'))
    save_linear(clip_attention.out_proj, pathlib.Path(path, 'out'))
    save_scalar(clip_attention.num_heads, 'n_head', path)

def save_open_clipmlp(clip_mlp, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_linear(clip_mlp[0], pathlib.Path(path, 'fc1'))
    save_linear(clip_mlp[2], pathlib.Path(path, 'fc2'))

def save_open_clip_encoder_layer(clip_encoder_layer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    save_open_clip_attention(clip_encoder_layer.attn, pathlib.Path(path, 'attn'))
    save_layer_norm(clip_encoder_layer.ln_1, pathlib.Path(path, 'attn_ln'))
    save_open_clipmlp(clip_encoder_layer.mlp, pathlib.Path(path, 'mlp'))
    save_layer_norm(clip_encoder_layer.ln_2, pathlib.Path(path, 'mlp_ln'))

def save_open_clip_encoder(clip_encoder, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for i, layer in enumerate(clip_encoder.resblocks):
        save_open_clip_encoder_layer(layer, pathlib.Path(path, f'blocks/{i}'))
    save_scalar(len(clip_encoder.resblocks), "n_layer", path)

def save_open_clip_text_transformer(clip_text_transformer, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        save_embedding(clip_text_transformer.token_embedding, pathlib.Path(path, 'token_embedding'))
        save_tensor(clip_text_transformer.positional_embedding, 'weight', pathlib.Path(path, 'position_embedding'))
        save_open_clip_encoder(clip_text_transformer.transformer, path)
        save_layer_norm(clip_text_transformer.ln_final, pathlib.Path(path, 'layer_norm'))
        if clip_text_transformer.text_projection is not None:
            save_tensor(clip_text_transformer.text_projection, 'text_projection', path)
