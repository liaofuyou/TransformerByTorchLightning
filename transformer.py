import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable

from data_module import Dimension


class TokenEmbedding(nn.Module):

    def __init__(self, dim: Dimension):
        super(TokenEmbedding, self).__init__()
        self.dim = dim
        self.token_embedding = nn.Embedding(dim.vocab, dim.token)

    def forward(self, input_tensor):
        return self.lut(input_tensor) * math.sqrt(self.dim.token)


class PositionalEncoding(nn.Module):

    def __init__(self, dim: Dimension):
        super(PositionalEncoding, self).__init__()
        # Dim: [32, 300]
        encoding = torch.zeros(dim.sents, dim.token)
        # Dim: [32, 1]
        position = torch.arange(0, dim.sents).unsqueeze(1)
        # Dim: [1, 150]
        div_term = torch.exp(torch.arange(0, dim.token, 2) * -(math.log(10000.0) / dim.token))
        # SIN FUCKING COS
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        # Dim: [1, 32, 300]
        self.encoding = encoding.unsqueeze(0)
        # BUFFER
        self.register_buffer('pe', encoding)

    def forward(self, input_tensor):
        return input_tensor + Variable(self.encoding, requires_grad=False)


class DecoderFinalLayer(nn.Module):

    def __init__(self, dim: Dimension):
        super(DecoderFinalLayer, self).__init__()
        self.project = nn.Linear(dim.token, dim.vocab)

    def forward(self, x):
        return f.log_softmax(self.project(x), dim=-1)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, head_count, dim: Dimension):
        super(MultiHeadSelfAttention, self).__init__()
        self.head_count = head_count
        self.dim = dim
        self.slimer_and_taller_token_dim = dim.token // self.head_count
        self.query_layer = nn.Linear(dim.token, dim.token)
        self.key_layer = nn.Linear(dim.token, dim.token)
        self.value_layer = nn.Linear(dim.token, dim.token)

    def forward(self, input_tensor, encoder_output_tensor=None, mask=None) -> torch.Tensor:
        # 一气化三清
        queries = self.query_layer(input_tensor)
        keys = self.key_layer(input_tensor if encoder_output_tensor is None else encoder_output_tensor)
        values = self.value_layer(input_tensor if encoder_output_tensor is None else encoder_output_tensor)
        # 女生梦想，变高变瘦
        slimer_and_taller_queries = self.transpose_for_scores(queries)
        slimer_and_taller_keys = self.transpose_for_scores(keys)
        slimer_and_taller_values = self.transpose_for_scores(values)
        # Do attention
        output_tensor = self.self_attention(slimer_and_taller_queries, slimer_and_taller_keys, slimer_and_taller_values,
                                            mask=mask)
        # 变回来
        return self.transpose_back(output_tensor)

    def transpose_for_scores(self, input_tensor) -> torch.Tensor:
        batch_size = len(input_tensor)
        return input_tensor.view(batch_size, -1, self.head_count, self.slimer_and_taller_token_dim).transpose(1, 2)

    def transpose_back(self, input_tensor) -> torch.Tensor:
        batch_size = len(input_tensor)
        return input_tensor.transpose(1, 2).contiguous().view(batch_size, -1, self.dim.token)

    @staticmethod
    def self_attention(queries, keys, values, mask=None) -> torch.Tensor:
        token_dim = keys.size(-1)
        keys = keys.transpose(-2, -1)
        scores = torch.matmul(queries, keys) / math.sqrt(token_dim)
        # MASK
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        # SOFTMAX
        attention = f.softmax(scores, dim=-1)
        return torch.matmul(attention, values)


class FeedForwardLayer(nn.Module):

    def __init__(self, dim: Dimension):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(dim.token, 1000)  # 1000 ? what-fucking-ever
        self.fc2 = nn.Linear(1000, dim.token)

    def forward(self, x):
        return self.fc2(f.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, dim: Dimension):
        super(EncoderLayer, self).__init__()
        self.head_count = 3
        self.self_attention = MultiHeadSelfAttention(self.head_count, dim)
        self.feed_forward_layer = FeedForwardLayer(dim)
        self.layer_norm = nn.LayerNorm(dim.token)

    def forward(self, input_tensor):
        # Attention
        attention_data = self.self_attention(input_tensor)
        # Add & Norm
        residual_data = self.layer_norm(attention_data + input_tensor)
        # Feed Forward
        feed_forward_data = self.feed_forward_layer(residual_data)
        # Add & Norm Again
        return self.layer_norm(residual_data + feed_forward_data)


class DecoderLayer(nn.Module):

    def __init__(self, dim: Dimension):
        super(DecoderLayer, self).__init__()
        self.head_count = 3
        self.dim = dim
        self.self_attention = MultiHeadSelfAttention(self.head_count, dim)
        self.self_attention2 = MultiHeadSelfAttention(self.head_count, dim)
        self.feed_forward_layer = FeedForwardLayer(dim)
        self.decoder_attention_mask = self.decoder_attention_mask()
        self.layer_norm = nn.LayerNorm(dim.token)

    def forward(self, input_tensor, encoder_output_tensor):
        # Attention 1
        attention_data = self.self_attention(input_tensor, mask=self.decoder_attention_mask)
        # Add & Norm
        input_tensor = self.layer_norm(attention_data + input_tensor)
        # Attention 2
        attention_data = self.self_attention2(input_tensor, encoder_output_tensor)
        # Add & Norm
        residual_data = self.layer_norm(attention_data + input_tensor)
        # Feed Forward
        feed_forward_data = self.feed_forward_layer(residual_data)
        # Add & Norm Again
        return self.layer_norm(residual_data + feed_forward_data)

    def decoder_attention_mask(self):
        attn_shape = [self.dim.batch, self.head_count, self.dim.sents, self.dim.sents]
        decoder_attention_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
        decoder_attention_mask = torch.from_numpy(decoder_attention_mask)  # .byte()
        return decoder_attention_mask  # [batch_size, head_count, sents_len, sents_len]


class Encoder(nn.Module):

    def __init__(self, dim: Dimension, encoder_layer_num=1):
        super(Encoder, self).__init__()
        self.encoder_layer_num = encoder_layer_num
        self.encoder_layers = nn.ModuleList([EncoderLayer(dim) for _ in range(encoder_layer_num)])

    def forward(self, input_tensor):
        # Encoder Layers
        for layer in self.encoder_layers:
            input_tensor = layer(input_tensor)
        # 你要的
        return input_tensor


class Decoder(nn.Module):

    def __init__(self, dim, decoder_layer_num=1):
        super(Decoder, self).__init__()
        self.decoder_layer_num = decoder_layer_num
        self.decoder_layers = nn.ModuleList([DecoderLayer(dim) for _ in range(decoder_layer_num)])
        self.final_layer = DecoderFinalLayer(dim)

    def forward(self, input_tensor, encoder_output_tensor):
        # Decoder Layers
        for layer in self.decoder_layers:
            input_tensor = layer(input_tensor, encoder_output_tensor)
        # 你要的
        return self.final_layer(input_tensor)


class Transformer(pl.LightningModule):

    def __init__(self, dim=Dimension()):
        super(Transformer, self).__init__()
        # Embedding
        self.token_embedding = nn.Embedding(dim.vocab, dim.token)
        self.positional_encoding = PositionalEncoding(dim)
        # Encoder & Decoder
        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)
        self.dim = dim

    def forward(self, source_data, target_data) -> torch.Tensor:
        # 编码 & 位置编码
        encoder_input_tensor = self.token_embedding(source_data)
        encoder_input_tensor = self.positional_encoding(encoder_input_tensor)
        # 编码 &位置编码
        decoder_input_tensor = self.token_embedding(target_data)
        decoder_input_tensor = self.positional_encoding(decoder_input_tensor)
        # 算吧
        encoder_output_tensor = self.encoder(encoder_input_tensor)
        decoder_output_tensor = self.decoder(decoder_input_tensor, encoder_output_tensor)
        # 结果
        target_outputs = decoder_output_tensor
        # 你要的
        # return target_outputs
        return target_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):

        source = batch[0]
        target = batch[1]

        outputs = self(source, target)

        loss = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss(outputs.contiguous().view(-1, outputs.size(-1)), target.view(-1))
        return loss
