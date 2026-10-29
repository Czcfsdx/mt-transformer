# Reference: https://nlp.seas.harvard.edu/annotated-transformer

import math

import torch
from torch import nn
from torch.nn.functional import log_softmax, relu

# Positional encoding
class PositionalEncoding(nn.Module):
    """
    PE_(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE_(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` has shape (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# Multi-head attention
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention block"""

    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model: int = d_model
        self.n_head: int = n_head
        self.d_k: int = d_model // n_head

        # bundle Q/K/V projections into a single linear layer for speed
        self.w_q: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.w_k: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.w_v: nn.Linear = nn.Linear(d_model, d_model, bias=False)
        self.w_o: nn.Linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections in batch: (B, T, D) -> (B, n_head, T, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # O = softmax(K^T * Q / (sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)

        # "Concat" using a view and apply a final linear.
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        del query
        del key
        del value
        return self.w_o(context)


# Position-wise feed-forward network
class PositionwiseFFN(nn.Module):
    """
    Feed Forward block

    FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(relu(self.w_1(x))))


# Encoder / Decoder Layer
class EncoderLayer(nn.Module):
    """Encoder block: Multi-Head Attention + Feed Forward."""

    def __init__(
            self, d_model: int, n_head: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention
        attn_out = self.attn(x, x, x, src_mask)

        # Add & Norm
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)

        # Add & Norm
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class DecoderLayer(nn.Module):
    """Decoder block: Masked Multi-Head Attention + Cross Multi-Head Attention + Feed Forward."""

    def __init__(
            self, d_model: int, n_head: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(d_model=d_model, n_head=n_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Masked Multi-Head Attention
        self_attn_out = self.self_attn(x, x, x, tgt_mask)

        # Add & Norm
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Cross Mutil-Head Attention
        # key & value from encoder output, query from decoder
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)

        # Add & Norm
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)

        # Add & Norm
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


# Encoder / Decoder
class Encoder(nn.Module):
    """Encoder: Input Embedding + Position Encoding + N * Encoder Layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Input Embedding (scaled by sqrt(d_model))
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        # Position Encoding
        x = self.pos_encoding(x)
        # N * Encoder Layers
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    """Decoder: Input Embedding + Position Encoding + N * Decoder Layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Input Embedding (scaled by sqrt(d_model))
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        # Position Encoding
        x = self.pos_encoding(x)
        # N * Decoder Layers
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# Generator
class Generator(nn.Module):
    """Generator block: Linear + Softmax"""

    def __init__(self, d_model: int, embedding_weight: torch.Tensor) -> None:
        super().__init__()
        vocab_size, _ = embedding_weight.shape
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        self.proj.weight = embedding_weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax(self.proj(x), dim=-1)


# Transformer
class Transformer(nn.Module):
    """Encoder-Decoder Transformer."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.generator = Generator(d_model=d_model, embedding_weight=self.decoder.embedding.weight)

        # Xavier-style initialisation
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt_in, memory, src_mask, tgt_mask)
        return self.generator(decoded)
