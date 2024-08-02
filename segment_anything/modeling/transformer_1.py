import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens

        # print('attn.shape',attn.shape)

        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    

class MLPBlock_low_rank(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class parallel_fused_AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        # self-attention for each other
        self.self_attn_image = Attention(embedding_dim, num_heads)
        self.norm_image_self = nn.LayerNorm(embedding_dim)

        self.self_attn_lang = Attention(embedding_dim, num_heads)
        self.norm_lang_self = nn.LayerNorm(embedding_dim)

        # cross-attention for each other
        self.cross_image_to_lang = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_image_to_lang_norm = nn.LayerNorm(embedding_dim)

        self.cross_lang_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_lang_to_image_norm = nn.LayerNorm(embedding_dim)

        # mlp for image and lang
        # each_dim = mlp_dim / 2
        self.image_mlp = MLPBlock(embedding_dim, mlp_dim , activation)
        self.image_mlp_norm = nn.LayerNorm(embedding_dim)

        # self.lang_mlp = MLPBlock(embedding_dim, mlp_dim , activation)
        # self.lang_mlp_norm = nn.LayerNorm(embedding_dim)

        # last cross for image
        self.cross_lang_to_image_last = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_lang_to_image_last_norm = nn.LayerNorm(embedding_dim)

        self.cross_image_to_lang_last = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_image_to_lang_last_norm = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, image: Tensor, lang: Tensor, imagepe: Tensor, langpe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # self attention for each other
        if self.skip_first_layer_pe:
            lang = self.self_attn_lang(q=lang, k=lang, v=lang)
        else:
            lang_pe = lang + langpe
            lang_attn_out = self.self_attn_lang(q=lang_pe, k=lang_pe, v=lang)
            lang = lang + lang_attn_out
        lang = self.norm_lang_self(lang)

        if self.skip_first_layer_pe:
            image = self.self_attn_image(q=image, k=image, v=image)
        else:
            image_pe = image + imagepe.to(image.device)
            image_attn_out = self.self_attn_image(q=image_pe, k=image_pe, v=image)
            image = image + image_attn_out
        image = self.norm_image_self(image)

        # cross attention for each other
        image_pe = image + imagepe.to(image.device)
        lang_pe = lang + langpe

        attn_out = self.cross_image_to_lang(q=image_pe, k=lang_pe, v=lang)
        image = image + attn_out
        image = self.cross_image_to_lang_norm(image)

        attn_out = self.cross_lang_to_image(q=lang_pe, k=image_pe, v=image)
        lang = lang + attn_out
        lang = self.cross_lang_to_image_norm(lang)

        # mlp block for each other
        mlp_out = self.image_mlp(image)
        image = image + mlp_out
        image = self.image_mlp_norm(image)

        # mlp_out = self.lang_mlp(lang)
        # lang = lang + mlp_out
        # lang = self.lang_mlp_norm(lang)

        # last cross-attention for image
        image_pe = image + imagepe.to(image.device)
        lang_pe = lang + langpe

        attn_out = self.cross_lang_to_image_last(q=lang_pe, k=image_pe, v=image)
        lang = lang + attn_out
        lang = self.cross_lang_to_image_last_norm(lang)

        attn_out = self.cross_image_to_lang_last(q=image_pe, k=lang_pe, v=lang)
        image = image + attn_out
        image = self.cross_image_to_lang_last_norm(image)

        return image, lang
    

class fused_AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()

        low_rank_mlp_dim = 16
        # self-attention for each other
        self.self_attn_lang = Attention(embedding_dim, num_heads)
        self.norm_lang_self = nn.LayerNorm(embedding_dim)

        # cross-attention for each other
        self.cross_image_to_lang = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_image_to_lang_norm = nn.LayerNorm(embedding_dim)

        self.cross_lang_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.cross_lang_to_image_norm = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, image: Tensor, lang: Tensor, imagepe: Tensor, langpe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            lang = self.self_attn_lang(q=lang, k=lang, v=lang)
        else:
            lang_pe = lang + langpe
            lang_attn_out = self.self_attn_lang(q=lang_pe, k=lang_pe, v=lang)
            lang = lang + lang_attn_out
        lang = self.norm_lang_self(lang)

        # cross attention for each other
        image_pe = image + imagepe.to(image.device)
        lang_pe = lang + langpe

        attn_out = self.cross_lang_to_image(q=lang_pe, k=image_pe, v=image)
        lang = lang + attn_out
        lang = self.cross_lang_to_image_norm(lang)

        # last cross-attention for image
        image_pe = image + imagepe.to(image.device)
        lang_pe = lang + langpe

        attn_out = self.cross_image_to_lang(q=image_pe, k=lang_pe, v=lang)
        image = image + attn_out
        image = self.cross_image_to_lang_norm(image)

        return image, lang




class parallel_fused_Attention(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                fused_AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        lang = point_embedding
        image = image_embedding

        for layer in self.layers:
            image, lang = layer(
                image=image,
                lang=lang,
                imagepe=image_pe,
                langpe=point_embedding,
            )

        lang_pe = lang + point_embedding
        image_pe = image + image_pe.to(image.device)
        attn_out = self.final_attn_token_to_image(q=lang_pe, k=image_pe, v=image)
        lang = lang + attn_out
        lang = self.norm_final_attn(lang)

        return lang, image
