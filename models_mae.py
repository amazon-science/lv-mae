# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # Expand attn_mask to match the shape of (B, num_heads, N, N)
            attn_mask = attn_mask[:, None, None, :]  # (B, 1, 1, N)
            attn = attn.masked_fill(attn_mask, float('-inf'))  # Fill masked positions with -inf

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTrans   former backbone
    """

    def __init__(self, args=None, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.args = args
        self.num_patches = self.args.num_patches
        self.similarity_loss = self.args.cos_loss

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches,
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.num_patches ** .5), cls_token=True)
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # Function to initialize parameters to zero
    def initialize_parameters_to_zero(self):
        for param in self.parameters():
            if param.requires_grad:  # Check if the parameter is trainable
                nn.init.constant_(param, 0)  # Set parameter to zero

    def forward_encoder(self, x, attn_mask, ids_shuffle=None, length=None):

        B, N, C = x.shape
        pos_embed_shuffled_batch = []

        if ids_shuffle is not None:
            for i in range(B):
                # Extract the positional embeddings and shuffle them according to ids_shuffle
                pos_embed_shuffled = torch.gather(
                    self.pos_embed[:, 1:length[i] + 1, :],  # take only the relevant length
                    dim=1,
                    index=ids_shuffle[i][:length[i]].unsqueeze(-1).repeat(1, 1, self.pos_embed.shape[-1]).to(dtype=torch.int64)
                )

                # Pad the shuffled positional embeddings to match the length of x
                padding_length = N - length[i]
                if padding_length > 0:
                    padding = torch.zeros(padding_length, C, device=x.device)
                    pos_embed_shuffled = torch.cat((pos_embed_shuffled, padding[None]), dim=1)

                pos_embed_shuffled_batch.append(pos_embed_shuffled.squeeze())

            # Stack the sequences back into a batch
            pos_embed_shuffled_batch = torch.stack(pos_embed_shuffled_batch, dim=0)

            # Add positional embeddings to the input
            x = x + pos_embed_shuffled_batch

        else:
            x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, ids_restore, masked_length, length, mask):
        # embed tokens
        x = self.decoder_embed(x)
        B, N, C = x.shape

        decoded_sequences = []
        attn_masks = []

        for i in range(B):
            L_full = length[i]  # Full length (including masked and unmasked tokens)

            # Determine the number of mask tokens needed for this sequence
            mask_tokens_needed = L_full - masked_length[i].item()

            mask_tokens = self.mask_token.repeat(1, mask_tokens_needed, 1)

            # Remove CLS token and prepare the sequence for restoration
            x_seq = x[i, 1:masked_length[i] + 1, :][None]

            # Concatenate the unmasked sequence with the mask tokens
            x_full = torch.cat([x_seq, mask_tokens], dim=1)

            # Restore the original sequence order using ids_restore
            x_restored = torch.gather(x_full, dim=1, index=ids_restore[i][:L_full].unsqueeze(0).unsqueeze(-1).repeat(1, 1, C).to(dtype=torch.int64))

            # Append the CLS token back to the restored sequence
            x_restored = torch.cat([x[i, :1, :][None], x_restored], dim=1)

            x_restored_pos_embed = x_restored + self.decoder_pos_embed[:, :L_full+1]

            # Pad the shuffled positional embeddings to match the length of x
            padding_length = N - (length[i] + 1)
            if padding_length > 0:
                padding = torch.zeros(padding_length, C, device=x.device)
                x_restored_pos_embed_padded = torch.cat((x_restored_pos_embed, padding[None]), dim=1)
            else:
                x_restored_pos_embed_padded = x_restored_pos_embed

            attn_mask = torch.cat((torch.zeros(L_full+1, dtype=torch.bool),
                                   torch.ones(padding_length, dtype=torch.bool)), dim=0)[None].to(device=x.device)

            decoded_sequences.append(x_restored_pos_embed_padded)
            attn_masks.append(attn_mask)

        # Stack all sequences back into a batch
        x = torch.cat(decoded_sequences, dim=0)
        attn_mask = torch.cat(attn_masks, dim=0)

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove CLS token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, length, temperature=0.07):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs.clone()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        if self.similarity_loss:
            loss = 1 - self.cos_loss(pred, target)
        else:
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, masked_padded_embeddings, ids_restore, ids_shuffle, masked_length, normalized_stacked_embeddings,
                length, mask, attn_mask=None, mask_ratio=0.75):
        latent = self.forward_encoder(masked_padded_embeddings, attn_mask, ids_shuffle=ids_shuffle, length=length)
        pred = self.forward_decoder(latent, ids_restore, masked_length, length, mask)  # [N, L, p*p*3]
        loss = self.forward_loss(normalized_stacked_embeddings, pred, mask, length)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_small_patch192_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=kwargs['args'].num_patches, embed_dim=kwargs['args'].input_size, depth=kwargs['args'].depth, num_heads=kwargs['args'].num_heads,
        decoder_embed_dim=kwargs['args'].decoder_embed_dim, decoder_depth=kwargs['args'].decoder_depth, decoder_num_heads=kwargs['args'].decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_custom = mae_vit_small_patch192_dec512d8b  # decoder: 512 dim, 8 blocks