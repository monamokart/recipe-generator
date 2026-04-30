import torch
import torch.nn as nn


class RecipeTransformer(nn.Module):
    def __init__(
        self,
        config: dict,
        n_ingr: int,
        n_cat: int,
        n_cuisine: int,
        n_recipe_cat: int,
        num_numerical: int,
        ingr_cat_buf: torch.Tensor,
        ingr_num_buf: torch.Tensor,
        pad_idx: int,
        cat_pad_idx: int = 0,
    ) -> None:
        super().__init__()

        d_ingr = config["d_ingr_emb"]
        d_cat = config["d_cat_emb"]
        d_model = config["d_model"]

        self.register_buffer("ingr_cat_buf", ingr_cat_buf)
        self.register_buffer("ingr_num_buf", ingr_num_buf)

        self.ingr_emb = nn.Embedding(n_ingr, d_ingr, padding_idx=pad_idx)
        self.cat_emb = nn.Embedding(n_cat, d_cat, padding_idx=cat_pad_idx)

        in_dim = d_ingr + d_cat + num_numerical
        self.input_proj = nn.Linear(in_dim, d_model)
        self.dropout_in = nn.Dropout(config["dropout"])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=config["n_layers"],
            enable_nested_tensor=False,
        )

        self.ar_head = nn.Linear(d_model, n_ingr)
        self.cuisine_head = nn.Linear(d_model, n_cuisine)
        self.cat_head = nn.Linear(d_model, n_recipe_cat)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.ingr_emb, self.cat_emb):
            nn.init.trunc_normal_(emb.weight, std=0.02)
        for layer in (self.input_proj, self.ar_head, self.cuisine_head, self.cat_head):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        ingr_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        cls_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = ingr_ids.shape
        device = ingr_ids.device

        name_emb = self.ingr_emb(ingr_ids)
        cat_ids = self.ingr_cat_buf[ingr_ids]
        cat_emb = self.cat_emb(cat_ids)
        num_feat = self.ingr_num_buf[ingr_ids]

        feat = torch.cat([name_emb, cat_emb, num_feat], dim=-1)
        feat = self.dropout_in(self.input_proj(feat))

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

        out = self.transformer(
            feat,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )

        ar_logits = self.ar_head(out)

        cls_idx = cls_positions.view(batch_size, 1, 1).expand(batch_size, 1, out.size(-1))
        cls_out = out.gather(1, cls_idx).squeeze(1)

        cuisine_logits = self.cuisine_head(cls_out)
        cat_logits = self.cat_head(cls_out)
        return ar_logits, cuisine_logits, cat_logits
