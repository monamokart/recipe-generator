import ast
import datetime
import json
import matplotlib.pyplot as plt
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score as sk_f1_score
from torch.utils.data import Dataset

from model import RecipeTransformer
from google.cloud import storage


NUMERICAL_COLS = ["kcal", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g", "sodium_mg"]
INGREDIENT_CATEGORY_PRIORITY = [
    "meats & fishes",
    "vegetables",
    "fruits",
    "cereals & starchy foods",
    "dairy products",
    "nuts",
    "condiments & spices",
]
INGREDIENT_CATEGORY_RANK = {
    category: rank for rank, category in enumerate(INGREDIENT_CATEGORY_PRIORITY)
}


def parse_str_list(value: str) -> list[str]:
    try:
        return [item.strip() for item in ast.literal_eval(value)]
    except Exception:
        return []


def get_top_labels(series: pd.Series, top_n: int) -> set[str]:
    counter = Counter()
    for val in series.dropna():
        counter.update(parse_str_list(str(val)))
    return {label for label, _ in counter.most_common(top_n)}


def filter_multilabel(value: str, keep: set[str], other_label: str = "Other") -> str:
    labels = parse_str_list(value)
    filtered = [label for label in labels if label in keep]
    if any(label not in keep for label in labels):
        filtered.append(other_label)
    return str(filtered)


def encode_multilabel(value: str, vocab: list[str]) -> torch.Tensor:
    labels = set(parse_str_list(value))
    return torch.tensor([1.0 if token in labels else 0.0 for token in vocab], dtype=torch.float32)


@dataclass
class TrainingArtifacts:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    numerical_cols: list[str]
    special_tokens: list[str]
    all_ingredients: list[str]
    ingr_to_idx: dict[str, int]
    idx_to_ingr: dict[int, str]
    cat_vocab: list[str]
    cat_to_idx: dict[str, int]
    cuisine_vocab: list[str]
    recipe_cat_vocab: list[str]
    ingr_cat_ids: torch.Tensor
    ingr_numerical_norm: torch.Tensor
    num_mean: torch.Tensor
    num_std: torch.Tensor
    pad_idx: int
    bos_idx: int
    cls_idx: int


class RecipeDataset(Dataset):
    def __init__(
        self,
        recipes_df: pd.DataFrame,
        ingr_to_idx: dict[str, int],
        pad_idx: int,
        bos_idx: int,
        cuisine_vocab: list[str],
        recipe_cat_vocab: list[str],
    ) -> None:
        self.records: list[tuple[list[int], torch.Tensor, torch.Tensor]] = []

        for _, row in recipes_df.iterrows():
            try:
                ingr_list: list[str] = ast.literal_eval(row["Ingredients"])
            except Exception:
                continue

            ingr_ids = [ingr_to_idx.get(ingr, pad_idx) for ingr in ingr_list]
            if len(ingr_ids) < 2 or pad_idx in ingr_ids:
                continue

            seq = [bos_idx] + ingr_ids
            cuisine_vec = encode_multilabel(str(row.get("Cuisine", "[]")), cuisine_vocab)
            cat_vec = encode_multilabel(str(row.get("Category", "[]")), recipe_cat_vocab)
            self.records.append((seq, cuisine_vec, cat_vec))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        return self.records[idx]
    

class DPORecipeDataset(Dataset):
    def __init__(
        self,
        dpo_recipes_df: pd.DataFrame,
        ingr_to_idx: dict[str, int],
        pad_idx: int,
        bos_idx: int,
        cuisine_vocab: list[str],
        recipe_cat_vocab: list[str],
    ) -> None:
        self.records: list[tuple[list[int], torch.Tensor, torch.Tensor]] = []

        for _, row in dpo_recipes_df.iterrows():
            try:
                ingr_list_w: list[str] = ast.literal_eval(row["recipe_w"])
                ingr_list_l: list[str] = ast.literal_eval(row["recipe_l"])
            except Exception:
                continue

            ingr_ids_w = [ingr_to_idx.get(ingr, pad_idx) for ingr in ingr_list_w]
            ingr_ids_l = [ingr_to_idx.get(ingr, pad_idx) for ingr in ingr_list_l]
            if len(ingr_ids_w) < 2 or pad_idx in ingr_ids_w or len(ingr_ids_l) < 2 or pad_idx in ingr_ids_l:
                continue

            seq_w = [bos_idx] + ingr_ids_w
            seq_l = [bos_idx] + ingr_ids_l
            cuisine_vec = encode_multilabel(str(row.get("Cuisine", "[]")), cuisine_vocab)
            cat_vec = encode_multilabel(str(row.get("Category", "[]")), recipe_cat_vocab)
            self.records.append((seq_w, seq_l, cuisine_vec, cat_vec))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        return self.records[idx]




class RecipeCollator:
    def __init__(self, pad_idx: int, cls_idx: int) -> None:
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx

    def __call__(
        self,
        batch: list[tuple[list[int], torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs, cuisine_vecs, cat_vecs = zip(*batch)

        seqs_cls = [seq + [self.cls_idx] for seq in seqs]
        max_len = max(len(seq) for seq in seqs_cls)
        batch_size = len(seqs_cls)

        ingr_ids = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long)
        pad_mask = torch.ones((batch_size, max_len), dtype=torch.bool)
        cls_positions = torch.zeros(batch_size, dtype=torch.long)

        for index, seq in enumerate(seqs_cls):
            ingr_ids[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            pad_mask[index, : len(seq)] = False
            cls_positions[index] = len(seq) - 1

        return ingr_ids, pad_mask, cls_positions, torch.stack(cuisine_vecs), torch.stack(cat_vecs)

class DPORecipeCollator:
    def __init__(self, pad_idx: int, cls_idx: int) -> None:
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx

    def __call__(
        self,
        batch: list[tuple[list[int], list[int], torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_w, seq_l, cuisine_vecs, cat_vecs = zip(*batch)

        seqs_cls_w = [seq + [self.cls_idx] for seq in seq_w]
        seqs_cls_l = [seq + [self.cls_idx] for seq in seq_l]
        max_len = max(max(len(seq) for seq in seqs_cls_w), max(len(seq) for seq in seqs_cls_l))
        batch_size = len(seqs_cls_w)

        ingr_ids_w = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long)
        ingr_ids_l = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long)
        
        pad_mask_w = torch.ones((batch_size, max_len), dtype=torch.bool)
        pad_mask_l = torch.ones((batch_size, max_len), dtype=torch.bool)
        
        cls_positions_w = torch.zeros(batch_size, dtype=torch.long)
        cls_positions_l = torch.zeros(batch_size, dtype=torch.long)

        for index, seq in enumerate(seqs_cls_w):
            ingr_ids_w[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            pad_mask_w[index, : len(seq)] = False
            cls_positions_w[index] = len(seq) - 1

        for index, seq in enumerate(seqs_cls_l):
            ingr_ids_l[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            pad_mask_l[index, : len(seq)] = False
            cls_positions_l[index] = len(seq) - 1

        return ingr_ids_w, pad_mask_w, cls_positions_w, ingr_ids_l, pad_mask_l, cls_positions_l, torch.stack(cuisine_vecs), torch.stack(cat_vecs)


def reorder_ingredients_by_category(
    value: Any,
    ingredient_to_category: dict[str, str],
) -> str:
    try:
        ingredients = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return str(value)

    if not isinstance(ingredients, list) or len(ingredients) < 2:
        return str(ingredients)

    indexed_ingredients = list(enumerate(ingredients))

    def category_sort_key(item: tuple[int, Any]) -> tuple[int, int]:
        index, ingredient = item
        category = ingredient_to_category.get(str(ingredient).strip().lower(), "")
        return INGREDIENT_CATEGORY_RANK.get(category, len(INGREDIENT_CATEGORY_PRIORITY)), index

    ordered = [ingredient for _, ingredient in sorted(indexed_ingredients, key=category_sort_key)]
    return str(ordered)


def augment_recipes_by_category_order(
    recipes_df: pd.DataFrame,
    ingredient_to_category: dict[str, str],
) -> pd.DataFrame:

    augmented = recipes_df.copy()
    augmented["Ingredients"] = augmented["Ingredients"].apply(
        lambda value: reorder_ingredients_by_category(value, ingredient_to_category)
    )

    return augmented


def build_training_artifacts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ingredient_df: pd.DataFrame,
    top_n_cuisines: int,
    top_n_categories: int,
) -> TrainingArtifacts:
    special_tokens = ["<PAD>", "<BOS>", "<CLS>"]
    all_ingredients = special_tokens + ingredient_df["ingredient"].tolist()
    ingr_to_idx = {name: idx for idx, name in enumerate(all_ingredients)}
    idx_to_ingr = {idx: name for name, idx in ingr_to_idx.items()}

    pad_idx = ingr_to_idx["<PAD>"]
    bos_idx = ingr_to_idx["<BOS>"]
    cls_idx = ingr_to_idx["<CLS>"]

    cat_vocab = ["<PAD>"] + sorted(ingredient_df["category"].dropna().astype(str).unique().tolist())
    cat_to_idx = {name: idx for idx, name in enumerate(cat_vocab)}

    ingr_cat_ids = torch.zeros(len(all_ingredients), dtype=torch.long)
    ingr_numerical = torch.zeros(len(all_ingredients), len(NUMERICAL_COLS), dtype=torch.float32)

    for _, row in ingredient_df.iterrows():
        idx = ingr_to_idx.get(str(row["ingredient"]))
        if idx is None:
            continue
        ingr_cat_ids[idx] = cat_to_idx.get(str(row["category"]), 0)
        ingr_numerical[idx] = torch.tensor([float(row[col]) for col in NUMERICAL_COLS], dtype=torch.float32)

    real_mask = torch.tensor([ingr_to_idx[name] >= len(special_tokens) for name in all_ingredients])
    num_mean = ingr_numerical[real_mask].mean(0)
    num_std = ingr_numerical[real_mask].std(0).clamp(min=1e-6)
    ingr_numerical_norm = (ingr_numerical - num_mean) / num_std
    ingr_numerical_norm[: len(special_tokens)] = 0.0

    all_cuisine = pd.concat([train_df["Cuisine"], val_df["Cuisine"]], ignore_index=True)
    all_category = pd.concat([train_df["Category"], val_df["Category"]], ignore_index=True)

    top_cuisines = get_top_labels(all_cuisine, top_n=top_n_cuisines)
    top_categories = get_top_labels(all_category, top_n=top_n_categories)

    train_df = train_df.copy()
    val_df = val_df.copy()

    ingredient_to_category = {
        str(row["ingredient"]).strip().lower(): str(row["category"]).strip().lower()
        for _, row in ingredient_df.iterrows()
    }

    for frame in (train_df, val_df):
        frame["Cuisine"] = frame["Cuisine"].apply(
            lambda value: filter_multilabel(str(value), top_cuisines, "Other")
        )
        frame["Category"] = frame["Category"].apply(
            lambda value: filter_multilabel(str(value), top_categories, "Other")
        )

    train_df = augment_recipes_by_category_order(
        recipes_df=train_df,
        ingredient_to_category=ingredient_to_category,
    )

    val_df = augment_recipes_by_category_order(
        recipes_df=val_df,
        ingredient_to_category=ingredient_to_category,
    )

    cuisine_vocab = sorted(top_cuisines) + ["Other"]
    recipe_cat_vocab = sorted(top_categories) + ["Other"]

    return TrainingArtifacts(
        train_df=train_df,
        val_df=val_df,
        numerical_cols=NUMERICAL_COLS,
        special_tokens=special_tokens,
        all_ingredients=all_ingredients,
        ingr_to_idx=ingr_to_idx,
        idx_to_ingr=idx_to_ingr,
        cat_vocab=cat_vocab,
        cat_to_idx=cat_to_idx,
        cuisine_vocab=cuisine_vocab,
        recipe_cat_vocab=recipe_cat_vocab,
        ingr_cat_ids=ingr_cat_ids,
        ingr_numerical_norm=ingr_numerical_norm,
        num_mean=num_mean,
        num_std=num_std,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        cls_idx=cls_idx,
    )


def ar_loss_and_acc(
    ar_logits: torch.Tensor,
    ingr_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    cls_positions: torch.Tensor,
    pad_idx: int,
) -> tuple[torch.Tensor, float]:
    batch_size, seq_len, _ = ar_logits.shape
    device = ingr_ids.device

    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    cls_pos = cls_positions.unsqueeze(1)
    valid = (pos_idx < cls_pos - 1) & (~pad_mask)

    targets = torch.roll(ingr_ids, -1, dims=1)
    targets[:, -1] = pad_idx

    logits_flat = ar_logits[valid]
    targets_flat = targets[valid]

    if logits_flat.size(0) == 0:
        return ar_logits.sum() * 0.0, 0.0

    loss = F.cross_entropy(logits_flat, targets_flat)
    with torch.no_grad():
        acc = (logits_flat.argmax(-1) == targets_flat).float().mean().item()
    return loss, acc


def cls_loss_and_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, float]:
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= threshold).cpu().numpy().astype(int)
        tgts = targets.cpu().numpy().astype(int)
        f1 = sk_f1_score(tgts, preds, average="macro", zero_division=0)
    return loss, float(f1)


def save_run_to_gcs(
    model_state_dict: dict[str, Any],
    config: dict[str, Any],
    history: dict[str, list[float]],
    vocab_meta: dict[str, Any],
) -> str:

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{config['gcs_model_prefix']}/{timestamp}"

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(config["gcs_bucket"])

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        tmp_path = handle.name

    torch.save(model_state_dict, tmp_path)
    bucket.blob(f"{run_prefix}/model.pt").upload_from_filename(tmp_path)
    os.unlink(tmp_path)

    bucket.blob(f"{run_prefix}/config.json").upload_from_string(
        json.dumps(config, indent=2),
        content_type="application/json",
    )

    history_serializable = {k: [float(v) for v in values] for k, values in history.items()}
    bucket.blob(f"{run_prefix}/history.json").upload_from_string(
        json.dumps(history_serializable, indent=2),
        content_type="application/json",
    )

    bucket.blob(f"{run_prefix}/vocab_meta.json").upload_from_string(
        json.dumps(vocab_meta, indent=2),
        content_type="application/json",
    )

    return f"gs://{config['gcs_bucket']}/{run_prefix}"


def load_latest_model_from_gcs(gcs_bucket: str, gcs_model_prefix: str) -> tuple[RecipeTransformer, dict[str, Any], dict[str, Any], dict[str, Any]]:
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket)

    # Collect unique top-level run timestamps under models/
    blobs = list(bucket.list_blobs(prefix=f"{gcs_model_prefix}/"))
    run_timestamps = sorted(
        {b.name.split("/")[1] for b in blobs if len(b.name.split("/")) > 2},
        reverse=True,
    )
    if not run_timestamps:
        raise RuntimeError(f"No model runs found under gs://{gcs_bucket}/{gcs_model_prefix}/")

    latest_ts = run_timestamps[0]
    run_prefix = f"{gcs_model_prefix}/{latest_ts}"
    print(f"Loading run: gs://{gcs_bucket}/{run_prefix}")

    # ── Download artifacts ───────────────────────────────────────────────────────
    def _download_json(blob_name: str) -> dict:
        return json.loads(bucket.blob(blob_name).download_as_text())

    config    = _download_json(f"{run_prefix}/config.json")
    history   = _download_json(f"{run_prefix}/history.json")
    vocab_meta = _download_json(f"{run_prefix}/vocab_meta.json")

    # Download model.pt to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        model_tmp = fh.name
    bucket.blob(f"{run_prefix}/model.pt").download_to_filename(model_tmp)

    # ── Reconstruct and load model ───────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The registered buffers are inside model.pt; we pass dummy tensors that will
    # be overwritten by load_state_dict.
    n_ingr      = vocab_meta["n_ingr"]
    n_cat       = vocab_meta["n_cat"]
    n_cuisine   = vocab_meta["n_cuisine"]
    n_recipe_cat = vocab_meta["n_recipe_cat"]
    num_numerical = vocab_meta["num_numerical"]
    pad_idx     = vocab_meta["pad_idx"]

    model = RecipeTransformer(
        config=config,
        n_ingr=n_ingr,
        n_cat=n_cat,
        n_cuisine=n_cuisine,
        n_recipe_cat=n_recipe_cat,
        num_numerical=num_numerical,
        ingr_cat_buf=torch.zeros(n_ingr, dtype=torch.long),
        ingr_num_buf=torch.zeros(n_ingr, num_numerical),
        pad_idx=pad_idx,
        cat_pad_idx=0,
    ).to(device)

    state_dict = torch.load(model_tmp, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    os.unlink(model_tmp)

    print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Epochs in history: {len(history['train_total_loss'])}")
    return model, config, history, vocab_meta


def plot_losses(history: dict[str, list[float]]) -> None:
    loss_keys = ["total_loss", "ar_loss", "cuisine_loss", "cat_loss"]
    loss_labels = {
        "total_loss":   "Total loss",
        "ar_loss":      "AR loss",
        "cuisine_loss": "Cuisine loss",
        "cat_loss":     "Category loss",
    }

    epochs = range(1, len(history["train_total_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Loss curves", fontsize=14, fontweight="bold")

    for ax, key in zip(axes.flat, loss_keys):
        train_vals = history[f"train_{key}"]
        val_vals   = history[f"val_{key}"]
        ax.plot(epochs, train_vals, label="Train", marker="o", markersize=3)
        ax.plot(epochs, val_vals,   label="Val",   marker="s", markersize=3, linestyle="--")
        ax.set_title(loss_labels[key])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        best_epoch = int(np.argmin(val_vals)) + 1
        ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.6, label=f"Best val (ep {best_epoch})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_metrics(history: dict[str, list[float]]) -> None:
    metric_keys = ["ar_acc", "cuisine_f1", "cat_f1"]
    metric_labels = {
        "ar_acc":      "AR accuracy",
        "cuisine_f1":  "Cuisine F1 (macro)",
        "cat_f1":      "Category F1 (macro)",
    }
    epochs = range(1, len(history["train_total_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Metric curves", fontsize=14, fontweight="bold")

    for ax, key in zip(axes, metric_keys):
        train_vals = history[f"train_{key}"]
        val_vals   = history[f"val_{key}"]
        ax.plot(epochs, train_vals, label="Train", marker="o", markersize=3)
        ax.plot(epochs, val_vals,   label="Val",   marker="s", markersize=3, linestyle="--")
        ax.set_title(metric_labels[key])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        best_epoch = int(np.argmax(val_vals)) + 1
        ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.6, label=f"Best val (ep {best_epoch})")
        ax.legend(fontsize=8)

        # Print best val score
        print(f"{metric_labels[key]:30s} | best val = {max(val_vals):.4f} (epoch {best_epoch})")

    plt.tight_layout()
    plt.show()