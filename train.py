import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import RecipeTransformer
from utils import (
    RecipeCollator,
    RecipeDataset,
    ar_loss_and_acc,
    build_training_artifacts,
    cls_loss_and_f1,
    save_run_to_gcs,
)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config: dict[str, Any], artifacts, device: torch.device) -> RecipeTransformer:
    model = RecipeTransformer(
        config=config,
        n_ingr=len(artifacts.all_ingredients),
        n_cat=len(artifacts.cat_vocab),
        n_cuisine=len(artifacts.cuisine_vocab),
        n_recipe_cat=len(artifacts.recipe_cat_vocab),
        num_numerical=len(artifacts.numerical_cols),
        ingr_cat_buf=artifacts.ingr_cat_ids.clone(),
        ingr_num_buf=artifacts.ingr_numerical_norm.clone(),
        pad_idx=artifacts.pad_idx,
        cat_pad_idx=0,
    ).to(device)
    return model


def run_epoch(
    model: RecipeTransformer,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    pad_idx: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "ar_loss": 0.0,
        "ar_acc": 0.0,
        "cuisine_loss": 0.0,
        "cuisine_f1": 0.0,
        "cat_loss": 0.0,
        "cat_f1": 0.0,
        "total_loss": 0.0,
    }

    context = torch.enable_grad() if training else torch.no_grad()
    batches = 0

    with context:
        for ingr_ids, pad_mask, cls_positions, cuisine_vecs, cat_vecs in loader:
            ingr_ids = ingr_ids.to(device)
            pad_mask = pad_mask.to(device)
            cls_positions = cls_positions.to(device)
            cuisine_vecs = cuisine_vecs.to(device)
            cat_vecs = cat_vecs.to(device)

            ar_logits, cuisine_logits, cat_logits = model(ingr_ids, pad_mask, cls_positions)

            ar_l, ar_a = ar_loss_and_acc(ar_logits, ingr_ids, pad_mask, cls_positions, pad_idx)
            cus_l, cus_f1 = cls_loss_and_f1(cuisine_logits, cuisine_vecs)
            cat_l, cat_f1 = cls_loss_and_f1(cat_logits, cat_vecs)

            total = (
                config["w_ar"] * ar_l
                + config["w_cuisine"] * cus_l
                + config["w_category"] * cat_l
            )

            if training:
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            totals["ar_loss"] += ar_l.item()
            totals["ar_acc"] += ar_a
            totals["cuisine_loss"] += cus_l.item()
            totals["cuisine_f1"] += cus_f1
            totals["cat_loss"] += cat_l.item()
            totals["cat_f1"] += cat_f1
            totals["total_loss"] += total.item()
            batches += 1

    return {name: value / max(batches, 1) for name, value in totals.items()}


def train(config: dict[str, Any], train_data: str, val_data: str, ingredients_data: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_data)
    val_df = pd.read_csv(val_data)
    ingredient_df = pd.read_csv(ingredients_data)

    artifacts = build_training_artifacts(
        train_df=train_df,
        val_df=val_df,
        ingredient_df=ingredient_df,
        top_n_cuisines=int(config["top_n_cuisines"]),
        top_n_categories=int(config["top_n_categories"]),
    )

    train_dataset = RecipeDataset(
        recipes_df=artifacts.train_df,
        ingr_to_idx=artifacts.ingr_to_idx,
        pad_idx=artifacts.pad_idx,
        bos_idx=artifacts.bos_idx,
        cuisine_vocab=artifacts.cuisine_vocab,
        recipe_cat_vocab=artifacts.recipe_cat_vocab,
    )
    val_dataset = RecipeDataset(
        recipes_df=artifacts.val_df,
        ingr_to_idx=artifacts.ingr_to_idx,
        pad_idx=artifacts.pad_idx,
        bos_idx=artifacts.bos_idx,
        cuisine_vocab=artifacts.cuisine_vocab,
        recipe_cat_vocab=artifacts.recipe_cat_vocab,
    )

    collator = RecipeCollator(pad_idx=artifacts.pad_idx, cls_idx=artifacts.cls_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    model = build_model(config=config, artifacts=artifacts, device=device)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))

    history = {
        f"{split}_{metric}": []
        for split in ("train", "val")
        for metric in (
            "ar_loss",
            "ar_acc",
            "cuisine_loss",
            "cuisine_f1",
            "cat_loss",
            "cat_f1",
            "total_loss",
        )
    }

    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            config=config,
            device=device,
            pad_idx=artifacts.pad_idx,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            config=config,
            device=device,
            pad_idx=artifacts.pad_idx,
            optimizer=None,
        )
        scheduler.step()

        for metric, value in train_metrics.items():
            history[f"train_{metric}"].append(value)
        for metric, value in val_metrics.items():
            history[f"val_{metric}"].append(value)

        print(
            f"Epoch {epoch:02d}/{int(config['epochs'])} "
            f"| train loss={train_metrics['total_loss']:.4f} ar_acc={train_metrics['ar_acc']:.3f} "
            f"cus_f1={train_metrics['cuisine_f1']:.3f} cat_f1={train_metrics['cat_f1']:.3f} "
            f"| val loss={val_metrics['total_loss']:.4f} ar_acc={val_metrics['ar_acc']:.3f} "
            f"cus_f1={val_metrics['cuisine_f1']:.3f} cat_f1={val_metrics['cat_f1']:.3f}"
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(config["early_stopping_patience"]):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {int(config['early_stopping_patience'])} epochs)."
                )
                break

    model.load_state_dict(best_state_dict)

    vocab_meta = {
        "n_ingr": len(artifacts.all_ingredients),
        "n_cat": len(artifacts.cat_vocab),
        "n_cuisine": len(artifacts.cuisine_vocab),
        "n_recipe_cat": len(artifacts.recipe_cat_vocab),
        "num_numerical": len(artifacts.numerical_cols),
        "pad_idx": artifacts.pad_idx,
        "bos_idx": artifacts.bos_idx,
        "cls_idx": artifacts.cls_idx,
        "ingr_to_idx": artifacts.ingr_to_idx,
        "cuisine_vocab": artifacts.cuisine_vocab,
        "recipe_cat_vocab": artifacts.recipe_cat_vocab,
        "cat_vocab": artifacts.cat_vocab,
        "numerical_cols": artifacts.numerical_cols,
        "num_mean": artifacts.num_mean.tolist(),
        "num_std": artifacts.num_std.tolist(),
    }

    gcs_path = save_run_to_gcs(
        model_state_dict=model.state_dict(),
        config=config,
        history=history,
        vocab_meta=vocab_meta,
    )
    print(f"Artifacts saved at: {gcs_path}")

    return model, history, gcs_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train recipe multi-task transformer.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON file.")
    parser.add_argument(
        "--train-data",
        default="gs://recipe-generation/train_recipes.csv",
        help="Path to training recipes CSV.",
    )
    parser.add_argument(
        "--val-data",
        default="gs://recipe-generation/test_recipes.csv",
        help="Path to validation recipes CSV.",
    )
    parser.add_argument(
        "--ingredients-data",
        default="gs://recipe-generation/ingredient_profiles.csv",
        help="Path to ingredient profiles CSV.",
    )
    return parser


if __name__ == "__main__":

    args = build_arg_parser().parse_args()
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = load_config(args.config)

    train(
        config=cfg,
        train_data=args.train_data,
        val_data=args.val_data,
        ingredients_data=args.ingredients_data,
    )
