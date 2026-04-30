import torch

def build_vocab_helpers(vocab_meta, config):
    ingr_to_idx   = vocab_meta["ingr_to_idx"]
    idx_to_ingr   = {int(k): v for k, v in vocab_meta.get("idx_to_ingr", {}).items()}
    if not idx_to_ingr:
        idx_to_ingr = {v: k for k, v in ingr_to_idx.items()}

    cuisine_vocab    = vocab_meta["cuisine_vocab"]
    recipe_cat_vocab = vocab_meta["recipe_cat_vocab"]

    bos_idx = vocab_meta["bos_idx"]
    cls_idx = vocab_meta["cls_idx"]
    _pad_idx = vocab_meta["pad_idx"]
    _special_ids = {_pad_idx, bos_idx, cls_idx}
    max_seq_len = config["max_seq_len"]
    return ingr_to_idx, idx_to_ingr, cuisine_vocab, recipe_cat_vocab, bos_idx, cls_idx, _special_ids, max_seq_len


@torch.no_grad()
def predict_cuisine_and_category(model, seq_ids: list[int], cuisine_vocab, recipe_cat_vocab, cls_idx, top_k: int = 3) -> tuple[list, list]:
    """Predict top-k cuisines and categories from the CLS token."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_cls = seq_ids + [cls_idx]
    ingr_t = torch.tensor([seq_cls], dtype=torch.long, device=device)
    pad_m  = torch.zeros(1, len(seq_cls), dtype=torch.bool, device=device)
    cls_pos = torch.tensor([len(seq_cls) - 1], dtype=torch.long, device=device)

    _, cuisine_logits, cat_logits = model(ingr_t, pad_m, cls_pos)

    cus_probs  = torch.sigmoid(cuisine_logits[0])
    cat_probs  = torch.sigmoid(cat_logits[0])

    top_cus  = sorted(zip(cuisine_vocab,    cus_probs.tolist()), key=lambda x: -x[1])[:top_k]
    top_cats = sorted(zip(recipe_cat_vocab, cat_probs.tolist()), key=lambda x: -x[1])[:top_k]
    return top_cus, top_cats

@torch.no_grad()
def generate_recipe(
    model: torch.nn.Module,
    starter_ingredients: list[str],
    ingr_to_idx: dict[str, int],
    idx_to_ingr: dict[int, str],
    cuisine_vocab: list[str],
    recipe_cat_vocab: list[str],
    _special_ids: set[int],
    bos_idx: int = 1,
    cls_idx: int = 2,
    max_seq_len: int = 10,
    top_k: int = 1,
    temperature: float = 1.0,
) -> dict:
    """Auto-regressively generate ingredients and predict cuisine/category."""
    # Encode starter ingredients; skip unknowns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = [bos_idx]
    unknown = []
    for name in starter_ingredients:
        idx = ingr_to_idx.get(name)
        if idx is None:
            unknown.append(name)
        else:
            seq.append(idx)

    if unknown:
        print(f"[Warning] Unknown ingredients (skipped): {unknown}")

    generated = []
    while len(seq) < max_seq_len - 1:

        seq_cls = seq + [cls_idx]
        ingr_t  = torch.tensor([seq_cls], dtype=torch.long, device=device)
        pad_m   = torch.zeros(1, len(seq_cls), dtype=torch.bool, device=device)
        cls_pos = torch.tensor([len(seq_cls) - 1], dtype=torch.long, device=device)

        ar_logits, _, _ = model(ingr_t, pad_m, cls_pos)
        next_pos_logits = ar_logits[0, len(seq) - 1]  # predict at last real token

        # Mask specials
        for sid in _special_ids:
            next_pos_logits[sid] = -1e9
        # Mask already-used ingredients
        for used_id in seq:
            next_pos_logits[used_id] = -1e9

        if temperature != 1.0:
            next_pos_logits = next_pos_logits / temperature

        if top_k == 1:
            next_id = int(next_pos_logits.argmax())
        else:
            probs = torch.softmax(next_pos_logits, dim=-1)
            topk_probs, topk_ids = probs.topk(top_k)
            next_id = int(topk_ids[torch.multinomial(topk_probs, 1)])

        seq.append(next_id)
        generated.append(idx_to_ingr.get(next_id, f"<{next_id}>"))

    top_cuisines, top_cats = predict_cuisine_and_category(model, seq, cuisine_vocab, recipe_cat_vocab, cls_idx, top_k=3)
    return {
        "starter":    starter_ingredients,
        "generated":  generated,
        "full_recipe": [idx_to_ingr.get(i, f"<{i}>") for i in seq[1:]],  # skip BOS
        "top_cuisines":   top_cuisines,
        "top_categories": top_cats,
    }
