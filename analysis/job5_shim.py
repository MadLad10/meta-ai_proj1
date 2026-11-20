"""class_token_map copied verbatim from gpu_job/job5_duo_zeroshot.py (Grounding DINO positive map)."""

def class_token_map(caption, class_names, offsets):
    """Token positions of each class name in the caption (from the tokenizer's character offsets), in class order.
    Names are searched left to right, so 'fish' after 'starfish' finds its own word."""
    rows, pos = [], 0
    for nm in class_names:
        s = caption.index(nm, pos); e = s + len(nm); pos = e
        toks = [i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e]
        if not toks:
            raise ValueError(f'no tokens for class name {nm!r} in {caption!r}')
        rows.append(toks)
    return rows


