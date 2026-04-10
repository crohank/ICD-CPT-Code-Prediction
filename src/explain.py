"""
Explainability utilities for Model C (Label Attention).
Extracts attention weights and generates highlighted text showing
which words/phrases contributed to each predicted ICD-10 code.
"""
import numpy as np
import torch
from transformers import AutoTokenizer

from .config import TRANSFORMER_MODEL, MAX_SEQ_LEN, MODEL_C_MAX_CHUNKS, MODEL_C_CHUNK_STRIDE
from .data import clean_text, ChunkedICDDataset


def extract_attention_for_text(
    model,
    text: str,
    tokenizer=None,
    device: str = 'cuda',
    top_k_tokens: int = 20,
):
    """
    Run a single text through Model C and extract per-label attention weights.

    Args:
        model:         LabelAttentionClassifier instance (eval mode)
        text:          raw or cleaned discharge note
        tokenizer:     HuggingFace tokenizer (auto-loaded if None)
        device:        'cuda' or 'cpu'
        top_k_tokens:  number of top-attended tokens to return per label

    Returns:
        logits:     (num_labels,) raw logit values
        probs:      (num_labels,) sigmoid probabilities
        attention:  dict mapping label_index -> list of (token_str, weight)
                    sorted by weight descending, top_k_tokens entries
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    model.eval()

    # Create a single-sample chunked input
    dummy_labels = np.zeros((1, model.num_labels), dtype=np.float32)
    ds = ChunkedICDDataset(
        texts=[text],
        labels=dummy_labels,
        tokenizer=tokenizer,
    )
    sample = ds[0]

    input_ids      = sample['input_ids'].unsqueeze(0).to(device)       # (1, C, S)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)   # (1, C, S)

    with torch.no_grad(), torch.amp.autocast(device, enabled=device == 'cuda'):
        logits, attn_weights = model(
            input_ids, attention_mask, return_attention=True,
        )

    logits_np = logits.cpu().float().numpy()[0]              # (num_labels,)
    probs_np  = torch.sigmoid(logits).cpu().float().numpy()[0]
    attn_np   = attn_weights.cpu().float().numpy()[0]        # (num_labels, total_tokens)

    # Map token indices back to strings
    flat_ids = input_ids.view(-1).cpu().tolist()             # (C * S,)
    flat_mask = attention_mask.view(-1).cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(flat_ids)

    # Build per-label top-K attended tokens
    attention_dict = {}
    for label_idx in range(model.num_labels):
        weights = attn_np[label_idx]

        # Filter out padding and special tokens
        token_weights = []
        for t_idx, (tok, w, m) in enumerate(zip(tokens, weights, flat_mask)):
            if m == 0 or tok in ('[PAD]', '[CLS]', '[SEP]'):
                continue
            token_weights.append((tok, float(w)))

        # Sort by weight descending
        token_weights.sort(key=lambda x: x[1], reverse=True)
        attention_dict[label_idx] = token_weights[:top_k_tokens]

    return logits_np, probs_np, attention_dict


def highlight_text_html(
    text: str,
    attention_weights: list,
    max_highlight: int = 15,
) -> str:
    """
    Generate HTML with spans colored by attention weight intensity.

    Args:
        text:              original text string
        attention_weights: list of (token_str, weight) from extract_attention_for_text
        max_highlight:     maximum number of tokens to highlight

    Returns:
        HTML string with colored <span> tags
    """
    # Get top tokens to highlight
    top_tokens = {tok.replace('##', ''): w
                  for tok, w in attention_weights[:max_highlight]}

    if not top_tokens:
        return f'<p>{text}</p>'

    max_w = max(top_tokens.values()) if top_tokens else 1.0

    words = text.split()
    html_parts = []
    for word in words:
        word_lower = word.lower().strip('.,;:!?')
        if word_lower in top_tokens:
            # Normalize weight to 0-1 range for opacity
            intensity = top_tokens[word_lower] / max_w
            r, g, b = 255, int(255 * (1 - intensity)), int(255 * (1 - intensity))
            html_parts.append(
                f'<span style="background-color: rgba({r},{g},{b},0.6); '
                f'padding: 2px 4px; border-radius: 3px; '
                f'font-weight: {"bold" if intensity > 0.5 else "normal"}">'
                f'{word}</span>'
            )
        else:
            html_parts.append(word)

    return ' '.join(html_parts)


def explain_prediction(
    text: str,
    model,
    mlb,
    tokenizer=None,
    threshold: float = 0.5,
    top_n_codes: int = 10,
    top_k_tokens: int = 15,
    device: str = 'cuda',
    icd_descriptions: dict = None,
):
    """
    End-to-end explainable prediction: clean text -> predict -> explain.

    Args:
        text:              raw discharge summary
        model:             LabelAttentionClassifier instance
        mlb:               fitted MultiLabelBinarizer
        tokenizer:         HuggingFace tokenizer
        threshold:         prediction threshold
        top_n_codes:       number of top codes to return
        top_k_tokens:      tokens per code for explanation
        device:            'cuda' or 'cpu'
        icd_descriptions:  optional dict mapping ICD code -> description

    Returns:
        list of dicts: [{
            'code': str,
            'description': str,
            'probability': float,
            'predicted': bool,
            'evidence': [(token, weight), ...],
            'evidence_html': str,
        }]
    """
    cleaned = clean_text(text)
    vocab = list(mlb.classes_)

    logits, probs, attention = extract_attention_for_text(
        model, cleaned, tokenizer=tokenizer,
        device=device, top_k_tokens=top_k_tokens,
    )

    # Rank by probability
    ranked_indices = np.argsort(probs)[::-1][:top_n_codes]

    results = []
    for idx in ranked_indices:
        code = vocab[idx]
        prob = float(probs[idx])
        evidence = attention.get(idx, [])

        results.append({
            'code':         code,
            'description':  icd_descriptions.get(code, '') if icd_descriptions else '',
            'probability':  round(prob, 4),
            'predicted':    prob >= threshold,
            'evidence':     evidence,
            'evidence_html': highlight_text_html(cleaned, evidence),
        })

    return results
