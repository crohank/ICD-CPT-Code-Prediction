"""
Data loading, text preprocessing, and PyTorch Dataset classes.
Handles both standard (Model B) and chunked (Model C) tokenization.
"""
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .config import (
    DATA_DIR, SECTION_PATTERNS, MAX_SEQ_LEN, TRANSFORMER_MODEL,
    MODEL_C_MAX_CHUNKS, MODEL_C_CHUNK_STRIDE,
)

# ── Text cleaning ──────────────────────────────────────────────────────

_DEID_RE       = re.compile(r'\[\*\*[^\]]*\*\*\]')
_WHITESPACE_RE = re.compile(r'[\s\n\r\t]+')

def clean_text(text: str) -> str:
    """Remove MIMIC de-identification tokens, lowercase, strip non-alphanumeric."""
    if not isinstance(text, str):
        return ''
    text = _DEID_RE.sub(' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,;:\-/]', ' ', text)
    text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


# ── Section-based smart truncation (for Model B) ──────────────────────

_COMPILED_SECTIONS = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in SECTION_PATTERNS
]

def smart_truncate(text: str, max_chars: int = 2048) -> str:
    """
    Extract high-priority clinical sections first, then fill remainder
    from the beginning of the note.  Used by Model B to make the most
    of the 512-token BERT limit.
    """
    segments = []
    total = 0
    for pat in _COMPILED_SECTIONS:
        m = pat.search(text)
        if m:
            seg = m.group(0).strip()
            if total + len(seg) <= max_chars:
                segments.append(seg)
                total += len(seg)
    if total < max_chars:
        segments.insert(0, text[:max_chars - total])
    return ' '.join(segments)[:max_chars]


# ── Data loading helpers ───────────────────────────────────────────────

def load_splits():
    """Load train/val/test DataFrames from processed parquet files."""
    dfs = {}
    for name in ['train', 'val', 'test']:
        df = pd.read_parquet(DATA_DIR / f'cohort_{name}_clean.parquet')
        df['icd_codes'] = df['icd_codes_str'].str.split('|')
        dfs[name] = df
    return dfs['train'], dfs['val'], dfs['test']


def load_label_binarizer():
    """Load the fitted MultiLabelBinarizer."""
    with open(DATA_DIR / 'mlb.pkl', 'rb') as f:
        return pickle.load(f)


def build_label_matrix(df, mlb):
    """Transform ICD code lists into a binary label matrix."""
    return mlb.transform(df['icd_codes']).astype(np.float32)


# ── Standard Dataset (Model B — single 512-token input) ───────────────

class ICDDataset(Dataset):
    """
    Standard dataset for BERT-based classification.
    Each sample is truncated/padded to MAX_SEQ_LEN tokens.
    """
    def __init__(self, texts, labels, tokenizer=None, max_seq_len=MAX_SEQ_LEN):
        self.texts      = list(texts)
        self.labels     = labels
        self.max_seq_len = max_seq_len
        self.tokenizer  = tokenizer or AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Chunked Dataset (Model C — multiple 512-token chunks) ─────────────

class ChunkedICDDataset(Dataset):
    """
    Dataset for chunk-based BERT processing.  Each document is tokenized
    in full, then split into overlapping 512-token windows.

    Returns:
        input_ids:      (max_chunks, max_seq_len)
        attention_mask: (max_chunks, max_seq_len)
        chunk_count:    scalar — number of real (non-padding) chunks
        labels:         (num_labels,)
    """
    def __init__(
        self,
        texts,
        labels,
        tokenizer=None,
        max_seq_len: int = MAX_SEQ_LEN,
        stride: int = MODEL_C_CHUNK_STRIDE,
        max_chunks: int = MODEL_C_MAX_CHUNKS,
    ):
        self.texts       = list(texts)
        self.labels      = labels
        self.max_seq_len = max_seq_len
        self.stride      = stride
        self.max_chunks  = max_chunks
        self.tokenizer   = tokenizer or AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def __len__(self):
        return len(self.texts)

    def _chunk_tokens(self, text: str):
        """Tokenize the full text, then create overlapping chunks."""
        # Tokenize without truncation to get all tokens
        full_enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        token_ids = full_enc['input_ids']

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # Content length per chunk (minus [CLS] and [SEP])
        content_len = self.max_seq_len - 2
        stride = min(self.stride, content_len)

        chunks_ids  = []
        chunks_mask = []

        start = 0
        while start < len(token_ids) and len(chunks_ids) < self.max_chunks:
            end = min(start + content_len, len(token_ids))
            chunk = token_ids[start:end]

            # Add [CLS] ... [SEP] + padding
            ids  = [cls_id] + chunk + [sep_id]
            mask = [1] * len(ids)
            pad_len = self.max_seq_len - len(ids)
            ids  += [pad_id] * pad_len
            mask += [0] * pad_len

            chunks_ids.append(ids)
            chunks_mask.append(mask)

            if end >= len(token_ids):
                break
            start += stride

        chunk_count = len(chunks_ids)

        # Pad to max_chunks with zero-chunks
        while len(chunks_ids) < self.max_chunks:
            chunks_ids.append([pad_id] * self.max_seq_len)
            chunks_mask.append([0] * self.max_seq_len)

        return (
            torch.tensor(chunks_ids,  dtype=torch.long),
            torch.tensor(chunks_mask, dtype=torch.long),
            chunk_count,
        )

    def __getitem__(self, idx):
        input_ids, attention_mask, chunk_count = self._chunk_tokens(self.texts[idx])
        return {
            'input_ids':      input_ids,         # (max_chunks, max_seq_len)
            'attention_mask': attention_mask,     # (max_chunks, max_seq_len)
            'chunk_count':    chunk_count,        # int
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float32),
        }
