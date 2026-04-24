"""
Everything that turns raw discharge rows into tensors: cleaning, optional
section-aware truncation, HuggingFace tokenization, chunking for long notes,
and word-level batching for the BiLSTM model.
"""
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from collections import Counter

from .config import (
    DATA_DIR, SECTION_PATTERNS, MAX_SEQ_LEN, TRANSFORMER_MODEL,
    MODEL_C_MAX_CHUNKS, MODEL_C_CHUNK_STRIDE,
    MODEL_D_MAX_TOKENS, MODEL_D_VOCAB_SIZE,
)

# Strip MIMIC-style placeholders and normalize so train/test see the same alphabet.

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


# Prefer discharge diagnosis / course chunks when we cannot fit the whole note in 512 tokens.

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


# Small I/O helpers used by training scripts and notebooks.

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


# Long-form descriptions — used when initializing label attention queries, not for loss.

# Top-50 ICD-10 code descriptions (manually curated from CMS)
# Used by LabelAttentionClassifier.init_label_queries_from_descriptions()
ICD10_DESCRIPTIONS = {
    "E119":  "Type 2 diabetes mellitus without complications",
    "I10":   "Essential primary hypertension",
    "E780":  "Pure hypercholesterolemia",
    "E785":  "Hyperlipidemia unspecified",
    "Z87891":"Personal history of nicotine dependence",
    "I2510": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "Z7901": "Long term current use of anticoagulants",
    "N179":  "Acute kidney failure unspecified",
    "E1165": "Type 2 diabetes mellitus with hyperglycemia",
    "Z79899":"Other long term current drug therapy",
    "I4891": "Unspecified atrial fibrillation",
    "Z7982": "Long term current use of aspirin",
    "I509":  "Heart failure unspecified",
    "J449":  "Chronic obstructive pulmonary disease unspecified",
    "Z66":   "Do not resuscitate status",
    "E1122": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
    "I2699": "Other pulmonary embolism without acute cor pulmonale",
    "N189":  "Chronic kidney disease unspecified",
    "D649":  "Anemia unspecified",
    "K219":  "Gastro esophageal reflux disease without esophagitis",
    "I480":  "Paroxysmal atrial fibrillation",
    "Z794":  "Long term current use of insulin",
    "N390":  "Urinary tract infection site not specified",
    "G4733": "Obstructive sleep apnea",
    "J9601": "Acute respiratory failure with hypoxia",
    "E876":  "Hypokalemia",
    "I482":  "Chronic atrial fibrillation",
    "Z951":  "Presence of aortocoronary bypass graft",
    "F329":  "Major depressive disorder single episode unspecified",
    "I350":  "Nonrheumatic aortic valve stenosis",
    "Z930":  "Tracheostomy status",
    "I5020": "Unspecified systolic congestive heart failure",
    "J189":  "Pneumonia unspecified organism",
    "Z8546": "Personal history of malignant neoplasm of prostate",
    "G8929": "Other chronic pain",
    "E039":  "Hypothyroidism unspecified",
    "I4892": "Unspecified atrial flutter",
    "Z853":  "Personal history of malignant neoplasm of breast",
    "Z9811": "Acquired absence of right knee joint",
    "I252":  "Old myocardial infarction",
    "B9620": "Unspecified Escherichia coli as the cause of diseases classified elsewhere",
    "J9600": "Acute respiratory failure with hypercapnia",
    "R6520": "Severe sepsis without septic shock",
    "Z68.41":"Body mass index 40.0 to 44.9 adult",
    "K5900": "Constipation unspecified",
    "Z79.01":"Long term current use of anticoagulants",
    "N183":  "Chronic kidney disease stage 3 unspecified",
    "N184":  "Chronic kidney disease stage 4",
    "Z87.11":"Personal history of peptic ulcer disease",
    "I5032": "Chronic diastolic congestive heart failure",
}


def get_icd_descriptions(vocab: list) -> list:
    """
    Get ICD-10 descriptions for a list of codes.
    Falls back to the code itself if no description is available.

    Args:
        vocab: list of ICD-10 code strings

    Returns:
        list of description strings (same order as vocab)
    """
    descriptions = []
    for code in vocab:
        # Try exact match first, then normalized versions
        desc = ICD10_DESCRIPTIONS.get(code)
        if desc is None:
            desc = ICD10_DESCRIPTIONS.get(code.replace('.', ''))
        if desc is None:
            # Fallback: use the code itself as description
            desc = f"ICD-10 code {code}"
        descriptions.append(desc)
    return descriptions


def build_label_matrix(df, mlb):
    """Transform ICD code lists into a binary label matrix."""
    return mlb.transform(df['icd_codes']).astype(np.float32)


# Model B dataset: one truncated window per document.

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


# Model C dataset: stack overlapping BERT windows; collate pads chunk dimension.

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


# Build / persist the word vocab Model D indexes into (frequency-trimmed).

PAD_IDX = 0
UNK_IDX = 1

def build_word_vocab(texts, max_vocab_size: int = MODEL_D_VOCAB_SIZE):
    """
    Build a word-level vocabulary from a list of texts.
    Returns a dict mapping word → index (0=<PAD>, 1=<UNK>, 2+ = words).
    """
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # Most common words (reserve 0 for PAD, 1 for UNK)
    most_common = counter.most_common(max_vocab_size - 2)
    word2idx = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
    for i, (word, _) in enumerate(most_common):
        word2idx[word] = i + 2

    print(f'Built word vocab: {len(word2idx):,} words '
          f'(from {len(counter):,} unique tokens)')
    return word2idx


# Model D dataset: integer word ids up to MAX_TOKENS, no subword splitting.

class BiLSTMDataset(Dataset):
    """
    Dataset for BiLSTM model.  Tokenizes at the word level (whitespace split),
    maps to vocabulary indices, and pads/truncates to max_tokens.

    Returns:
        input_ids:      (max_tokens,)      word indices
        attention_mask: (max_tokens,)      1 for real tokens, 0 for padding
        labels:         (num_labels,)
    """
    def __init__(
        self,
        texts,
        labels,
        word2idx: dict,
        max_tokens: int = MODEL_D_MAX_TOKENS,
    ):
        self.texts      = list(texts)
        self.labels     = labels
        self.word2idx   = word2idx
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()[:self.max_tokens]
        length = len(words)

        # Map words to indices
        ids = [self.word2idx.get(w, UNK_IDX) for w in words]

        # Pad to max_tokens
        pad_len = self.max_tokens - length
        mask = [1] * length + [0] * pad_len
        ids  = ids + [PAD_IDX] * pad_len

        return {
            'input_ids':      torch.tensor(ids,  dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float32),
        }
