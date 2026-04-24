"""Regression tests for preprocessing helpers in `src.data`.

Focus is on de-id token stripping and smart truncation preserving clinical cues.
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import pytest
from src.data import clean_text, smart_truncate


class TestCleanText:
    def test_removes_deid_tokens(self):
        text = "Patient [**Name**] admitted on [**2023-01-01**]"
        result = clean_text(text)
        assert "[**" not in result
        assert "**]" not in result

    def test_lowercases(self):
        assert clean_text("HELLO World") == "hello world"

    def test_strips_special_chars(self):
        result = clean_text("temp=98.6°F; BP: 120/80")
        assert "°" not in result

    def test_handles_none(self):
        assert clean_text(None) == ''

    def test_handles_empty(self):
        assert clean_text('') == ''

    def test_preserves_medical_terms(self):
        text = "WBC-4.8 RBC-2.04 HGB-5.4 HCT-18.3"
        result = clean_text(text)
        assert "wbc-4.8" in result
        assert "rbc-2.04" in result


class TestSmartTruncate:
    def test_respects_max_chars(self):
        text = "a" * 5000
        result = smart_truncate(text, max_chars=2000)
        assert len(result) <= 2000

    def test_extracts_discharge_diagnoses(self):
        text = (
            "HISTORY: long history\n"
            "DISCHARGE DIAGNOSES:\n"
            "1. Heart failure\n"
            "2. Diabetes\n"
            "FOLLOW UP:\n"
            "See PCP in 2 weeks"
        )
        result = smart_truncate(text, max_chars=500)
        assert "heart failure" in result.lower() or "discharge diagnos" in result.lower()

    def test_short_text_unchanged(self):
        text = "Short note about patient."
        result = smart_truncate(text, max_chars=2000)
        assert "short note" in result.lower()
