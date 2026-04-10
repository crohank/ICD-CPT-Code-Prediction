"""Tests for the FastAPI endpoints (uses TestClient, no real models needed)."""
import sys
sys.path.insert(0, '..')

import pytest


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_prediction_request_valid(self):
        from api.schemas import PredictionRequest
        req = PredictionRequest(text="Patient admitted with chest pain.", top_n=5)
        assert req.top_n == 5
        assert req.threshold is None

    def test_prediction_request_short_text(self):
        from api.schemas import PredictionRequest
        with pytest.raises(Exception):  # Pydantic ValidationError
            PredictionRequest(text="short")

    def test_prediction_request_defaults(self):
        from api.schemas import PredictionRequest
        req = PredictionRequest(text="A long enough discharge summary text here.")
        assert req.top_n == 10
        assert req.explain is False

    def test_code_prediction(self):
        from api.schemas import CodePrediction
        pred = CodePrediction(
            icd_code="I50.9",
            probability=0.85,
            predicted=True,
        )
        assert pred.icd_code == "I50.9"
        assert pred.evidence is None
