"""Test quality filtering and duplicate detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.stage_a_mt.build_data import _row_quality_reasons


def test_reject_empty_text():
    row = {"tvl": "", "en": "hello"}
    reasons = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=10,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=False,
    )
    assert "empty_text" in reasons


def test_reject_too_short():
    row = {"tvl": "hi", "en": "hi", "alignment_confidence": 0.9}
    reasons = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=10,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=False,
    )
    assert "too_short" in reasons


def test_reject_bad_length_ratio():
    row = {
        "tvl": "a" * 100,
        "en": "b" * 10,
        "alignment_confidence": 0.9,
        "length_ratio": 10.0,
    }
    reasons = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=5,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=False,
    )
    assert "bad_length_ratio" in reasons


def test_reject_low_confidence():
    row = {
        "tvl": "Toku taeao" * 3,
        "en": "Good morning" * 3,
        "alignment_confidence": 0.3,
    }
    reasons = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=5,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=False,
    )
    assert "low_alignment_confidence" in reasons


def test_accept_good_row():
    row = {
        "tvl": "Ko te mea tino moni tena",
        "en": "That is a very true thing",
        "alignment_confidence": 0.95,
    }
    reasons = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=5,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=False,
    )
    assert reasons == []


def test_allow_low_conf_article_flag():
    row = {
        "tvl": "article paragraph text here",
        "en": "article paragraph text here",
        "alignment_confidence": 0.5,
        "content_type": "article_paragraph",
        "alignment_method": "document_level",
    }
    # With flag off: rejected
    reasons_off = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=5,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=False,
    )
    assert "low_alignment_confidence" in reasons_off

    # With flag on: accepted
    reasons_on = _row_quality_reasons(
        row,
        min_confidence=0.8,
        min_chars=5,
        max_chars=4096,
        ratio_min=0.4,
        ratio_max=2.5,
        allow_low_conf_article=True,
    )
    assert "low_alignment_confidence" not in reasons_on
