"""Test split determinism for Stage A data builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tv.training.stage_a_mt.build_data import (
    _assign_split,
    _stable_hash,
    _normalize_for_hash,
    _normalize_preserve_structure,
)


def test_stable_hash_determinism():
    """Same input always gives same hash."""
    assert _stable_hash("bible_book_8") == _stable_hash("bible_book_8")
    assert _stable_hash("doc_12345") == _stable_hash("doc_12345")
    assert _stable_hash("a") != _stable_hash("b")


def test_normalize_for_hash():
    """Whitespace normalization for dedup hashing."""
    assert _normalize_for_hash("  hello   world  ") == "hello world"
    assert _normalize_for_hash("a\n\nb") == "a b"


def test_normalize_preserve_structure():
    """Structure-preserving normalization keeps newlines."""
    text = "line 1  \n  line 2  "
    result = _normalize_preserve_structure(text)
    assert result == "line 1\nline 2"


def test_bible_split_by_book():
    """Bible verses split by held-out books, not random."""
    # Ruth (book 8) -> test
    row_ruth = {"content_type": "bible_verse", "book_num": 8}
    assert _assign_split(row_ruth, non_bible_val_frac=0.05, non_bible_test_frac=0.05) == "test"

    # Obadiah (book 31) -> validation
    row_obadiah = {"content_type": "bible_verse", "book_num": 31}
    assert _assign_split(row_obadiah, non_bible_val_frac=0.05, non_bible_test_frac=0.05) == "validation"

    # Genesis (book 1) -> train
    row_genesis = {"content_type": "bible_verse", "book_num": 1}
    assert _assign_split(row_genesis, non_bible_val_frac=0.05, non_bible_test_frac=0.05) == "train"


def test_article_split_by_doc_id():
    """Articles split by doc_id, same doc always same split."""
    row1 = {"content_type": "article_paragraph", "doc_id": "42"}
    row2 = {"content_type": "article_paragraph", "doc_id": "42"}
    s1 = _assign_split(row1, non_bible_val_frac=0.05, non_bible_test_frac=0.05)
    s2 = _assign_split(row2, non_bible_val_frac=0.05, non_bible_test_frac=0.05)
    assert s1 == s2


def test_daily_text_split_by_date():
    """Daily texts split by date, same date always same split."""
    row1 = {"content_type": "daily_text", "date": "2025-01-01"}
    row2 = {"content_type": "daily_text", "date": "2025-01-01"}
    s1 = _assign_split(row1, non_bible_val_frac=0.05, non_bible_test_frac=0.05)
    s2 = _assign_split(row2, non_bible_val_frac=0.05, non_bible_test_frac=0.05)
    assert s1 == s2


def test_split_determinism_across_calls():
    """Multiple calls with same input give identical results."""
    rows = [
        {"content_type": "bible_verse", "book_num": 1, "id": "bv_1_1_1"},
        {"content_type": "article_paragraph", "doc_id": "100", "id": "art_100_1"},
        {"content_type": "daily_text", "date": "2025-06-15", "id": "dt_20250615"},
    ]
    splits_1 = [
        _assign_split(r, non_bible_val_frac=0.05, non_bible_test_frac=0.05) for r in rows
    ]
    splits_2 = [
        _assign_split(r, non_bible_val_frac=0.05, non_bible_test_frac=0.05) for r in rows
    ]
    assert splits_1 == splits_2
