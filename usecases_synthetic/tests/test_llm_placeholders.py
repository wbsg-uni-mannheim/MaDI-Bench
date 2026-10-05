"""LLM placeholder strings become missing values in K2's synthetic records."""
from usecases_synthetic.lib.llm_placeholders import is_placeholder, sanitize_llm_record


def test_placeholders_become_none_and_values_stay() -> None:
    rec = {"title": "WD Blue 1TB", "vram_gb": "N/A", "storage_gb": "NA", "color": "null", "brand": " Unknown ",
           "price": 49.99, "tracks": ["Intro", "N/A", "Outro"], "note": "None", "empty": ""}
    out = sanitize_llm_record(rec)
    assert out["title"] == "WD Blue 1TB" and out["price"] == 49.99
    assert out["vram_gb"] is None and out["storage_gb"] is None and out["color"] is None and out["brand"] is None
    assert out["tracks"] == ["Intro", "Outro"]
    assert out["note"] == "None"          # papers' real records use the string "None"
    assert out["empty"] is None
    assert rec["vram_gb"] == "N/A"         # the input is not mutated


def test_is_placeholder_is_strict() -> None:
    assert is_placeholder("n/a") and is_placeholder(" NULL ")
    assert not is_placeholder("NAS") and not is_placeholder("None") and not is_placeholder(0)
