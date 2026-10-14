"""Unit tests for recovery recommendation engine."""

import pytest
from recovery_engine import (
    DEFAULT_RULE,
    RECOVERY_RULES,
    get_recovery_info,
    normalize_class_name,
)


def test_normalize_class_name():
    """Verify normalization logic for labels."""
    assert normalize_class_name("Plastic") == "plastic"
    assert normalize_class_name(" Cardboard ") == "cardboard"
    assert normalize_class_name("SAFE DISPOSAL") == "safe_disposal"
    assert normalize_class_name(None) == "unknown"
    assert normalize_class_name("") == "unknown"
    assert normalize_class_name(123) == "unknown"


def test_get_recovery_info_valid_classes():
    """Verify recovery information for each supported category."""
    expected_classes = ["cardboard", "glass", "metal", "organic", "paper", "plastic", "trash"]

    for class_name in expected_classes:
        info = get_recovery_info(class_name, weight_kg=2.0)
        assert info["class_name"] == class_name
        assert "recovery_method" in info
        assert "bin_color" in info
        assert isinstance(info["process_steps"], list)
        assert len(info["process_steps"]) > 0
        assert info["weight_kg"] == 2.0
        assert info["estimated_value"] == round(2.0 * info["recovery_value_per_kg"], 2)
        assert info["estimated_co2_saved"] == round(2.0 * info["co2_saved_per_kg"], 2)


def test_get_recovery_info_unknown_class():
    """Verify fallback to DEFAULT_RULE for unknown waste."""
    info = get_recovery_info("unknown_item", weight_kg=1.5)
    assert info["recovery_method"] == DEFAULT_RULE["recovery_method"]
    assert info["bin_color"] == DEFAULT_RULE["bin_color"]
    assert info["estimated_value"] == round(1.5 * DEFAULT_RULE["recovery_value_per_kg"], 2)


def test_get_recovery_info_invalid_weights():
    """Verify handling of invalid or negative weights."""
    with pytest.raises(ValueError, match="Weight must be non-negative"):
        get_recovery_info("plastic", weight_kg=-1.0)

    # String weight should be coerced safely
    info = get_recovery_info("plastic", weight_kg="3.5")
    assert info["weight_kg"] == 3.5

    # Invalid non-numeric string should fallback to 1.0 kg
    info = get_recovery_info("plastic", weight_kg="invalid")
    assert info["weight_kg"] == 1.0


def test_get_recovery_info_zero_weight():
    """Verify calculations for zero weight."""
    info = get_recovery_info("glass", weight_kg=0.0)
    assert info["estimated_value"] == 0.0
    assert info["estimated_co2_saved"] == 0.0
