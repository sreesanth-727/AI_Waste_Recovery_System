"""Recovery recommendation engine for waste-to-recovery classification."""

from __future__ import annotations

from typing import Dict, List

RECOVERY_RULES: Dict[str, Dict[str, object]] = {
    "cardboard": {
        "recovery_method": "Recycle",
        "process_steps": [
            "Collect clean cardboard boxes and remove tape or plastic layers.",
            "Shred and bale the material for recycling facilities.",
            "Send to paper mills for pulping and remanufacturing.",
        ],
        "recovery_value_per_kg": 11.0,
        "co2_saved_per_kg": 1.2,
        "bin_color": "Blue",
    },
    "glass": {
        "recovery_method": "Recycle",
        "process_steps": [
            "Separate broken or clear glass from other waste streams.",
            "Crush and sort by color before melting.",
            "Use in glass production or construction material after cleaning.",
        ],
        "recovery_value_per_kg": 16.0,
        "co2_saved_per_kg": 1.5,
        "bin_color": "Green",
    },
    "metal": {
        "recovery_method": "Recycle",
        "process_steps": [
            "Sort ferrous and non-ferrous metals separately.",
            "Compress and transport to scrap recycling facilities.",
            "Smelt and reshape into new metal products.",
        ],
        "recovery_value_per_kg": 42.0,
        "co2_saved_per_kg": 2.8,
        "bin_color": "Yellow",
    },
    "paper": {
        "recovery_method": "Recycle",
        "process_steps": [
            "Remove contaminants, staples, and heavy coating.",
            "Pulverize and mix with water to form slurry.",
            "Dry and press into recycled paper sheets.",
        ],
        "recovery_value_per_kg": 14.0,
        "co2_saved_per_kg": 1.3,
        "bin_color": "Blue",
    },
    "plastic": {
        "recovery_method": "Recycle",
        "process_steps": [
            "Wash and dry the plastic to remove residues.",
            "Sort by resin type such as PET, HDPE, or PP.",
            "Shred, melt, and pelletize for reuse in manufacturing.",
        ],
        "recovery_value_per_kg": 18.0,
        "co2_saved_per_kg": 1.8,
        "bin_color": "Orange",
    },
    "organic": {
        "recovery_method": "Compost",
        "process_steps": [
            "Mix food scraps and garden waste with dry carbon-rich material.",
            "Maintain moisture, aeration, and temperature for composting.",
            "Use the compost as soil enrichment for agriculture or landscaping.",
        ],
        "recovery_value_per_kg": 9.0,
        "co2_saved_per_kg": 2.2,
        "bin_color": "Brown",
    },
    "trash": {
        "recovery_method": "Safe Disposal",
        "process_steps": [
            "Inspect for hazardous or contaminated material.",
            "Segregate non-recyclable waste before disposal.",
            "Send to landfill or approved waste treatment facilities.",
        ],
        "recovery_value_per_kg": 1.0,
        "co2_saved_per_kg": 0.2,
        "bin_color": "Grey",
    },
}

DEFAULT_RULE = {
    "recovery_method": "Safe Disposal",
    "process_steps": [
        "Classify the item manually if the AI result is uncertain.",
        "Store in a safe waste stream and avoid mixing with recyclables.",
        "Send the material to a licensed disposal or treatment facility.",
    ],
    "recovery_value_per_kg": 0.5,
    "co2_saved_per_kg": 0.05,
    "bin_color": "Grey",
}


def normalize_class_name(class_name: str | None) -> str:
    """Normalize class labels to ensure consistent lookup."""
    if not class_name or not isinstance(class_name, str):
        return "unknown"
    return class_name.strip().lower().replace(" ", "_")


def get_recovery_info(class_name: str | None, weight_kg: float = 1.0) -> Dict[str, object]:
    """Return the recovery guidance and financial/environmental value for a waste item.

    Parameters
    ----------
    class_name : str or None
        Waste class predicted by the model.
    weight_kg : float
        Estimated item weight in kilograms.

    Returns
    -------
    Dict[str, object]
        Recovery method, steps, per-kg rates, and estimated total value.
    """
    try:
        weight_kg = float(weight_kg)
    except (ValueError, TypeError):
        weight_kg = 1.0

    if weight_kg < 0:
        raise ValueError("Weight must be non-negative.")

    normalized_name = normalize_class_name(class_name)
    rule = RECOVERY_RULES.get(normalized_name, DEFAULT_RULE)

    recovery_value_per_kg = float(rule.get("recovery_value_per_kg", 0.0))
    co2_saved_per_kg = float(rule.get("co2_saved_per_kg", 0.0))

    estimated_value = round(weight_kg * recovery_value_per_kg, 2)
    estimated_co2_saved = round(weight_kg * co2_saved_per_kg, 2)

    return {
        "class_name": normalized_name,
        "recovery_method": rule.get("recovery_method", "Safe Disposal"),
        "process_steps": list(rule.get("process_steps", [])),
        "recovery_value_per_kg": recovery_value_per_kg,
        "co2_saved_per_kg": co2_saved_per_kg,
        "bin_color": rule.get("bin_color", "Grey"),
        "weight_kg": round(float(weight_kg), 2),
        "estimated_value": estimated_value,
        "estimated_co2_saved": estimated_co2_saved,
    }


if __name__ == "__main__":
    sample = get_recovery_info("plastic", 2.5)
    print(sample)
