"""Lightweight severity estimation helper."""

def estimate_severity(confidence: float) -> dict:
    # Confidence near 1 means the model is very sure. Lower confidence -> more uncertainty.
    score = max(0.0, min(1.0, 1.0 - confidence)) * 100.0

    if score < 30.0:
        label = "Mild"
        color = "#4CAF50"  # green
    elif score < 60.0:
        label = "Moderate"
        color = "#FF9800"  # orange
    else:
        label = "Severe"
        color = "#F44336"  # red

    return {
        "severity": label,
        "infected_pct": round(score, 1),
        "color": color,
    }
