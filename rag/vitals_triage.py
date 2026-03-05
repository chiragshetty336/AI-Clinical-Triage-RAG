def calculate_vital_triage(
    heart_rate=None, oxygen=None, temperature=None, systolic_bp=None
):

    score = 0

    # Oxygen level
    if oxygen is not None:
        if oxygen < 90:
            score += 3
        elif oxygen < 94:
            score += 2

    # Heart rate
    if heart_rate is not None:
        if heart_rate > 130:
            score += 3
        elif heart_rate > 110:
            score += 2

    # Temperature
    if temperature is not None:
        if temperature > 39:
            score += 2
        elif temperature > 38:
            score += 1

    # Blood pressure
    if systolic_bp is not None:
        if systolic_bp < 90:
            score += 3
        elif systolic_bp < 100:
            score += 2

    # Convert score → triage level
    if score >= 6:
        return "RED"

    elif score >= 3:
        return "YELLOW"

    else:
        return "GREEN"
