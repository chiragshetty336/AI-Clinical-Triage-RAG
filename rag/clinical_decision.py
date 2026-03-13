def admission_decision(triage_level):

    if triage_level == "RED":

        return {
            "admission": "ICU",
            "priority": "Immediate life-saving intervention required",
            "action": "Stabilize airway, breathing, circulation immediately",
        }

    elif triage_level == "YELLOW":

        return {
            "admission": "Emergency Ward",
            "priority": "Urgent medical attention required",
            "action": "Monitor patient closely and start treatment",
        }

    else:

        return {
            "admission": "Outpatient / Observation",
            "priority": "Minor condition",
            "action": "Basic treatment and discharge if stable",
        }
