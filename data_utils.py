import pandas as pd
import json

def get_demo_patient_cases():
    """
    Returns a list of deliberately tricky "trap" cases designed to test the 
    Cognitive Swarm's ability to overcome Anchoring and Availability biases.
    """
    return [
        {
            "id": "PT-TRAP-01",
            "name": "Case 1: The 'Anxiety' Trap",
            "notes": "Patient is a 24-year-old female presenting with chest tightness, heart palpitations, and extreme feeling of dread. Has a history of generalized anxiety disorder (GAD). Vitals show HR 120, Temp 101.2F, BP 100/60. She mentions she had a 'stomach bug' with muscle aches 2 weeks ago."
            # Trap: The physician/LLM anchors on "Anxiety" and young age, ignoring fever/viral prodrome which points to life-threatening Myocarditis.
        },
        {
            "id": "PT-TRAP-02",
            "name": "Case 2: The 'Food Poisoning' Trap",
            "notes": "45-year-old male with severe nausea, vomiting, and epigastric pain that started 4 hours ago. Recently ate at a sketchy seafood restaurant. Pulse 110. Patient is sweating profusely and complains that the pain occasionally radiates to his jaw/left arm."
            # Trap: Anchoring on the seafood restaurant (availability heuristic) -> Food poisoning, ignoring the classic signs of Myocardial Infarction (heart attack).
        },
        {
            "id": "PT-TEST-03",
            "name": "Case 3: Base Rate Reality",
            "notes": "60-year-old male complains of generalized fatigue, frequent urination, and feeling thirsty all the time. BMI is 32. History of hypertension."
            # Solid standard case (Type 2 Diabetes). Tests if the system gives a straightforward diagnosis when no bias triggers are present.
        }
    ]

def parse_agent_json(json_str):
    try:
        # Sometimes LLMs wrap JSON in markdown blocks
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Failed to parse JSON", "rawOutput": json_str}
