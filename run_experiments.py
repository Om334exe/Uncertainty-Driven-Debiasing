import pandas as pd
import time
from agents import CognitiveSwarm
from data_utils import get_demo_patient_cases, parse_agent_json

def run_all_experiments():
    print("🚀 Initializing Comparative Analysis Across Multi-LLMs...")
    cases = get_demo_patient_cases()
    
    # We will test against 3 major open weights models
    models_to_test = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it" 
    ]
    
    results = []
    
    for model_name in models_to_test:
        print(f"\n==============================")
        print(f"  TESTING MODEL: {model_name} ")
        print(f"==============================")
        swarm = CognitiveSwarm(primary_model=model_name, fast_model="llama-3.1-8b-instant")
        
        for case in cases:
            print(f"--- Processing Case: {case['name']} ---")
            
            structured = swarm.run_agent_1_extractor(case['notes'])
            baseline = swarm.run_agent_2_diagnostician(structured)
            biases = swarm.run_agent_3_bias_analyzer(structured, baseline)
            rag_data = swarm.run_agent_5_epistemic_rag(structured, baseline)
            final_str = swarm.run_agent_4_tot_mitigator(structured, baseline, biases, rag_data)
            
            final_json = parse_agent_json(final_str)
            diagnoses = final_json.get("diagnoses", [])
            
            for dx in diagnoses:
                results.append({
                    "Model": model_name,
                    "Case_ID": case["id"],
                    "Case_Name": case["name"],
                    "Condition": dx.get("condition", "N/A"),
                    "Raw_Confidence": float(dx.get("raw_confidence", 0) or 0),
                    "Mitigated_Confidence": float(dx.get("mitigated_confidence", 0) or 0),
                    "Epistemic_Uncertainty": float(dx.get("epistemic_uncertainty", 0) or 0),
                    "Flagged_Bias": dx.get("flagged_bias", "None"),
                })
                
            time.sleep(3) # Prevent rate-limiting across models
        
    df = pd.DataFrame(results)
    output_file = "comparative_model_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Comparative analysis finished! Results saved to '{output_file}'")

if __name__ == "__main__":
    run_all_experiments()
