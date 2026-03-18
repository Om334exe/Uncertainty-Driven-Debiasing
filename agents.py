import os
import json
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class CognitiveSwarm:
    def __init__(self, api_key=None, primary_model="llama-3.3-70b-versatile", fast_model="llama-3.1-8b-instant"):
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            
        self.primary_model = primary_model
        self.fast_model = fast_model

    def _call_llm(self, messages, model, temperature=0.7, json_mode=False):
        try:
            kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"{{\"error\": \"GROQ API Error: {str(e)}\"}}"

    def run_agent_1_extractor(self, raw_patient_notes):
        prompt = f"""You are an expert Clinical Data Extractor. 
Extract the following information from the clinical notes and output valid JSON only:
- age (int)
- gender (string)
- symptoms (list of strings)
- vitals (dict)
- medical_history (list of strings)

Notes: {raw_patient_notes}"""
        return self._call_llm([{"role": "system", "content": prompt}], self.fast_model, temperature=0.1, json_mode=True)

    def run_agent_2_diagnostician(self, structured_data):
        prompt = f"""You are a Primary Care Physician. Review the structured patient data and provide a baseline diagnosis.
Specifically point out 1 primary hypothesis and 2 differential diagnoses.
Patient Data: {structured_data}"""
        return self._call_llm([{"role": "system", "content": prompt}], self.primary_model, temperature=0.5)

    def run_agent_3_bias_analyzer(self, structured_data, baseline_diagnosis):
        prompt = f"""You are a strict Medical Bias Analyst. Audit the physician's baseline diagnosis for cognitive biases:
1. Anchoring Bias (latching onto first symptom)
2. Availability Heuristic (diagnosing common disease, ignoring red flags)
3. Base-Rate Neglect

Patient Data: {structured_data}
Physician's Diagnosis: {baseline_diagnosis}

Describe what biases are present. If none, say 'No major biases detected'."""
        return self._call_llm([{"role": "system", "content": prompt}], self.primary_model, temperature=0.6)

    def run_agent_5_epistemic_rag(self, structured_data, baseline_diagnosis):
        """
        Agent 5: Epidemiological RAG Simulator.
        Simulates retrieving real-world literature to combat Availability Bias.
        """
        prompt = f"""You are a Medical Librarian AI (RAG System). Based on the following case, simulate retrieving 2 recent high-impact medical journal findings that either support or contradict the baseline diagnosis based on statistical disease prevalence.
Patient Data: {structured_data}
Baseline Diagnosis: {baseline_diagnosis}

Provide a short summary of 'retrieved' literature."""
        return self._call_llm([{"role": "system", "content": prompt}], self.fast_model, temperature=0.3)

    def run_agent_4_tot_mitigator(self, structured_data, baseline_diagnosis, bias_critique, rag_data):
        """
        Agent 4: Multi-Path / Tree-of-Thoughts Synthesizer with Epistemic Uncertainty.
        Evaluates multiple reasoning paths and quantifies uncertainty.
        """
        prompt = f"""You are the Chief Medical Meta-Diagnostician utilizing a Tree-of-Thoughts methodology. 
Synthesize the evidence from the initial diagnosis, the bias critique, and the literature RAG.

Patient Data: {structured_data}
Initial Diagnosis: {baseline_diagnosis}
Bias Critique: {bias_critique}
Literature (RAG): {rag_data}

Output valid JSON exactly matching this schema:
{{
  "diagnoses": [
    {{
       "condition": "Name1", 
       "raw_confidence": <0-100>, 
       "mitigated_confidence": <0-100>, 
       "epistemic_uncertainty": <0-100, representing missing data/knowledge gaps>,
       "flagged_bias": "Bias Name or None"
    }},
    {{
       "condition": "Name2", 
       "raw_confidence": <0-100>, 
       "mitigated_confidence": <0-100>, 
       "epistemic_uncertainty": <0-100>,
       "flagged_bias": "None"
    }}
  ],
  "reasoning_summary": "Explanation of your multi-path synthesis and uncertainty calculation."
}}"""
        return self._call_llm([{"role": "system", "content": prompt}], self.primary_model, temperature=0.2, json_mode=True)
