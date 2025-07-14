import os
from openai import OpenAI
import json
from typing import Dict, Any

class SOAPSummaryGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
    def generate_soap_summary(self, text: str) -> Dict[str, str]:
        """
        Generate SOAP (Subjective, Objective, Assessment, Plan) summary using AI.
        
        Args:
            text: The journal entry text to summarize
            
        Returns:
            Dictionary containing SOAP components
        """
        prompt = f"""Analyze the following journal entry from a patient and generate a high-level, clinically interpretable SOAP summary. The summary should be useful for a clinician to quickly understand the patient's state. Focus on extracting key psychological signals and potential areas of concern.

**Journal Entry:**
{text}

**Instructions:**
Format the output as a JSON object with four keys: 'subjective', 'objective', 'assessment', and 'plan'.
- **Subjective:** Capture the patient's reported feelings, symptoms, and direct quotes about their experience (e.g., 'feeling overwhelmed', 'sleeping poorly').
- **Objective:** Identify observable behaviors or patterns mentioned in the text (e.g., 'reports crying spells', 'mentions social withdrawal'). If none, state 'No objective data mentioned'.
- **Assessment:** Provide a brief clinical impression based on the subjective and objective data. Mention potential themes like mood disturbances, anxiety, or thought pattern irregularities.
- **Plan:** Suggest potential next steps for the clinician, such as areas to explore in the next session, or recommendations for the patient (e.g., 'Monitor mood and sleep patterns', 'Explore coping mechanisms for anxiety').

**Example JSON Output:**
{{
    "subjective": "Reports feeling 'constantly on edge' and experiencing trouble concentrating.",
    "objective": "Mentions staying up late and avoiding social events.",
    "assessment": "The entry suggests potential anxiety and depressive symptoms, with possible sleep cycle disruption.",
    "plan": "In the next session, explore the triggers for anxiety and assess the severity of depressive symptoms. Recommend practicing mindfulness exercises."
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="mistralai/mistral-7b-instruct-v0.2", # Using a slightly updated model
                messages=[
                    {"role": "system", "content": "You are an expert clinical assistant AI. Your task is to generate a clinically relevant SOAP summary from a patient's journal entry. Follow the user's JSON format instructions precisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400, # Increased tokens for better quality summary
                temperature=0.6 # Adjusted for a balance of creativity and consistency
            )
            
            # Parse the JSON response
            # The response content might be a string that needs parsing.
            response_text = response.choices[0].message.content
            # Find the start of the JSON object
            json_start = response_text.find('{')
            if json_start != -1:
                response_text = response_text[json_start:]
            
            summary = json.loads(response_text)
            return summary
            
        except Exception as e:
            print(f"Error generating SOAP summary: {str(e)}")
            return {
                "subjective": "Error generating summary",
                "objective": "Error generating summary",
                "assessment": "Error generating summary",
                "plan": "Error generating summary"
            }

    def process_journal_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a journal entry and add SOAP summary.
        
        Args:
            entry: Dictionary containing journal entry data
            
        Returns:
            Updated entry with SOAP summary
        """
        if "text" in entry:
            soap_summary = self.generate_soap_summary(entry["text"])
            entry["soap_summary"] = soap_summary
        return entry
