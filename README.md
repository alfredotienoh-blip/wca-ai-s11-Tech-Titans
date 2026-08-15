import os
import json
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

## Define structured json schema for the input data

class FeedbackAnalysisSchema(BaseModel):
    sentiment: str = Field(
        description="Must be strictly one of: Positive, Neutral, Negative"
    )
    sentiment_score: float = Field(
        description="Confidence scale from 0.0 (low) to 1.0 (high)"
    )
    category: str = Field(
        description="Primary bucket: Bug, Feature Request, Pricing, Customer Support, UI/UX, Other"
    )
    summary: str = Field(
        description="One sentence summarizing the user's issue or praise."
    )
    tags: List[str] = Field(
        description="Extracted keywords for technical filters (e.g., 'checkout', 'latency', 'billing')"
    )
    urgent_action_required: bool = Field(
        description="True if customer is highly frustrated, experiencing an outage, or threatening churn."
    )

## Raw structured feedback(Stage 1 data)

STAGE_1_FEEDBACK = [
    {
        "feedback_id": "cust_901",
        "text": "The app keeps freezing up at the payment gateway screen. It took my balance but didn't book my ticket! Get back to me ASAP."
    },
    {
        "feedback_id": "cust_902",
        "text": "Honestly, the onboarding wizard is so much faster now. Great design cleanup on the text sizing."
    },
    {
        "feedback_id": "cust_903",
        "text": "Are there any plans to add an export to PDF option for monthly reports? CSV works fine but clients want visuals."
    }
]

## Gemini APA engines and orchestraton

def run_stage_1_analysis():
    # Fetch key from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing environment variable: GEMINI_API_KEY")

   ## Initialize the modern, official client
   client = genai.Client(api_key=api_key)
    
   ## Using 'gemini-2.5-flash' for optimal latency and processing cost
   model_name = "gemini-2.5-flash"
    
   system_instruction = (
        "You are an automated Customer Operations Triaging Agent. Your task is to process "
        "incoming text feedback and strictly populate the requested structured JSON data blueprint."
    )

  print(f"🚀 Initializing Analysis Pipeline via {model_name}...")
    print(f"📥 Loaded {len(STAGE_1_FEEDBACK)} items for Stage 1 evaluation.\n" + "="*50)

   for item in STAGE_1_FEEDBACK:
        print(f"\n[Processing ID: {item['feedback_id']}]")
        print(f"Raw Input: \"{item['text']}\"")
        
   try:
            # Generate contents using structural constraints
            response = client.models.generate_content(
                model=model_name,
                contents=f"Analyze this customer feedback instance: {item['text']}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,  # Low temperature ensures deterministic sorting
                    response_mime_type="application/json",
                    response_schema=FeedbackAnalysisSchema,
                ),
            )
            
   ## The SDK handles serialization back into Pydantic validation fields automatically
   structured_data: FeedbackAnalysisSchema = response.parsed
            
   ## Form final merged payload
   output_payload = {
                "metadata": {
                    "feedback_id": item["feedback_id"]
                },
                "analysis_results": structured_data.model_dump()
            }
            
   ## Print the clean, valid JSON output string
   print("✨ Parsed JSON Output:")
            print(json.dumps(output_payload, indent=2))
            
  ## Trigger conditional actions based on structure
  if structured_data.urgent_action_required:
                print("🚨 ALERT: Immediate operational routing flagged for this issue!")
                
  except Exception as e:
            print(f"❌ Failed to parse data for ID {item['feedback_id']}. Error details: {e}")

if __name__ == "__main__":
    run_stage_1_analysis()
