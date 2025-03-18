from fastapi import FastAPI
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler
import requests
import json
import numpy as np
import ollama

app = FastAPI()


# Load ML models based on sector and gas
def load_model(sector: str, gas: str):
    model_path = f"Models/{sector}/{gas}.h5"
    return tf.keras.models.load_model(model_path)


# Define request body
class PredictionRequest(BaseModel):
    sector: str
    data: Dict[str, Any]


@app.post("/predict")
def predict_emissions(request: PredictionRequest):
    try:

        start_time = request.data.pop("start_time") # NTS: Future versions need to handle if the CSV use other column names for the dates
        df = pd.DataFrame(request.data, index=start_time)

        gases = df.columns  # Retrieve the list of gases present in the CSV; some CSVs don't have them all
        forecasts = {}

        for gas in gases:
            # loading the corresponding model
            try:

                model = load_model(request.sector, gas)
                scaler = MinMaxScaler(feature_range=(0, 1))
                values = np.array(df[[gas]].values).reshape(-1, 1)
                scaled_values = scaler.fit_transform(values) # converting the values to np array
                predictions = model.predict(scaled_values)
                forecasts[gas] = scaler.inverse_transform(predictions).flatten().tolist()

            except Exception as e: # when the input csv has a gas that we dont have a model for, we skip.
                continue

        forecasts_formatted = json.dumps(forecasts, indent=2)

        sector_name = request.sector

        # Generate suggestions using Deepseek-R1 locally using OpenRouter (shit was free)
        prompt = f"""
You are an AI analyst. Analyze the **PREDICTED** emissions per day for various gases in metric tonnes.  
Compare them with **Indian government regulations** and **sectoral limits** (search reliable sources).  

If **any gas exceeds permissible limits**, provide **3-5 actionable suggestions** to reduce emissions **without affecting business profitability**.  
If all emissions are within limits, return **no suggestions needed**.  

Your response MUST be in following format:
```
    **Conclusion**: 
    - "point 1"
    - "point 2"
    \n
    **Recommendations**: 
    - "step 1" 
    - "step 2" 
    - "step 3"
```

STRICT RULES:
- DO NOT include extra text before Conclusions or after Recommendations.
- DO NOT explain reasoning.
- DO NOT include speculative statements.

**Sector:** {sector_name}  
**Emission Forecasts:**  
{forecasts_formatted}  
"""


        sugg_req = ollama.chat(model="phi3.5", messages=[{"role": "user", "content": prompt}])
        suggestions = sugg_req.get("message", "Failed to generate suggestions. Try again later.")

        return {"forecasts": forecasts, "suggestions": suggestions}

    except Exception as e:
        return {"error": str(e)}
