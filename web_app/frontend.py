import json
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from io import StringIO

# Streamlit UI Setup
st.set_page_config(page_title="Emission Forecasts", layout="wide")
st.title("Emission Forecast Dashboard")

# File Upload & Sector Selection
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
sector = st.selectbox("Select Sector", [
    "agriculture", "buildings", "fluorinated_gases", "forestry_and_land_use",
    "fossil_fuel_operations", "manufacturing", "mineral_extraction",
    "power", "transportation", "waste"
])
submit_button = st.button("Submit")

if submit_button and uploaded_file is not None:
    # Read CSV content
    file_content = uploaded_file.getvalue().decode("utf-8")
    csv_data = pd.read_csv(StringIO(file_content))

    # Send data to backend
    response = requests.post("http://127.0.0.1:8000/predict", json={
        "sector": sector,
        "data": csv_data.to_dict(orient="list")  # Ensure correct JSON formatting
    })

    if response.status_code == 200:
        result = response.json()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Forecast Results")

            try:
                forecasts_df = pd.DataFrame(result["forecasts"])  # Already a dict, no need for json.loads()

                # Reshape data for better visualization
                forecasts_melted = forecasts_df.melt(ignore_index=False, var_name="Gas", value_name="Emissions")

                # Plotly Interactive Line Chart
                fig = px.line(
                    forecasts_melted, x=forecasts_melted.index, y="Emissions",
                    color="Gas", markers=True, labels={"index": "Day", "Emissions": "Metric Tonnes"}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in plotting data: {str(e)}")

        with col2:
            st.subheader("LLM Suggestions")
            suggestions = result.get("suggestions", "No suggestions needed.")  # Directly retrieve suggestions

            if isinstance(suggestions, str):
                st.write(suggestions)
            elif isinstance(suggestions, dict) and "content" in suggestions:
                suggestion_text = suggestions["content"]
                if "</think>" in suggestion_text:
                    suggestion_text = suggestion_text.split("</think>")[1].strip()  # Remove `<think>` section safely
                st.write(suggestion_text)
            else:
                st.write("No valid suggestions found.")


    else:
        st.error(f"Error in processing: {response.text}")  # Display backend error message
