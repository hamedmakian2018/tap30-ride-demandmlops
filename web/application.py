import os
import threading
import time
import webbrowser
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, conint

from src.config_reader import read_config

# Read configuration
config_path = "config/config.yaml"
web_config = read_config(config_path)["web"]

model_dir = web_config["model_output_dir"]
model_name = web_config["model_name"]

# Load model
model = joblib.load(Path(model_dir) / model_name)

# Initialize FastAPI app
app = FastAPI()


@app.on_event("startup")
def startup_message():
    print(
        "\n✅ Server is running. Please open http://127.0.0.1:8080 in your browser.\n"
    )
    print("\nOr click on above link.\n")


# Request schema
class DemandRequest(BaseModel):
    hour_of_day: conint(ge=0, le=23)
    day: conint(ge=0)
    row: conint(ge=0, le=7)
    col: conint(ge=0, le=7)


# Response schema
class DemandResponse(BaseModel):
    demand: int


# Homepage with button to docs
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Ride Demand Predictor</title>
        </head>
        <body style="font-family: Arial; text-align: center; padding-top: 100px;">
            <h1>Welcome to the Ride Demand Prediction API 🚗</h1>
            <p>Click the button below to access the API documentation.</p>
            <a href="/docs">
                <button style="padding: 10px 20px; font-size: 16px;">Go to Docs</button>
            </a>
        </body>
    </html>
    """


# Prediction endpoint
@app.post("/predict", response_model=DemandResponse)
def predict_demand(request: DemandRequest):
    features = pd.DataFrame(
        [
            {
                "hour_of_day": request.hour_of_day,
                "day": request.day,
                "row": request.row,
                "col": request.col,
            }
        ]
    )
    prediction = model.predict(features)[0]
    return {"demand": round(prediction)}


# Server entry point


def open_browser():
    time.sleep(1)  # Let the server start up
    url = f"http://{os.environ.get('WEB_HOST', web_config['host'])}:{os.environ.get('WEB_PORT', web_config['port'])}"
    webbrowser.open(url)


if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    uvicorn.run(
        "web.application:app",
        host=os.environ.get("WEB_HOST", web_config["host"]),
        port=int(os.environ.get("WEB_PORT", web_config["port"])),
        reload=True,
    )
