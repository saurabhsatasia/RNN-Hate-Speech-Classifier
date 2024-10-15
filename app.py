import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from hate_speech_classifier.constants import *
from hate_speech_classifier.pipeline.train_pipeline import TrainPipeline
from hate_speech_classifier.pipeline.prediction_pipeline import PredictionPipeline
from hate_speech_classifier.exception import CustomException

text: str = "What is machine learing?"

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.run_pipeline(text)
        return text
    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)