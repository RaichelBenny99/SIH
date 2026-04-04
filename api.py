from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import os

from services.treatment import get_treatment
from services.weather import get_weather
from services.severity import estimate_severity
from services.gemini import ask_agronomist

MODEL_PATH = "plant_disease_model.pth"
CLASSES_FILE = "ML_Models/PlantVillage/classes.txt"
NUM_CLASSES = 39

app = FastAPI(title="Plant Disease AgTech API")

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None
_class_names = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file not found: " + MODEL_PATH)

        m = models.efficientnet_b0(weights=None)
        num_ftrs = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        m.eval()
        _model = m
    return _model


def get_class_names():
    global _class_names
    if _class_names is None:
        if not os.path.exists(CLASSES_FILE):
            raise FileNotFoundError("classes.txt not found")
        with open(CLASSES_FILE, "r") as f:
            _class_names = [line.strip() for line in f.readlines()]
    return _class_names


def compute_entropy(probs: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.sum(probs * np.log(probs + eps)))


@app.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")

    model = get_model()
    class_names = get_class_names()

    image_tensor = _transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        raw_probs = torch.softmax(outputs[0], dim=-1).cpu().numpy()
        pred_idx = int(np.argmax(raw_probs))
        raw_conf = float(raw_probs[pred_idx])

    entropy = compute_entropy(raw_probs)
    ood = (raw_conf < 0.55) or (entropy > 1.8)
    if ood:
        return {
            "status": "unknown",
            "message": "Image is not a recognizable plant leaf",
            "entropy": entropy,
            "raw_confidence": raw_conf,
        }

    disease = class_names[pred_idx]
    severity = estimate_severity(raw_conf)
    treatment = get_treatment(disease)

    return {
        "status": "ok",
        "prediction": disease,
        "raw_confidence": raw_conf,
        "severity": severity,
        "treatment": treatment,
        "entropy": entropy,
    }


class ChatRequest(BaseModel):
    disease: str
    severity: str
    question: str
    location: dict


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    lat = req.location.get("lat")
    lon = req.location.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="location must include lat and lon")

    weather_data = get_weather(float(lat), float(lon))
    severity_obj = {"severity_label": req.severity, "severity_score": None}

    gemini_res = ask_agronomist(
        disease=req.disease,
        severity=severity_obj,
        weather=weather_data,
        user_question=req.question,
    )

    return {
        "disease": req.disease,
        "severity": req.severity,
        "weather": weather_data,
        "gemini": gemini_res,
    }
