# Project Review README (What is where, and what each part does)

This file is made for your viva/review so you can quickly explain:
- what this project is about,
- where each important file is,
- how training and inference happen,
- and what each major code block/line section does.

---

## 1) What this project is about

This is an **AI-based crop health system** with two pipelines:

1. **Plant disease classification (PlantVillage)**
   - Model: EfficientNet-B0 (PyTorch)
   - Output: disease class (39 classes), confidence, Grad-CAM explanation, severity, and treatment advice.

2. **Insect/pest classification & detection (IP102)**
   - ViT-based image classifier (102 insect classes)
   - YOLOv8 training script for detection experiments.

So the project is not only “predict disease name”, but also:
- explain model focus (Grad-CAM),
- check image quality before prediction,
- calibrate confidence using temperature scaling,
- estimate disease severity from heatmap,
- and give treatment/pesticide guidance.

---

## 2) High-level folder map (where what lies)

### Root level
- `requirements.txt`
  - Global Python dependencies used by app/model utilities.
- `Research_Paper_Content.md`
  - Full write-up of the research motivation, architecture, and modules.

### Plant disease production pipeline
- `ML_Models/PlantVillage/`
  - Main production-ready code for the PlantVillage system (training + Streamlit app + explainability + utilities).

Key files inside:
- `ML_Models/PlantVillage/plantVillage.py`
  - **Training script** for EfficientNet-B0 on PlantVillage dataset.
- `ML_Models/PlantVillage/streamlit_app.py`
  - **Main deployed app** (prediction + Grad-CAM + quality checks + severity + treatments).
- `ML_Models/PlantVillage/plant_disease_model.pth`
  - Saved trained model weights.
- `ML_Models/PlantVillage/classes.txt`
  - 39 disease class labels used in prediction output.
- `ML_Models/PlantVillage/gradcam.py`
  - Explainability module (visual attention heatmap).
- `ML_Models/PlantVillage/image_quality.py`
  - Blur/brightness/resolution checks before inference.
- `ML_Models/PlantVillage/severity_estimator.py`
  - Converts Grad-CAM activation area to Mild/Moderate/Severe.
- `ML_Models/PlantVillage/temperature_scaling.py`
  - Confidence calibration (learns temperature `T`).
- `ML_Models/PlantVillage/export_logits.py`
  - Exports validation logits/labels for calibration.
- `ML_Models/PlantVillage/robustness_test.py`
  - Evaluates model under blur/noise/low-light/rotation perturbations.
- `ML_Models/PlantVillage/treatment_info.py`
  - Disease-to-description/treatment/pesticide mapping.

### Additional experiments / alternate apps
- `Model/PlantVillage/`
  - Alternate copy of PlantVillage scripts, including a simpler Streamlit app (`app2.py`).

### IP102 insect branch
- `Model/IP102/`
  - Insect classification/detection work.
- `Model/IP102/ViT/trainModel.py`
  - Script used to train YOLOv8 model for insect detection experiments.
- `Model/IP102/ViT/app.py`
  - Streamlit app for insect classifier inference using saved ViT weights.
- `Model/IP102/classes.txt`
  - Insect class list (102 classes; file shown starts with classes like rice leaf roller).
- `Model/IP102/dataset/`
  - Dataset split text files and YOLO-format labels.

---

## 3) Where training happens (exact answer for review)

### Plant disease training
- File: `ML_Models/PlantVillage/plantVillage.py`
- Dataset source in code: `DATA_DIR = "./Plant_leave_diseases_dataset_with_augmentation"`
- Train/val split: 80/20 using `random_split`
- Data augmentation (train):
  - RandomResizedCrop, HorizontalFlip, Rotation, normalization
- Validation transform:
  - Resize, CenterCrop, normalization
- Model:
  - `models.efficientnet_b0(weights=...)`
  - Feature extractor frozen (`param.requires_grad = False`)
  - Final classifier replaced with `Linear(num_ftrs, num_classes)`
- Loss/optimizer:
  - CrossEntropyLoss
  - Adam on classifier parameters only
- Save checkpoint:
  - `torch.save(model.state_dict(), MODEL_SAVE_PATH)`

### Insect/detection training
- File: `Model/IP102/ViT/trainModel.py`
- Uses Ultralytics YOLOv8 (`YOLO("yolov8m.pt")`)
- Trains with config `plant_yolo.yaml` for ~140 epochs.

---

## 4) Where inference/prediction happens

### Main plant app (most important for demo)
- File: `ML_Models/PlantVillage/streamlit_app.py`
- Flow:
  1. Load model + class names
  2. Upload image in Streamlit
  3. Run image quality checks
  4. Transform image and run model forward pass
  5. Compute softmax confidence (raw + calibrated)
  6. Generate Grad-CAM overlay
  7. Estimate severity from heatmap
  8. Fetch treatment and pesticide recommendation
  9. Display all outputs in UI

### Alternate plant app
- File: `Model/PlantVillage/app2.py`
- Simpler version of upload → classify → confidence display.

### Insect app
- File: `Model/IP102/ViT/app.py`
- Loads saved model (`vit_best.pth`) and predicts insect class from uploaded image.

---

## 5) Explainability, reliability, and safety modules

### Grad-CAM explainability
- File: `ML_Models/PlantVillage/gradcam.py`
- Core logic:
  - Hook final conv layer activations and gradients
  - Backprop target class score
  - Weight channels by pooled gradients
  - Build normalized heatmap and overlay on original image
- Why it matters:
  - Shows *where* model looked before making the decision.

### Image quality gate
- File: `ML_Models/PlantVillage/image_quality.py`
- Checks:
  - blur score (Laplacian variance)
  - brightness range
  - minimum resolution
- Why it matters:
  - Prevents unreliable predictions from bad images.

### Severity estimation
- File: `ML_Models/PlantVillage/severity_estimator.py`
- Logic:
  - threshold heatmap activation
  - compute infected area percentage
  - map to severity bands:
    - <15% Mild
    - 15–40% Moderate
    - >40% Severe

### Confidence calibration
- File: `ML_Models/PlantVillage/temperature_scaling.py`
- Learns single temperature `T` on validation logits
- Uses calibrated softmax: `softmax(logits / T)`
- Why it matters:
  - Confidence values become more trustworthy.

### Calibration data export
- File: `ML_Models/PlantVillage/export_logits.py`
- Purpose:
  - Runs validation set, saves:
    - `val_logits.npy`
    - `val_labels.npy`
  - Then can fit/save `temperature.pth`.

### Robustness testing
- File: `ML_Models/PlantVillage/robustness_test.py`
- Tests model under perturbations:
  - blur, low light, noise, rotation
- Reports accuracy by condition.

---

## 6) “Every line” explanation strategy for your viva

Your teacher may ask line-by-line meaning. Use this method:

1. **Imports block**
   - Explain each library role (PyTorch, torchvision, PIL, Streamlit, numpy, cv2).
2. **Config block**
   - Explain hardcoded constants/paths and why they are centralized.
3. **Transforms block**
   - Explain preprocessing and augmentation; mention match between training and validation/inference transforms.
4. **Model definition block**
   - Explain transfer learning + replacing classifier head.
5. **Training/inference function block**
   - Walk through forward pass, loss, backward, optimizer step, or prediction and softmax.
6. **Post-processing block**
   - Explain confidence, Grad-CAM, severity, treatment lookup.
7. **UI block (Streamlit)**
   - Explain uploader, button trigger, result rendering.

If asked “why this line exists?”, connect it to one of:
- correctness,
- performance,
- reliability,
- explainability,
- user usability.

---

## 7) Quick viva script (short speaking points)

- “Our core model is EfficientNet-B0 trained on PlantVillage 39 classes.”
- “Training code is in `ML_Models/PlantVillage/plantVillage.py` with transfer learning and 80/20 split.”
- “The deployed inference app is `ML_Models/PlantVillage/streamlit_app.py`.”
- “We added explainable AI using `gradcam.py` so users can see model attention regions.”
- “We added confidence calibration through temperature scaling for more reliable probabilities.”
- “We estimate severity from Grad-CAM activation ratio and map to mild/moderate/severe.”
- “`treatment_info.py` maps each predicted disease to practical treatment and pesticide guidance.”
- “There is also an IP102 insect branch with separate training/app scripts under `Model/IP102`.”

---

## 8) Important notes to say honestly (if asked)

- There are **duplicate/alternate scripts** in both `ML_Models/PlantVillage` and `Model/PlantVillage`; main enhanced app is in `ML_Models/PlantVillage/streamlit_app.py`.
- `Model/IP102/ViT/trainModel.py` currently trains YOLO (despite folder name suggesting ViT); mention this as experimental branch naming inconsistency.
- Dataset folder path is expected locally as `Plant_leave_diseases_dataset_with_augmentation`; this path may need local adjustment on another machine.

---

## 9) How to run (quick commands)

From project root:

1. Install dependencies
   - `pip install -r requirements.txt`

2. Run plant disease app (enhanced)
   - `cd ML_Models/PlantVillage`
   - `streamlit run streamlit_app.py`

3. Run alternate simple app
   - `cd Model/PlantVillage`
   - `streamlit run app2.py`

4. Run robustness test
   - `cd ML_Models/PlantVillage`
   - `python robustness_test.py --data_dir ./Plant_leave_diseases_dataset_with_augmentation --model_path plant_disease_model.pth --num_samples 500`

---

## 10) One-line summary

This project is a **complete plant health AI pipeline**: train a disease classifier, run explainable + calibrated inference, estimate severity, and provide actionable treatment recommendations, with a parallel insect-detection research branch.
