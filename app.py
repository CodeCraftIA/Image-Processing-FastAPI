import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Initialize FastAPI and templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load pre-trained models (MobileNet for classification, Haarcascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mobilenet_model = MobileNet(weights='imagenet')

# Helper function to save processed images
def save_image(img, path):
    cv2.imwrite(path, img)

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    # Render the upload page
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # Read image from upload
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "Error: Image not found or unable to load"

    # Make a copy of the original image for each task
    original_img = img.copy()

    # ---- Image Segmentation ----
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)  # Use the original image here
    _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    save_image(segmented, "static/segmented_image.png")

    # ---- Face Detection ----
    face_img = original_img.copy()  # Copy original image for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    save_image(face_img, "static/face_detection.png")

    # ---- MobileNet Image Classification ----
    # Resize and preprocess the original image
    img_resized = cv2.resize(original_img, (224, 224))  # Resize to MobileNet input size (224, 224)
    img_array = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for MobileNet
    preds = mobilenet_model.predict(img_array)
    prediction_results = decode_predictions(preds, top=3)[0]  # Get top 3 predictions

    # ---- Background Removal (GrabCut) ----
    bg_removal_img = original_img.copy()  # Copy the original image for background removal
    mask = np.zeros(bg_removal_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, bg_removal_img.shape[1] - 50, bg_removal_img.shape[0] - 50)
    cv2.grabCut(bg_removal_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_bg_removed = bg_removal_img * mask2[:, :, np.newaxis]
    save_image(img_bg_removed, "static/background_removed.png")

    # ---- Contour Detection ----
    contour_img = original_img.copy()  # Copy the original image for contour detection
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
    save_image(contour_img, "static/contour_image.png")

    # Render the result page with processed images and predictions
    return templates.TemplateResponse("result.html", {
        "request": request,
        "segmented_image_url": "static/segmented_image.png",
        "face_detection_url": "static/face_detection.png",
        "background_removed_url": "static/background_removed.png",
        "contour_image_url": "static/contour_image.png",
        "predictions": prediction_results,  # Pass top 3 predictions
    })

