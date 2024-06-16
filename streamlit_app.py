# This file implements the Streamlit web app for the project.
# Here's how the app works:
# We first show a couple of images of the training set with and without annotation.
# Then we ask the user to upload an image.
# We then show the image with the predicted bounding boxes.


import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from roboflow import Roboflow
from PIL import Image
from PIL import ImageDraw, ImageFont
import supervision as sv
import getpass
from inference import get_model

# Get a couple of images from the training set from the roboflow universe :
# 1. An image without annotation
# 2. An image with annotation

# Get the training set

api_key = "XTbXktlKCLz22xExgzYs"
project_name = "unmonitored-ghg-emission-sources"
version_number_FC = 5
version_number_SWIR = 8

rf = Roboflow(api_key=api_key)
project = rf.workspace().project(project_name)
datasetFC = project.version(version_number_FC).download("yolov9")
datasetSWIR = project.version(version_number_SWIR).download("yolov9")


# Use supervision to plot the images with bounding boxes:
# Get the images

# The training set images are located at the following path: C:\Users\gabma\Dropbox\PC\Documents\Master\HackForThePlanet\Unmonitored-GHG-Emission-Sources-5\train\images
# The annotations are located at the following path: C:\Users\gabma\Dropbox\PC\Documents\Master\HackForThePlanet\Unmonitored-GHG-Emission-Sources-5\train\labels
# The images are in the format: <image_id>.jpg
# The annotations are in the format: <image_id>.txt
# Get the current working directory
cwd = os.getcwd()
# Define the path to the images and labels by using the current working directory

path2imagesFC = r"Unmonitored-GHG-Emission-Sources-5\train\images"
path2labelsFC = r"Unmonitored-GHG-Emission-Sources-5\train\labels"
path2imagesSWIR = r"Unmonitored-GHG-Emission-Sources-8\train\images"
path2labelsSWIR = r"Unmonitored-GHG-Emission-Sources-8\train\labels"

# Get the full path to the images and labels
path2imagesFC = os.path.join(cwd, path2imagesFC)
path2labelsFC = os.path.join(cwd, path2labelsFC)
path2imagesSWIR = os.path.join(cwd, path2imagesSWIR)
path2labelsSWIR = os.path.join(cwd, path2labelsSWIR)

print(path2imagesFC)
print(path2labelsFC)
print(path2imagesSWIR)
print(path2labelsSWIR)


# Get the list of images
imagesFC = os.listdir(path2imagesFC)
imagesSWIR = os.listdir(path2imagesSWIR)
labelsFC = os.listdir(path2labelsFC)
labelsSWIR = os.listdir(path2labelsSWIR)


# Function that draws the bounding boxes on the image from the labels (example : 0 0.5938888888888889 0.48671875 0.02666666666666667 0.04296875) :
def draw_bounding_boxes(image, labels):
    # Get the width and height of the image
    img_width, img_height = image.size
    # Get the drawing context
    draw = ImageDraw.Draw(image)
    # Draw the bounding boxes
    for label in labels:
        # Get the class, x_center, y_center, width, height
        class_, x_center, y_center, width, height = label.split(" ")
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)
        # Get the coordinates of the bounding box
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        # Increase the width of the bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        # Add text above the bounding box and increase the text size
        draw.text((x1, y1-20), "Potential industrial processing facility" , fill="black", font=ImageFont.truetype("arial.ttf", 20))
    return image

def apply_bounding_boxes_inference(image, bounding_boxes):
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    for bbox in bounding_boxes:
        x, y, width, height = bbox
        x1 = int((x - width/2))
        y1 = int((y - height/2))
        x2 = int((x + width/2))
        y2 = int((y + height/2))
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1-20), "Potential industrial processing facility" , fill="black", font=ImageFont.truetype("arial.ttf", 20))

    return image
tab1, tab2 = st.tabs(["False color composite", "SWIR"]) 

with tab1:
    st.title("False Color Compositing Dataset Viewer")
    st.write("This app displays images from the False color composite training dataset.")

    image_names = [name for name in imagesFC if name.endswith(".jpg")]
    image_name = st.selectbox("Select an image", image_names)
    image_path = os.path.join(path2imagesFC, image_name)
    image = Image.open(image_path)
    st.write("Selected image")
    st.image(image, caption="Selected image")

    # Get the labels for the image
    label_id = image_name.replace(".jpg", ".txt")
    label_path = os.path.join(path2labelsFC, label_id)
    with open(label_path, "r") as f:
        labels = f.readlines()

    # Draw the bounding boxes on the image
    image = draw_bounding_boxes(image, labels)
    st.write("Image with annotation")
    st.image(image, caption="Image with annotation")

with tab2:
    image_names = [name for name in imagesSWIR if name.endswith(".jpg")]
    image_name = st.selectbox("Select an image", image_names)
    image_path = os.path.join(path2imagesSWIR, image_name)
    image = Image.open(image_path)
    st.write("Selected image")
    st.image(image, caption="Selected image")

    # Get the labels for the image
    label_id = image_name.replace(".jpg", ".txt")
    label_path = os.path.join(path2labelsSWIR, label_id)
    with open(label_path, "r") as f:
        labels = f.readlines()

    # Draw the bounding boxes on the image
    image = draw_bounding_boxes(image, labels)
    st.write("Image with annotation")
    st.image(image, caption="Image with annotation")

# Now get the trained model and ask the user to upload an image to predict the bounding boxes
# Get the trained model
modelFC = get_model(model_id="unmonitored-ghg-emission-sources/5", api_key=api_key)
modelSWIR = get_model(model_id="unmonitored-ghg-emission-sources/8", api_key=api_key)

# Ask the user to upload an image
st.write("Upload an image to predict the bounding boxes for the False color composite layer")
uploaded_file_FC = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file_FC is not None:
    # Display the uploaded image
    image_FC = Image.open(uploaded_file_FC)
    st.image(image_FC, caption="Uploaded image 1")
    
    # Predict the bounding box
    predictions_FC = modelFC.infer(image_FC, confidence = 0.1)[0]
    bounding_boxes_FC = [
        [prediction.x, prediction.y, prediction.width, prediction.height]
        for prediction in predictions_FC.predictions
    ]
    
    # Draw the bounding boxes on the image
    annotated_image = apply_bounding_boxes_inference(image_FC, bounding_boxes_FC)
    st.write("Predicted bounding boxes")
    st.image(annotated_image, caption="Predicted bounding boxes")

    response = st.radio("Have some Potential industrial processing facility been detected ?", ("Yes", "No"))

    if response == "Yes":
        st.write("Upload an image to predict the bounding boxes for the SWIR layer")
        uploaded_file_SWIR = st.file_uploader("Choose a SWIR image...", type=["jpg", "jpeg", "png"])
        if uploaded_file_SWIR is not None:
            image_SWIR = Image.open(uploaded_file_SWIR)
            st.image(image_SWIR, caption="Uploaded image 2")

            # Predict the bounding box
            predictions_SWIR = modelSWIR.infer(image_SWIR, confidence = 0.1)[0]
            bounding_boxes_SWIR = [
                [prediction.x, prediction.y, prediction.width, prediction.height]
                for prediction in predictions_SWIR.predictions
            ]

            # Draw the bounding boxes on the image
            annotated_image = apply_bounding_boxes_inference(image_SWIR, bounding_boxes_SWIR)
            st.write("Predicted bounding boxes")
            st.image(annotated_image, caption="Predicted bounding boxes")

            response = st.radio("Have some Potential industrial processing facility been detected on the SWIR layer ?", ("Yes", "No"))

            if response == "Yes":
                st.write("Potential industrial processing facility detected on both layers. We probably are in presence of an unmonitored GHG emitting facility.")


    
    
    









