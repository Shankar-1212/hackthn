import gradio as gr
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO(r'D:\Demo-1\yolomammo.pt')

# Create output directory if it doesn't exist
output_dir = 'output_dir/'
os.makedirs(output_dir, exist_ok=True)

# Function to perform inference and draw bounding boxes, and save the result
def display_and_save_results(image):
    # Run YOLOv8 inference
    results = model(image)
    results = results[0]
    
    # Plot results and convert to PIL
    img_with_boxes = results.plot()  # This adds bounding boxes
    img_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    
    # Resize the image to fit display size (optional)
    resized_img = img_pil.resize((800, 600))
    
    # Save result image
    output_path = os.path.join(output_dir, 'result_image.jpg')
    img_pil.save(output_path)
    print(f"Results saved to {output_path}")
    
    return resized_img, output_path

# Function to print bounding box details and confidence
def print_detection_results(results):
    for result in results.boxes.data:
        print(f"Bounding Box: {result[:4]}, Confidence: {result[4]}, Class: {result[5]}")

# Gradio interface function
def gradio_inference(image):
    results = model(image)  # Run inference
    results = results[0]  # Extract results
    
    # Display bounding box details
    print_detection_results(results)
    
    # Save and display result image
    result_image, result_path = display_and_save_results(image)
    
    return result_image, result_path

# Create Gradio interface
interface = gr.Interface(
    fn=gradio_inference, 
    inputs=gr.Image(type="pil"),  # Image input
    outputs=[gr.Image(type="pil"), gr.File(label="Download Processed Image")],  # Display image and download link
    live=True  # Option to perform live inference
)

# Launch the interface
interface.launch()
