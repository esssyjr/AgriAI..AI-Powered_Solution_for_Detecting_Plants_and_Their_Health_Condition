import gradio as gr
import pickle
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS  # Import Google Text-to-Speech for voice output
import tempfile  # To create temporary audio files

# Load the disease classification model
with open('Health_Condition.pkl', 'rb') as file:
    disease_model = pickle.load(file)

# Load the YOLO plant detection model
plant_model = YOLO('Plant_Detection.pt') 

# Define the binary class labels for the disease classification
class_labels = ["healthy", "unhealthy"]

# Translations for each language
translations = {
    "English": {"healthy": "The plant is healthy", "unhealthy": "The plant is unhealthy"},
    "Hausa": {"healthy": "Shuka tana da lafiya", "unhealthy": "Shuka bata da lafiya"},
    "Igbo": {"healthy": "Ahụ jịrị shụ̀kwa di mma", "unhealthy": "Ahụ jịrị shụ̀kwa adịghị mma"},
    "Yoruba": {"healthy": "Eweko wa ni ilera", "unhealthy": "Eweko ko ni ilera"}
}

# Function to preprocess image for disease classification
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0  
    return image

# Function to classify plant health (disease classification)
def predict_health(image, language):
    preprocessed_image = preprocess_image(image)
    prediction = disease_model.predict(preprocessed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_index]
    
    # Get translation for the selected language
    text = translations[language][predicted_class]
    tts = gTTS(text=text, lang='en' if language == "English" else 'ha' if language == "Hausa" else 'ig' if language == "Igbo" else 'yo')
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_file.name)
    
    return f"Predicted class: {predicted_class}", audio_file.name  # Return prediction text and audio file

# Function to detect plants and draw red bounding boxes (plant detection)
def detect_plant(image):
    results = plant_model(image)  # YOLO processes the image directly
    image_np = np.array(image)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
        labels = result.boxes.cls.cpu().numpy()  # Get the predicted class indices
        confs = result.boxes.conf.cpu().numpy()  # Get confidence scores
        
        for box, label, conf in zip(boxes, labels, confs):
            xmin, ymin, xmax, ymax = map(int, box)
            # Draw a red bounding box with thicker lines
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)  # Red color, thickness 4
            class_name = plant_model.names[int(label)]  # Get the class name
            text = f"{class_name} {conf:.2f}"
            # Put class label and confidence score above the bounding box
            cv2.putText(image_np, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Convert the result back to a PIL image for display in Gradio
    output_image = Image.fromarray(image_np)
    return output_image

# Main function to switch between models based on user input
def model_selector(image, choice, language):
    if choice == "Health Condition":
        prediction_text, audio_file = predict_health(image, language)
        return prediction_text, None, audio_file  # Return prediction text, no image, and audio file
    elif choice == "Plant Detection":
        detection_image = detect_plant(image)
        return None, detection_image, None  # Return no text, output image, and no audio

# Create the Gradio interface with a dropdown to select between the two models
interface = gr.Interface(
    fn=model_selector,
    inputs=[
        gr.Image(type="pil"), 
        gr.Radio(["Health Condition", "Plant Detection"], label="Select Model"),
        gr.Radio(["English", "Hausa", "Igbo", "Yoruba"], label="Select Language for Health Condition")
    ],
    outputs=[gr.Textbox(label="Health Prediction"), gr.Image(label="Plant Detection"), gr.Audio(label="Voice Feedback")],
    title="AgriAI: AI-Powered Solution for Detecting Plants and Their Health Condition"
)

interface.launch(share=True)