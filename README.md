# AgriAI: AI-Powered Solution for Detecting Plants and Their Health Condition

## Overview
This project is an AI-powered system designed to detect plants and classify their health status. The system leverages deep learning to provide accurate and efficient plant monitoring. The detection model, built using YOLOv8, identifies sesame plants in an image, while the classification model, based on a pre-trained MobileNet, determines whether a plant is healthy or diseased. The ultimate goal is to integrate this system into a drone for automated farm monitoring and crop health assessment.

## Features
- **Sesame Plant Detection:** Uses YOLOv8 to detect sesame plants in an image and draw bounding boxes.
- **Health Classification:** Employs a pre-trained MobileNet model to classify plants as either healthy or diseased.
- **Scalability:** The system is designed to be integrated into drones for automated surveillance and response.

## Dataset Link
   - **Detection Dataset:** I collected and annotated sesame plant images, available on Kaggle. 📂 **[Detection (Sesame) Dataset](https://www.kaggle.com/datasets/ismailismailtijjani/keke-napep-tricycle-dataset)**  
- **Classification Dataset:** I used open sourced dataset. 📂 **[Classification (Bean) Dataset](https://www.kaggle.com/datasets/therealoise/bean-disease-dataset)**
## Hugging Face Demo Link
📂 **[AGRIAI Huggung Face Demo](https://huggingface.co/spaces/esssyjr/AGRIAI)**  

## Model Details
### **Sesame Plant Detection**
- **Model:** YOLOv8
- **Training Data:** Custom dataset of sesame plants
- **Performance:** Optimized for high accuracy and real-time processing

### **Plant Health Classification**
- **Model:** MobileNet (Pre-trained and fine-tuned)
- **Categories:** Healthy, Unhealthy
- **Optimization:** Transfer learning applied for efficient classification

## Future Integrations
- **Drone Deployment:** Automate plant monitoring and health assessment
- **Actionable Insights:** Enable real-time intervention by integrating with spraying mechanisms
- **Expanded Dataset:** Improve accuracy by collecting more diverse images from different farm conditions

  #### **Video Demo**  
Watch the detection test video here:  
📹 [KEKE_NAPEP_VIDEO_TRACKING_DEMO_1](https://youtu.be/sZ4QVAU8XIg?si=ywSxweO6F7owK_5B)  


## Installation & Usage
## 1. **Clone the Repository:**
   ```sh
   git clone https://github.com/esssyjr/AgriAI..AI-Powered_Solution_for_Detecting_Plants_and_Their_Health_Condition.git
   cd AgriAI..AI-Powered_Solution_for_Detecting_Plants_and_Their_Health_Condition
   ```
## 2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
## 3. **Download the Dataset:**
   - Get the dataset from Kaggle and place it in the project directory.

## 4. **Run the Application:**
   ```sh
   python app.py
   ```

## 5. **Test with the Dataset:**
To test the model:

- Upload an image of a sesame plant or bean dataset.
- Choose detection for sesame plant and bean dataset forclassification.
- The system will analyze and return results.
- There is voice interface for the classification model



## Collaboration
This project welcomes contributions and collaborations from:
- **AI/Computer Vision Developers:** To improve detection and classification accuracy
- **Agricultural Experts:** To enhance dataset diversity and model validation
- **Drone Engineers:** To assist with seamless drone integration

🚀 If you are interested in contributing, feel free to reach out!

## Citation
The dataset used in this project was provided by **EJAZTECH.AI**, a company dedicated to **gathering local datasets for AI applications**.  

📧 **Contact:** [ejaztech.ai@gmail.com](mailto:ejaztech.ai@gmail.com)  

We acknowledge **EJAZTECH.AI** for their invaluable contributions in providing localized data that played a crucial role in training and validating the **AgriAI: AI-Powered Solution for Detecting Plants and Their Health Condition**. 
