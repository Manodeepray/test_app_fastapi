Sure! Here's a **README.md** file for your project. This will help you document the **FastAPI-based image classification service** along with the **Streamlit frontend** and **deployment details**.

---

# **Image Classification API & Streamlit Interface**

### **Project summary link ** : [Link](https://docs.google.com/document/d/13rOgY5U4i_d9K_Wt6KOXsMFOlEyFRx96KaJ8D5p7gtc/edit?usp=sharing)

A **deep learning-based image classification project** using **ResNet18**, implemented with **FastAPI** for backend inference and **Streamlit** for a user-friendly frontend interface.

## 🚀 **Project Overview**

This project classifies handwritten digits using a **ResNet18 model trained on the MNIST dataset**. The backend is built using **FastAPI**, and a **Streamlit-based web UI** allows users to upload images for real-time classification.

## 📌 **Features**

✅ **Image Preprocessing & Feature Engineering**  
✅ **ResNet18 Model Training & Optimization (PyTorch)**  
✅ **FastAPI-based RESTful API for Model Inference**  
✅ **Basic Authentication for API Security**  
✅ **Logging & Error Handling for Robustness**  
✅ **Grad-CAM / SHAP for Model Explainability**  
✅ **Streamlit Frontend for User Interaction**  
✅ **Docker Containerization & Cloud Deployment**

## 📂 **Project Structure**

```
📦 Image Classification Project
├── 📁 models             # Trained model weights
│   ├── resnet18.pth     # Saved model for inference
├── 📁 notebooks          # Jupyter notebooks for EDA & Training
|
|
├── app.py            # FastAPI main application
├── model.py          # ResNet18 model prediction and processing of input
├── model_arch.py     # ResNet18 model skeleton for loading
├── Dockerfile        # Containerization script
├── API.log           # Logging output
├── streamlit_app.py            # Streamlit UI for classification
├── README.md             # Project Documentation
├── requirements.txt
```

---

## 🔧 **Setup & Installation**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/Manodeepray/test_app_fastapi.git
cd test_app_fastapi
```

### **2️⃣ Set Up the Backend (FastAPI)**

#### **Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **Run the API Locally**

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

The API will be available at: **`http://localhost:8080/predict/`**

---

### **3️⃣ Set Up the Frontend (Streamlit)**

#### **Run the Streamlit App**

```bash
streamlit run app.py
```

🔗 **Live Deployed Version:** [Digit Recognizer Streamlit App](https://digitrecognizerv1.streamlit.app/)

---

## 🌐 **API Usage Guide**

### **Authentication**

- The API uses **Basic Authentication**.
- Valid credentials (username/password) are stored in the `VALID_USERS` dictionary.

### **Request Format**

**Endpoint:** `/predict/`  
**Method:** `POST`  
**Headers:**

```json
{
  "Authorization": "Basic <base64-encoded-credentials>"
}
```

**Body (multipart/form-data):**

- `file`: Image file (PNG, JPG, etc.)

### **Example API Call using cURL**

```bash
curl -X 'POST' \
  'http://localhost:8080/predict/' \
  -u "admin:password" \
  -F "file=@sample_image.png"
```

### **Example Response**

```json
{
  "prediction": "5"
}
```

---

## 🛠 **Deployment Details**

### **Docker Containerization**

To build and run the API in a container:

```bash
docker build -t image-classifier .
docker run -p 8080:8080 image-classifier
```

### **Cloud Deployment (EC2 - In Progress)**

- The backend will be deployed on **AWS EC2** for public access.
- Additional cloud hosting (Render, GCP, or Azure) may be considered.

---

## 📊 **Model Explainability (Grad-CAM & SHAP)**

- **Grad-CAM** is used to visualize which image regions influence model predictions.
- **SHAP** (SHapley Additive exPlanations) provides interpretability for individual classifications.

---

## 📎 **Links & Resources**

📂 **GitHub Repository:** [test_app_fastapi](https://github.com/Manodeepray/test_app_fastapi)  
🚀 **Streamlit App:** [Digit Recognizer](https://digitrecognizerv1.streamlit.app/)  
🔧 **Backend Deployment (EC2):** _Currently in progress_

---

## 🤝 **Contributing**

Want to contribute? Feel free to fork the repo and submit a PR!

---
