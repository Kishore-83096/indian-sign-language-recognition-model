# Indian Sign Language Recognition 👋

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[GitHub Repository](https://github.com/Kishore-83096/indian-sign-language-recognition-model)

---

## 🚀 Project Overview
This **Machine Learning project** aims to bridge the communication gap for hearing-impaired individuals by **translating Indian Sign Language (ISL) gestures into text or speech**.  

The system leverages Python libraries and ML algorithms to accurately detect and recognize hand gestures in real-time.

---

## 🛠 Technologies Used
- **Programming Language:** Python  
- **Libraries:** TensorFlow, OpenCV, Pandas, NumPy  

---

## ✨ Features
- Capture gesture data through a camera.  
- Detect contours to establish hand dimensions for accurate recognition.  
- Classify gestures into meaningful text or speech using a machine learning model.  
- Utilize a comprehensive dataset of gestures for effective model training.  

---

## 📦 Model & Data
The pre-trained model file `isl_model.h5` is **too large for GitHub** (>100 MB).  

- **Download link:** [Google Drive](https://drive.google.com/drive/folders/1Og4j6wpjkzyL4zz6HyeTjccAbeU7BgBw?usp=drive_link)  
- **Placement:** Place the downloaded model in the project root directory (`islr_project/isl_model.h5`).

> ⚠️ Tip: Accuracy can be improved by training with a diverse set of gestures to enhance recognition performance.

---

## 🏁 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Kishore-83096/indian-sign-language-recognition-model.git
cd indian-sign-language-recognition-model


### install dependencies
pip install -r requirements.txt

###3. Place the pre-trained model

Download isl_model.h5 and place it in the project root directory.

4. Run the project
# Example (modify according to your implementation)
python predict.py

