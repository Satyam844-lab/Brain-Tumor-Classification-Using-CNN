# 🧠 Brain Tumor Classification using CNN

A deep learning based web application that classifies Brain MRI images into **4 categories**:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

The model is built using a **Convolutional Neural Network (CNN)** and deployed as an interactive **Streamlit web app** that allows users to upload MRI images and receive instant predictions with confidence scores.

---

## 🚀 Live Features

✔ Upload MRI image
✔ Automatic image preprocessing
✔ Predict tumor type
✔ Displays confidence score
✔ Simple and user-friendly interface

---

## 🧠 Model Details

* Architecture: Convolutional Neural Network (CNN)
* Input size: 128 × 128 × 3
* Classes: 4 tumor categories
* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Achieved Accuracy: ~90% on test dataset

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```
brain-tumor-classification/
│
├── app.py
├── brain_tumor_4class_model.keras
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

### 1. Clone repository

```
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 📊 Example Workflow

1. Upload MRI scan image
2. Model processes image
3. Predicts tumor category
4. Displays confidence score

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used as a substitute for professional medical diagnosis.

---

## 📌 Future Improvements

* Transfer Learning (MobileNet / ResNet)
* Improved accuracy with advanced augmentation
* Model explainability (Grad-CAM visualization)
* Deployment with Docker

---

## 🤝 Contributions

Suggestions and improvements are welcome!
