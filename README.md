# Lung Cancer Probability Prediction

[![Render Deployment](https://img.shields.io/badge/Deployed%20on-Render-46a2f1.svg?logo=render)](https://lung-cancer-probability-prediction.onrender.com)
[![Flask](https://img.shields.io/badge/Built%20With-Flask-blue.svg?logo=flask)](https://flask.palletsprojects.com/)
[![GitHub](https://img.shields.io/badge/Open%20Source-GitHub-black.svg?logo=github)](https://github.com/YOUR_USERNAME/YOUR_REPO)

🔗 **Live Demo:**    [Lung Cancer Prediction](https://lung-cancer-probability-prediction.onrender.com)


# Lung Cancer Prediction

## 📌 Project Overview
This project focuses on predicting lung cancer using machine learning models. It utilizes multiple classifiers like **Logistic Regression, Random Forest, XGBoost, and Neural Networks** to analyze patient data and predict the likelihood of lung cancer based on various risk factors.

---

## 🛠️ Technologies Used
- **Programming Language**: Python 🐍
- **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, TensorFlow/Keras, XGBoost, imbalanced-learn
- **Machine Learning Models**: Logistic Regression, Random Forest, XGBoost, Neural Network
- **Data Preprocessing**: Label Encoding, Standardization, SMOTE (Synthetic Minority Over-sampling)

---

## 📂 Dataset
The dataset used for training and evaluation is **survey lung cancer.csv**.

- Ensure that the dataset is placed in the `Data/` directory before running the code.

---

## 🔧 Installation & Setup
Before running the project, install the required dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow keras joblib
```

---

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Lung_Cancer_Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Lung_Cancer_Prediction
   ```
3. Run the Python script:
   ```bash
   python lung_cancer_prediction.py
   ```

---

## 🏗️ Model Training & Evaluation
The following machine learning models are implemented:
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**
4. **Neural Network (Deep Learning)**

Each model's accuracy and performance are evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 📊 Model Performance Comparison
After training, the accuracy scores of different models are compared and displayed. The trained **Random Forest model** is saved as a `.pkl` file for future predictions.

```python
import joblib
joblib.dump(rf, 'lung_cancer_model.pkl')
```

---

## 📜 Results Visualization
Confusion matrices and performance metrics are plotted using **Seaborn and Matplotlib** to visually analyze the model's effectiveness.

```python
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

---

## 💾 Saving & Loading the Model
Once trained, the best model can be saved and used later for predictions:
```bash
# Save the model
joblib.dump(rf, 'lung_cancer_model.pkl')

# Load the model
model = joblib.load('lung_cancer_model.pkl')
```

---

## 🤝 Contributing
Feel free to contribute to this project by:
- Adding new ML models
- Improving data preprocessing
- Enhancing performance evaluation

Fork this repository, make your changes, and submit a pull request! 🎯

---

## 📩 Contact
For any queries or suggestions, feel free to reach out!

📧 **Email**: gaur.prateek.1609@gmail.com  

---

⭐ **If you find this project useful, don't forget to give it a star!** ⭐

