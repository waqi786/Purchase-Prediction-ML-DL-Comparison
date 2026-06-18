# 🛒 Consumer Purchase Prediction: ML vs. DL Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Project Overview
This project explores the predictability of consumer purchase behavior using a dataset of 1,500 samples. The objective was to compare traditional **Machine Learning (ML)** algorithms with **Deep Learning (DL)** architectures to determine the most effective approach for behavioral prediction.

In modern e-commerce, understanding the "intent to buy" is crucial for personalized marketing. This project serves as a benchmark study to see if behavioral metrics alone (like time on app or attention scores) are sufficient to predict a binary purchase outcome.

### 🧠 Why This Project is Important
In real-world data science, datasets are often "noisy" or have weak signals. This project demonstrates a **professional end-to-end pipeline** where the focus isn't just on high accuracy, but on:
- Comprehensive **Exploratory Data Analysis (EDA)**.
- Implementation of multiple classification algorithms.
- **Architecture Comparison**: Testing if a Deep Neural Network can outperform an SVM on tabular data.
- **Honest Model Assessment**: Diagnosing why models perform near random-chance and identifying the "Signal vs. Noise" problem.

---

## 🛠️ Tech Stack

| Category | Tools |
| :--- | :--- |
| **Languages** | Python (Pandas, NumPy) |
| **Visualization** | Seaborn, Matplotlib (High-Res 500 DPI) |
| **Machine Learning** | Scikit-Learn, XGBoost, SVM |
| **Deep Learning** | TensorFlow, Keras |

---

## 📊 Data Insights
- **Samples:** 1,500 | **Features:** 16
- **Target:** `purchase_likelihood` (Class Balance: 52.5% Purchase / 47.5% No Purchase)
- **Key Features:** Numerical (Interest level, Attention score, Time in zone) & Categorical (Loyalty status, Age group, Device type).

### 🔍 EDA Highlights
I generated professional-grade visualizations using a custom **Dark Orange & Blue** theme:
- **Correlation Heatmap:** Sabse ahem observation yeh thi ke features ke darmiyan correlation bht kam tha, jo ke prediction difficulty ko justify karta hai.
- **Distribution Plots:** Visualized high overlap in "Interest Level" and "Attention Score" between buyers and non-buyers.
- **Outlier Detection:** Used box plots to identify data variance across different age groups and loyalty statuses.

---

## ⚙️ Workflow & Preprocessing

A rigorous preprocessing pipeline was established to ensure model stability:

### 1. Feature Engineering & Cleaning
- **Identifier Removal**: Dropped `consumer_id` to prevent the model from memorizing specific users.
- **Categorical Handling**: 7 variables (Age Group, Gender, Loyalty Status, etc.) were transformed using **One-Hot Encoding**, expanding the feature space to 26 dimensions.

### 2. Data Preparation
- **Stratified Splitting**: 80% Training / 20% Testing split, ensuring the purchase-to-non-purchase ratio remains consistent in both sets.
- **Feature Scaling**: Applied `StandardScaler` ($\mu=0, \sigma=1$). This is critical for **SVM (distance-based)** and **Neural Networks (gradient descent stability)**.
- **Cross-Validation**: Used to ensure that our ML model metrics aren't just a result of a "lucky" split.

### 3. Deep Learning Architecture
We implemented two architectures using Keras:
- **Simple FNN**: 3 Layers (16 -> 8 -> 1) with ReLU activation.
- **Deep MLP**: 
  - **Input Layer**: 26 neurons (fully connected).
  - **Hidden Layers**: Dense 64 and Dense 32 neurons with **Dropout (0.2)** and L2 Regularization.
  - **Output Layer**: Sigmoid activation function for binary classification.
  - Optimizer: Adam | Loss: Binary Cross-Entropy.

---

## 🚀 Model Performance Comparison

### Traditional Machine Learning
| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **SVM (Best)** | **0.5300** | **0.5913** | **0.5147** |
| Random Forest | 0.5000 | 0.5455 | 0.4920 |
| KNN | 0.5100 | 0.5304 | 0.5131 |
| XGBoost | 0.4933 | 0.5250 | 0.5020 |
| Logistic Regression | 0.4933 | 0.5581 | 0.4547 |

### Deep Learning
| Model | Accuracy | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Simple FNN** | 0.5067 | **0.7325** | 0.6085 |
| Deep MLP (Dropout) | 0.4800 | 0.7006 | 0.5851 |

> **Key Finding:** The Neural Network was "over-optimistic," predicting purchases frequently (high recall), whereas the SVM maintained a more conservative and balanced decision boundary.

---

## 💡 Critical Analysis & Lessons Learned

### Why is accuracy low?
1. **Weak Signal:** EDA showed that buyers and non-buyers behave almost identically in this specific dataset.
2. **Data Volume:** 1,500 rows are insufficient for Deep Learning to extract complex patterns effectively.
3. **Feature Gap:** Missing critical features like "Historical Purchase Data", "Price Sensitivity", or "Session Frequency".

### Portfolio Value
In a professional environment, reporting a "failed" or "low signal" model is as important as reporting a successful one. It prevents the business from wasting resources on non-predictive features. This project showcases my ability to **think like a scientist**.

---

## 🚀 How to Run

### 1. Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/waqi786/Purchase-Prediction-ML-DL-Comparison.git
cd Purchase-Prediction-ML-DL-Comparison
pip install -r requirements.txt
```

### 2. Running the Analysis
Navigate to the `notebooks/` directory and open the Jupyter notebook:
```bash
jupyter notebook notebooks/consumer_behavior_analysis.ipynb
```

---

## 📂 Repository Structure
```
consumer-behavior-prediction/
│
├── notebooks/
│   └── consumer_behavior_analysis.ipynb   # Full Jupyter notebook
│
├── outputs/
│   ├── *.jpg                               # All high-resolution plots
│   └── *.csv                               # Model results
│
├── README.md
└── requirements.txt
```

## Acknowledgments
- Dataset provided by Kaggle user `freshersstaff`
- Built as part of my data science portfolio to demonstrate practical skills

## License
This project is for educational and portfolio purposes only.
