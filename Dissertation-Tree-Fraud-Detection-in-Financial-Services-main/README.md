
# Fraud Detection in Financial Services

This repository contains a machine learning project that focuses on **fraud detection** in financial services using the **PaySim1 dataset**. The goal is to develop a model that accurately predicts fraudulent transactions, helping financial institutions minimize fraud-related losses.

## Dataset

The dataset used for this project is the **PaySim1 Financial Services Simulation Dataset**, which is publicly available on Kaggle:

- [PaySim1 Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset simulates mobile money transactions, providing features such as transaction type, amount, origin, and destination, as well as a label indicating whether a transaction is fraudulent or not.

## Project Overview

This project demonstrates how to build and evaluate a machine learning model for detecting fraud in financial transactions. The focus is on using various classifiers, including **Support Vector Machines (SVM)**, **Random Forest**, and other algorithms, to accurately predict fraudulent activities.

### Key Features:

- **Data Preprocessing**: Handling missing values, feature scaling, and dealing with imbalanced data.
- **Model Implementation**: Training and testing machine learning models like **SVM**, **Random Forest**, and others for fraud detection.
- **Model Evaluation**: Using performance metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC** to assess model performance.
- **Hyperparameter Tuning**: Optimizing model performance using techniques like grid search and cross-validation.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Fraud-Detection-Financial-Services.git
   cd Fraud-Detection-Financial-Services
   ```

2. **Install the required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**

   Download the PaySim1 dataset from Kaggle and place it in your project folder, or specify the path to the dataset:

   ```python
   data = '/path-to-your-dataset/paysim1.csv'
   ```

## How to Run

Once you have the dataset and dependencies installed, run the script to train the model and make predictions:

```bash
python fraud_detection.py
```

The script will preprocess the data, train multiple machine learning models, and output the performance metrics for each model, allowing you to compare their accuracy in detecting fraudulent transactions.

### Example Python Code Snippet:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('/path-to-your-dataset/paysim1.csv')

# Preprocess the data (feature selection, scaling, etc.)
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Keywords

- Fraud detection
- Financial services
- Machine learning
- Predictive modeling
- Support Vector Machines (SVM)
- Random Forest
- Imbalanced data
- PaySim1 dataset
- Mobile transactions
- Classification

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Kaggle for providing the [PaySim1 Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).
- Scikit-learn for machine learning tools and libraries.
