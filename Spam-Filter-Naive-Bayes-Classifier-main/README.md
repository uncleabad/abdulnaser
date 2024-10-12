# Spam/Non-Spam Detection Using Naive Bayes

This repository contains a machine learning project that implements a **Naive Bayes classifier** to detect spam and non-spam SMS messages. By leveraging the simplicity and effectiveness of Naive Bayes, this project demonstrates a powerful approach to solving the classic text classification problem of distinguishing between unsolicited (spam) and legitimate (non-spam) messages.

## Dataset

The dataset used in this project is the **SMS Spam Collection Dataset**, available on Kaggle:

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset contains 5,574 SMS messages labeled as **spam** or **ham** (non-spam). Each entry consists of a message and its corresponding label, which makes it ideal for binary classification tasks like spam detection.

## Project Overview

The goal of this project is to accurately classify SMS messages into spam or non-spam categories using the **Naive Bayes algorithm**. This is achieved through text preprocessing and training the Naive Bayes model on labeled data.

### Key Features:

- **Data Preprocessing**:
  - Tokenization: Splitting the message into individual tokens (words).
  - Stopword Removal: Removing common words (e.g., "the", "and") that do not contribute to classification.
  - TF-IDF (Term Frequency-Inverse Document Frequency): Transforming the raw text into numerical features based on the importance of words.
  
- **Model Training**: Implementing the **Naive Bayes classifier** to predict whether an SMS message is spam or non-spam. The classifier is well-suited for this task because of its efficiency in handling text data and conditional independence assumption.

- **Performance Metrics**: The performance of the classifier is evaluated using accuracy, precision, recall, F1-score, and confusion matrix to ensure robust spam detection.

## Installation and Requirements

To run this project, ensure that the following libraries are installed:

```bash
pip install pandas scikit-learn numpy nltk
```

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/Spam-Detection-Naive-Bayes.git
   cd Spam-Detection-Naive-Bayes
   ```

2. **Download the dataset**:

   Download the SMS Spam Collection dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place the file (`SMSSpamCollection`) in your project folder.

3. **Run the spam detection script**:

   ```bash
   python spam_classifier.py
   ```

The script will preprocess the data, train the Naive Bayes model, and output classification metrics on the test set.

### Example Code Snippet:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Data preprocessing
X = data['message']
y = data['label'].map({'ham': 0, 'spam': 1})
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Keywords

- Spam Detection
- Naive Bayes
- Text Classification
- SMS Spam Classification
- Natural Language Processing (NLP)
- Binary Classification
- SMS Spam Collection Dataset
- Machine Learning

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Kaggle for providing the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- The Scikit-learn and NLTK libraries for powerful machine learning and text processing tools.
