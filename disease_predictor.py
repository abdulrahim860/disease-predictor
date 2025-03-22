import re
import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Download NLTK data if not available
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

medical_terms = ["fever", "cough", "headache", "nausea", "fatigue", "sore throat", "shortness of breath"]

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def extract_symptoms(text):
    tokens = preprocess_text(text)
    single_words = [word for word in tokens if word in medical_terms]
    bigrams = [" ".join(bigram) for bigram in ngrams(tokens, 2)]
    bigram_symptoms = [term for term in bigrams if term in medical_terms]
    return list(set(single_words + bigram_symptoms))

# Training Data (Feature Vectors & Labels)
data = [
    ([1, 1, 1, 0, 1, 0, 0], "Flu"),
    ([1, 1, 0, 1, 1, 1, 1], "COVID-19"),
    ([0, 0, 1, 0, 0, 0, 0], "Migraine"),
    ([0, 0, 0, 1, 1, 0, 0], "Food Poisoning"),
    ([0, 0, 0, 1, 1, 0, 0], "Gastritis"),
    ([0, 1, 0, 0, 1, 1, 0], "Bronchitis"),
    ([0, 0, 0, 0, 1, 0, 1], "Asthma")
]

X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Model Definition
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),  # Match training data shape
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_onehot, epochs=50, batch_size=4, verbose=0)

def predict_disease(user_input):
    symptoms = extract_symptoms(user_input)
    input_vector = np.array([[1 if term in symptoms else 0 for term in medical_terms]])
    prediction = model.predict(input_vector)[0]
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label, symptoms
