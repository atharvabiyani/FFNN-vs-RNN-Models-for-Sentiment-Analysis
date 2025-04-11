# FFNN vs RNN Models for Sentiment Analysis

This project explores and compares two neural network architectures—Feedforward Neural Networks (FFNN) and Recurrent Neural Networks (RNN)—for performing 5-class sentiment analysis on Yelp review data. The goal is to predict a review's star rating (1–5) based on the textual content.

---

## 📁 Dataset & Embeddings

You are provided with preprocessed Yelp review datasets and word embeddings:

- `train.pkl` – Training set  
- `val.pkl` – Validation set  
- `test.pkl` – Test set  
- `word_embedding.pkl` – Pre-trained word embeddings for RNN input

The dataset and embeddings are available via eLearning.

---

## 🧠 Models Implemented

### 🔹 1. Feedforward Neural Network (FFNN)

- Uses Bag-of-Words vectorization for review input
- One hidden layer with ReLU activation
- Output layer with Softmax for class probabilities
- Inputs are fixed-length vectors

### 🔹 2. Recurrent Neural Network (RNN)

- Utilizes pre-trained word embeddings to represent each word in the input sequence  
- Processes the sequence using a basic RNN layer, capturing word order and context  
- Produces hidden states at each timestep, which are **summed** to form a fixed-size vector  
- This final vector is passed through a linear layer and a **Softmax** to output class probabilities
