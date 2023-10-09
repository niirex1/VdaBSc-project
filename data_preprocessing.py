# data_preprocessing.py
# Implementation of a Data Preprocessing Algorithm for Smart Contract Vulnerability Detection
# VdaBSc-Project

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def runtime_batch_normalization(data):
    """
    Perform real-time runtime batch normalization.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

def data_augmentation(data):
    """
    Perform data augmentation techniques.
    """
    # Your code for data augmentation
  from sklearn.utils import shuffle

def data_augmentation(data):
    """
    Perform data augmentation techniques.
    """
    # Adding Gaussian noise
    noise_factor = 0.05
    data_with_noise = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    
    # Scaling
    scaling_factor = 1.2
    data_scaled = data * scaling_factor
    
    # Concatenating the original, noisy, and scaled data
    augmented_data = np.concatenate((data, data_with_noise, data_scaled))
    
    # Shuffling the data
    augmented_data = shuffle(augmented_data)
    
    return augmented_data

# Example usage in the main function or any other part of your script
if __name__ == "__main__":
    # ... (previous code)
    
    # Data Augmentation
    augmented_features = data_augmentation(normalized_features)
    
    # ... (subsequent code)

    return data

def feature_representation(data):
    """
    Apply N-grams and one-hot encoding for feature representation.
    """
    # Your code for N-grams and one-hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def generate_ngrams(opcodes, n=2):
    """
    Generate N-grams from the list of opcodes.
    """
    ngrams = []
    for i in range(len(opcodes) - n + 1):
        ngram = ' '.join(opcodes[i:i+n])
        ngrams.append(ngram)
    return ngrams

def one_hot_encoding(ngrams):
    """
    Perform one-hot encoding on the N-grams.
    """
    onehot_encoder = OneHotEncoder(sparse=False)
    ngrams_array = np.array(ngrams).reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(ngrams_array)
    return onehot_encoded

# Example usage in the main function or any other part of your script
if __name__ == "__main__":
    # Sample list of opcodes
    sample_opcodes = ['PUSH1', 'ADD', 'PUSH1', 'MSTORE', 'PUSH1', 'SHA3']
    
    # Generate 2-grams from the sample opcodes
    ngrams = generate_ngrams(sample_opcodes, n=2)
    print("Generated N-grams:", ngrams)
    
    # Perform one-hot encoding on the generated N-grams
    onehot_encoded_ngrams = one_hot_encoding(ngrams)
    print("One-hot Encoded N-grams:", onehot_encoded_ngrams)
  
    return data

def split_dataset(features, labels):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(features, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Load dataset
    data = load_dataset('smart_contract_dataset.csv')
    
    # Separate features and labels
    features = data.drop('label', axis=1)
    labels = data['label']
    
    # Real-time Runtime Batch Normalization
    normalized_features = runtime_batch_normalization(features)
    
    # Data Augmentation
    augmented_features = data_augmentation(normalized_features)
    
    # Feature Representation using N-grams and One-hot Encoding
    final_features = feature_representation(augmented_features)
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = split_dataset(final_features, labels)
    
    # Save or return the preprocessed data
    # Your code here
