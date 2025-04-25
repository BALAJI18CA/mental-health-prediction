import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import torch
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set dynamic paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model paths
MODEL_PATHS = {
    'full_fusion': os.path.join(MODEL_DIR, 'full_fusion_model.keras'),
    'bert_only': os.path.join(MODEL_DIR, 'bert_only_model.keras'),
    'autoencoder': os.path.join(MODEL_DIR, 'autoencoder.keras')
}

# Initialize BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.eval().to(device)

# Batch BERT Embedding Function
def get_batch_embeddings(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = bert_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            batch_embeds = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeds)
    return np.vstack(embeddings)

# Preprocessing Function
def preprocess_text(text):
    if pd.isna(text) or not text.strip():
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# Load models
try:
    full_fusion_model = load_model(MODEL_PATHS['full_fusion'])
    bert_only_model = load_model(MODEL_PATHS['bert_only'])
    autoencoder = load_model(MODEL_PATHS['autoencoder'])
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model file not found: {str(e)}")
def compute_metrics(y_true, y_pred):
    try:
        accuracy = 1.0 if y_true == y_pred else 0.0
        f1 = f1_score([y_true], [y_pred], average='weighted', zero_division=0)
        recall = recall_score([y_true], [y_pred], average='weighted', zero_division=0)
    except Exception as e:
        print(f"Metric computation error: {str(e)}")
        accuracy, f1, recall = 0.0, 0.0, 0.0
    return accuracy, f1, recall
# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        clinical_data = np.array(data.get('clinical', [0, 0, 0, 0]), dtype=np.float32).reshape(1, -1)
        phys_data = np.array(data.get('physiological', [0, 0, 0, 0, 0]), dtype=np.float32).reshape(1, -1)
        social_text = data.get('social_media', "")
        chatbot_text = data.get('chatbot', "")

        clinical_scaler = StandardScaler()
        clinical_data[:, 0:1] = clinical_scaler.fit_transform(clinical_data[:, 0:1])
        phys_scaler = StandardScaler()
        phys_data = phys_scaler.fit_transform(phys_data)
        phys_cnn = phys_data.reshape(1, phys_data.shape[1], 1)
        phys_lstm = phys_data.reshape(1, phys_data.shape[1], 1)

        social_clean = preprocess_text(social_text)
        social_emb = get_batch_embeddings([social_clean])[0] if social_clean else np.zeros(768)
        chatbot_clean = preprocess_text(chatbot_text)
        chatbot_emb = get_batch_embeddings([chatbot_clean])[0] if chatbot_clean else np.zeros(768)

        if social_emb.shape != (768,) or chatbot_emb.shape != (768,):
            raise ValueError(f"Invalid embedding shapes: social_emb {social_emb.shape}, chatbot_emb {chatbot_emb.shape}")

        full_pred = full_fusion_model.predict([clinical_data, phys_cnn, phys_lstm, social_emb.reshape(1, -1), chatbot_emb.reshape(1, -1)]).argmax(axis=1)[0]
        bert_pred = bert_only_model.predict([social_emb.reshape(1, -1), chatbot_emb.reshape(1, -1)]).argmax(axis=1)[0]

        phys_recon = autoencoder.predict(phys_data)
        mse = np.mean(np.square(phys_recon - phys_data))
        anomaly_threshold = 0.05  # Replace with actual threshold from training
        is_anomaly = mse > anomaly_threshold

        with open(os.path.join(OUTPUT_DIR, 'real_time_data.csv'), 'a') as f:
            f.write(f"{clinical_data.tolist()},{phys_data.tolist()},{social_text},{chatbot_text},{full_pred},{bert_pred},{is_anomaly}\n")

        ground_truth = data.get('depression_level', None)
        if ground_truth is not None:
            ground_truth = int(ground_truth)
            acc_full_client = accuracy_score([ground_truth], [full_pred])
            f1_full_client = f1_score([ground_truth], [full_pred], average='weighted')
            recall_full_client = recall_score([ground_truth], [full_pred], average='weighted')
            acc_bert_client = accuracy_score([ground_truth], [bert_pred])
            f1_bert_client = f1_score([ground_truth], [bert_pred], average='weighted')
            recall_bert_client = recall_score([ground_truth], [bert_pred], average='weighted')
            print(f"Client Full Fusion - Accuracy: {acc_full_client:.4f}, F1-Score: {f1_full_client:.4f}, Recall: {recall_full_client:.4f}")
            print(f"Client BERT-Only Fusion - Accuracy: {acc_bert_client:.4f}, F1-Score: {f1_bert_client:.4f}, Recall: {recall_bert_client:.4f}")

        return jsonify({
            'full_fusion_prediction': int(full_pred),
            'bert_only_prediction': int(bert_pred),
            'is_anomaly': bool(is_anomaly),
            'depression_levels': {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)