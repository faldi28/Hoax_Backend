import flask
from flask import request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
import re
import os
import sys
from langdetect import detect, LangDetectException

# --- Inisialisasi Aplikasi Flask dan CORS ---
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


# --- Konfigurasi dan Pemuatan Model ---

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model') 
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    print(f"Backend: Mencoba memuat model dari: {MODEL_PATH}", file=sys.stdout)
    model_file_safetensors = os.path.join(MODEL_PATH, 'model.safetensors')
    model_file_bin = os.path.join(MODEL_PATH, 'pytorch_model.bin')
    config_file = os.path.join(MODEL_PATH, 'config.json')

    if not os.path.exists(config_file) or not (os.path.exists(model_file_safetensors) or os.path.exists(model_file_bin)):
        print(">>> PERINGATAN: File model ('model.safetensors' atau 'pytorch_model.bin') tidak ditemukan.", file=sys.stdout)
        print(f">>> Pastikan Anda telah menyalin semua file model ke folder '{MODEL_PATH}'.", file=sys.stdout)
        print(">>> Server akan berjalan dengan fungsionalitas terbatas.", file=sys.stdout)
        return False
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        print(">>> Backend: Model dan tokenizer berhasil dimuat.", file=sys.stdout)
        return True
    except Exception as e:
        print(f"Backend: Gagal memuat model. Error: {e}", file=sys.stderr)
        return False

is_model_loaded = load_model_and_tokenizer()


# --- Fungsi Pra-pemrosesan (DISESUAIKAN DENGAN COLAB) ---
def preprocess_for_bert(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'(.)\1{2,}', r'\1', text) 
    text = re.sub(r'([.?,!])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def truncate_head_tail(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        head_len = 128
        tail_len = max_length - head_len - 2
        head_tokens = tokens[:head_len]
        tail_tokens = tokens[-tail_len:]
        return tokenizer.convert_tokens_to_string(head_tokens + tail_tokens)
    return text


# --- Fungsi Prediksi ---
def predict(text):
    if not is_model_loaded:
        return "Model Error"

    processed_text = preprocess_for_bert(text)
    
    final_text = truncate_head_tail(processed_text, tokenizer)

    inputs = tokenizer(
        final_text, 
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    )
    
    device = model.device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).squeeze()

    probabilities = probabilities.cpu().tolist()
    
    predicted_class_id = probabilities.index(max(probabilities))
    label = "Valid" if predicted_class_id == 1 else "Hoax"
    
    return label


# --- Endpoint API ---
@app.route('/', methods=['GET'])
def home():
    return "<h1>API Deteksi Hoax</h1><p>Server berjalan. Gunakan endpoint /predict.</p>"

@app.route('/predict', methods=['POST'])
def predict_api():
    if not is_model_loaded:
        return jsonify({"error": "Model tidak tersedia di server."}), 503

    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Permintaan harus dalam format JSON dan berisi key 'text'"}), 400

    content = request.json
    text_to_predict = content['text']

    # --- TAMBAHAN: VALIDASI BAHASA ---
    try:
        language = detect(text_to_predict)
        
        if language != 'id':
            return jsonify({"error": "NOT_INDONESIAN", "message": "Teks yang dianalisis harus dalam Bahasa Indonesia."}), 400
            
    except LangDetectException:
        return jsonify({"error": "LANGUAGE_UNKNOWN", "message": "Bahasa dari teks tidak dapat dideteksi."}), 400
    # --- AKHIR DARI TAMBAHAN ---

    try:
        prediction_result = predict(text_to_predict)
        
        if prediction_result == "Model Error":
            return jsonify({"error": "Prediksi gagal karena model tidak dimuat."}), 500
        
        return jsonify({
            "prediksi": prediction_result
        })
    except Exception as e:
        print(f"Backend: Error saat prediksi: {e}", file=sys.stderr)
        return jsonify({"error": "Terjadi kesalahan internal saat melakukan prediksi."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)