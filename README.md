# 🌐 English → Hindi Neural Machine Translator

Excited to share my latest deep learning project — an English to Hindi Neural Machine Translator!

## 🔧 What I built
- **Seq2Seq** model with a **Bidirectional LSTM** encoder  
- **Bahdanau Attention** mechanism  
- Deployed on **Streamlit Cloud**  
- Models hosted on **Hugging Face** 🤗  

## 🛠️ Tech Stack
Python | TensorFlow | Keras | Streamlit | Hugging Face | GitHub

## 🚀 How it works
- The **Streamlit app** (`app.py`) provides an input box for English text and shows the predicted Hindi translation.
- The **inference pipeline** (`translation.py`) loads:
  - `models/encoder_model.keras`
  - `models/decoder_model.keras`
- If the model file is missing locally, it **auto-downloads** the models from Hugging Face.
- The decoder uses **greedy decoding** (selects the most likely token each step via `argmax`) to generate the translation.

## ✅ Try it online
https://neural-machine-translation-english-to-hindi-djktxtlpqrkxpgwmhk.streamlit.app/

## 🤗 Model on Hugging Face
huggingface.co/CODE-WITH-MANISH337/English_to_hindi_transaltor

## 💻 GitHub Repository
https://github.com/CODE-WITH-MANISH337/Neural-Machine-Translation-English-to-Hindi

## 📌 Honest note
The model is not perfect yet — translation quality still has room to grow. But that’s the beauty of machine learning: there is always a next step.

## 🔮 What I plan to improve
- **Transformer** architecture (like mBART or IndicTrans)
- Larger and cleaner dataset
- **Beam search decoding** instead of greedy decoding
- Pretrained Hindi word embeddings

This project taught me a lot about seq2seq architecture, attention mechanisms, inference pipelines, and deploying ML models end to end.

Every wrong prediction is a lesson 💡

#MachineLearning #DeepLearning #NLP #Python #TensorFlow #HuggingFace #Streamlit #NeuralMachineTranslation #Hindi #OpenSource #StudentDeveloper #AIProjects

