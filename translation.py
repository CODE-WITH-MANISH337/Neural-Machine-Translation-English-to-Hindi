import json
from tensorflow import keras
import contractions
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
import os

REPO_ID = "CODE-WITH-MANISH337/English_to_hindi_transaltor"


print("Downloading models from Hugging Face...")
snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir="./models"
    )
print("Models downloaded!")

with open(r'tokenizers\eng_vocab.json', 'r', encoding='utf-8') as f:
    eng_vocab= json.load(f)

with open(r'tokenizers\hindi_vocab.json', 'r', encoding='utf-8') as f:
    hindi_vocab= json.load(f)

encoder_model=keras.models.load_model('models/encoder_model.keras')
decoder_model=keras.models.load_model('models/decoder_model.keras')

# Rebuild TextVectorization from vocabulary
english_vector = tf.keras.layers.TextVectorization(
    max_tokens=len(eng_vocab),
    output_sequence_length=62   # your max_eng_len
)
english_vector.set_vocabulary(eng_vocab)

vocab=hindi_vocab
output_dict={idx:word for idx,word in enumerate(vocab)}

start_index=vocab.index('<start>')
end_index=vocab.index('<end>')

print(f'start token index {start_index}')
print(f'end token index {end_index}')
# print(output_dict)
def translation(text,max_decode_len=20):
    text=text.lower()
    text=contractions.fix(text)
    input_vector=english_vector([text])
    input_vector=tf.cast(input_vector,tf.int32)

    encoder_output,h,c=encoder_model.predict(input_vector)
    encoder_output = tf.convert_to_tensor(encoder_output, dtype=tf.float32)
    h = tf.convert_to_tensor(h, dtype=tf.float32)
    c = tf.convert_to_tensor(c, dtype=tf.float32)

    decoder_input=np.array([[start_index]])
    output_word=[]
    for  i in range(max_decode_len):
        decoder_input_tensor = tf.convert_to_tensor(decoder_input, dtype=tf.int32)
        decoder_output,h,c=decoder_model(
            {'dec_single_input':decoder_input_tensor,
            'enc_out':encoder_output,
            'dec_h':h,
            'dec_c':c},
            verbose=0
        )
        word_idx=int(np.argmax(decoder_output[0,0,:]))
        word=output_dict.get(word_idx,'')

        if word_idx==end_index or word_idx==0:
            break
        output_word.append(word)
        
        decoder_input=np.array([[word_idx]])
        h = tf.convert_to_tensor(h, dtype=tf.float32)
        c = tf.convert_to_tensor(c, dtype=tf.float32)

    return " ".join(output_word)        

# print(translation('what are doing'))
