"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf
"""

import numpy as np
from w2v import train_word2vec  # KEMBALI MENGGUNAKAN KAMUS LOKAL
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Conv1D, Embedding, Concatenate
from collections import Counter
import itertools
import pickle
from keras.utils import pad_sequences
import data_helpers

np.random.seed(0)

# ---------------------- Parameters section -------------------
model_type = "CNN-non-static"  
data_source = "local_dir"  

# HYPERPARAMETER OPTIMIZED DARI EKSPERIMEN SEBELUMNYA
embedding_dim = 300
filter_sizes = (3, 4, 5)
num_filters = 100           # <-- Dioptimasi ke 100 agar tidak overfitting
dropout_prob = (0.5, 0.8)   # <-- Dipertahankan untuk regularisasi ketat
hidden_dims = 250 

batch_size = 64
num_epochs = 20             # <-- Cukup 20 saja agar tidak kelelahan
sequence_length = 350
max_words = 10000

min_word_count = 1
context = 15

# -------------------------------------------------------------
# 1. FUNGSI LOAD DATA
# -------------------------------------------------------------
def load_data(data_source):
    if data_source == "local_dir":
        x_train_raw, y_train, label_names = data_helpers.load_data_and_labels("data/Train_Data.csv")
        x_val_raw, y_val, _ = data_helpers.load_data_and_labels("data/Validation_Data.csv")
        x_test_raw, y_test, _ = data_helpers.load_data_and_labels("data/Testing_Data.csv")
        
        all_words = list(itertools.chain(*x_train_raw))
        word_counts = Counter(all_words)

        vocabulary_inv = [word[0] for word in word_counts.most_common(max_words - 1)]
        vocabulary_inv.insert(0, "<PAD/>")
        vocabulary = {word: i for i, word in enumerate(vocabulary_inv)}

        def process_sentences(sentences):
            mapped = []
            for sentence in sentences:
                mapped.append([vocabulary.get(word, 0) for word in sentence])
            return pad_sequences(mapped, maxlen=sequence_length, padding="post", truncating="post")

        x_train = process_sentences(x_train_raw)
        x_val = process_sentences(x_val_raw)
        x_test = process_sentences(x_test_raw)

        global num_classes
        num_classes = y_train.shape[1]

        return x_train, y_train, x_val, y_val, x_test, y_test, vocabulary_inv, label_names

# -------------------------------------------------------------
# 2. EKSEKUSI DATA PREPARATION
# -------------------------------------------------------------
print("Load data...")
x_train, y_train, x_val, y_val, x_test, y_test, vocabulary_inv, label_names = load_data(data_source)

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_val shape:", x_val.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# -------------------------------------------------------------
# 3. TRAINING CUSTOM WORD2VEC (KAMUS LOKAL)
# -------------------------------------------------------------
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    print("Membangun kamus mandiri (Custom Word2Vec)...")
    # Stack train, val, dan test untuk training word2vec
    embedding_weights = train_word2vec(np.vstack((x_train, x_val, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_val = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_val])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])

# -------------------------------------------------------------
# 4. BANGUN ARSITEKTUR MODEL (BUILD MODEL)
# -------------------------------------------------------------
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters, kernel_size=sz, padding="valid", activation="relu", strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(num_classes, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# -------------------------------------------------------------
# 5. SUNTIKKAN BOBOT WORD2VEC LOKAL & MULAI TRAINING
# -------------------------------------------------------------
if model_type == "CNN-non-static":
    # Ubah format dictionary dari w2v.py ke dalam bentuk array numpy
    weights = np.array([v for v in embedding_weights.values()])
    print("Menginisialisasi layer embedding dengan bobot Custom Word2Vec, bentuk:", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

print("\nMemulai proses Training...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_val, y_val), verbose=2)

# -------------------------------------------------------------
# 6. TAHAP AKHIR: MENGHITUNG THRESHOLD DOC & MENYIMPAN MODEL
# -------------------------------------------------------------
print("\nMenghitung Ambang Batas (Threshold) Gaussian DOC (Strict Paper Version)...")
y_pred_train = model.predict(x_train)
thresholds = []

for i in range(num_classes):
    pos_scores = y_pred_train[y_train[:, i] == 1, i]
    mirror_points = 1 + (1 - pos_scores)
    all_points = np.concatenate([pos_scores, mirror_points])
    sigma_i = np.std(all_points)
    
    alpha = 3
    t_i = max(0.5, 1 - alpha * sigma_i)
    thresholds.append(t_i)

print("Thresholds per kelas:", thresholds)

print("\nMenyimpan model dan kamus kata...")
model.save("model_desk_evaluation.keras")

with open("doc_assets.pkl", "wb") as f:
    pickle.dump({
        "vocab": vocabulary_inv,
        "labels": label_names,
        "thresholds": thresholds
    }, f)

print("Selesai! Model dan Threshold siap digunakan untuk pengujian Out-of-Scope.")