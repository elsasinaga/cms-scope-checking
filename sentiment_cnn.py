"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
from w2v import train_word2vec
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Conv1D, Embedding, Concatenate, GlobalMaxPooling1D
from collections import Counter
import itertools
import pickle
# Khusus untuk sequence, di Keras versi baru posisinya sedikit bergeser ke utils
from keras.utils import pad_sequences
from keras.datasets import imdb

import scipy.stats as stats
import data_helpers
np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 4, 5)
num_filters = 150
dropout_prob = (0.5, 0.8)
hidden_dims = 250 #fully connected layer after convolutional layers, before output layer

# Training parameters
batch_size = 64
num_epochs = 20

# Prepossessing parameters
sequence_length = 350
max_words = 10000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 15

## ---------------------- Parameters end -----------------------

def load_data(data_source):
    """
    Menyiapkan data training dan testing, serta membangun kamus kata (vocabulary).
    """
    if data_source == "local_dir":
        # memanggil helper untuk membaca csv
        x_raw, y, label_names = data_helpers.load_data_and_labels("data/Data_Final.csv")
        
        # membangun vocabulary (kamus kata)
        all_words = list(itertools.chain(*x_raw))
        word_counts = Counter(all_words)

        # PERBAIKAN 1: Gunakan max_words - 1 agar setelah ditambah <PAD/> jumlahnya pas
        vocabulary_inv = [word[0] for word in word_counts.most_common(max_words - 1)]
        vocabulary_inv.insert(0, "<PAD/>")
        vocabulary = {word: i for i, word in enumerate(vocabulary_inv)}

        x_mapped = []
        for sentence in x_raw:
            mapped_sentence = [vocabulary.get(word, 0) for word in sentence]
            x_mapped.append(mapped_sentence)

        x_final = pad_sequences(x_mapped, maxlen=sequence_length, padding="post", truncating="post")

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x_final[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        train_len = int(len(x_shuffled) * 0.9)
        x_train = x_shuffled[:train_len]
        y_train = y_shuffled[:train_len]
        x_test = x_shuffled[train_len:]
        y_test = y_shuffled[train_len:]

        print("Contoh isi 1 data training (harus ada angka selain 0):", x_train[0][:20])

        # PERBAIKAN 2: Simpan num_classes secara global agar model tahu jumlah kelas
        global num_classes
        num_classes = y.shape[1]

        # PERBAIKAN 3: Return 6 variabel
        return x_train, y_train, x_test, y_test, vocabulary_inv, label_names
    
# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv, label_names = load_data(data_source)

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(num_classes, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=2)

# =====================================================================
# TAHAP AKHIR: MENGHITUNG THRESHOLD DOC & MENYIMPAN MODEL
# =====================================================================
print("\nMenghitung Ambang Batas (Threshold) Gaussian DOC (Strict Paper Version)...")
y_pred_train = model.predict(x_train)
thresholds = []

# for i in range(num_classes):
#     pos_scores = y_pred_train[np.where(y_train[:, i] == 1)[0], i]
    
#     # Paper Asli: Mirroring berporos pada angka mutlak 1.0
#     mirror_points = 1.0 + (1.0 - pos_scores) 
#     combined = np.concatenate([pos_scores, mirror_points])
    
#     # Hitung deviasi standar (sigma)
#     sigma = np.std(combined)
    
#     # Paper Asli: Pengurangan dihitung dari angka 1.0
#     ti = max(0.5, 1.0 - (3.0 * sigma))
#     thresholds.append(ti)

for i in range(num_classes):
    pos_scores = y_pred_train[np.where(y_train[:, i] == 1), i]
    
    # 1. Buat titik cermin (mirror points) pada rata-rata 1
    mirror_points = 1 + (1 - pos_scores)
    
    # 2. Gabungkan data asli dengan cerminnya
    all_points = np.concatenate([pos_scores, mirror_points])
    
    # 3. Hitung standar deviasi (sigma)
    sigma_i = np.std(all_points)
    
    # 4. Tentukan threshold dengan alpha = 3
    alpha = 3
    t_i = max(0.5, 1 - alpha * sigma_i)
    
    thresholds.append(t_i)

print("Thresholds per kelas:", thresholds)

print("\nMenyimpan model dan kamus kata...")
# 1. Simpan arsitektur model dan bobotnya (Bisa dipakai kapan saja tanpa training ulang)
model.save("model_desk_evaluation.keras")

# 2. Simpan vocabulary, nama label, dan thresholds ke file pickle
with open("doc_assets.pkl", "wb") as f:
    pickle.dump({
        "vocab": vocabulary_inv,
        "labels": label_names,
        "thresholds": thresholds
    }, f)

print("Selesai! Model dan Threshold siap digunakan untuk pengujian Out-of-Scope.")