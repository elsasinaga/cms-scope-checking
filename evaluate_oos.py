import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils import pad_sequences
import re
import data_helpers # Gunakan helper untuk membaca data In-Scope

# --- Fungsi Preprocessing Teks ---
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
# ---------------------------------

print("Memuat aset (Kamus kata, Label, dan THRESHOLD)...")
with open("doc_assets.pkl", "rb") as f:
    assets = pickle.load(f)
    
vocabulary_inv = assets["vocab"]
label_names = assets["labels"]
thresholds = assets["thresholds"] 
vocabulary = {word: i for i, word in enumerate(vocabulary_inv)}

print("Memuat Model ML...")
model = tf.keras.models.load_model("model_desk_evaluation.keras")
sequence_length = model.input_shape[1] 

# ==============================================================
# 1. PERSIAPAN DATA IN-SCOPE (Testing_Data.csv)
# ==============================================================
print("Memproses Data In-Scope...")
x_test_raw, y_test, _ = data_helpers.load_data_and_labels("data/Testing_Data.csv")

# Ekstrak 'True Label' dari format One-Hot Encoding ke nama string
true_labels_in = [label_names[np.argmax(y)] for y in y_test]

# Ubah teks ke angka (Padding)
x_mapped_in = [[vocabulary.get(word, 0) for word in sentence] for sentence in x_test_raw]
x_in = pad_sequences(x_mapped_in, maxlen=sequence_length, padding="post", truncating="post")

# ==============================================================
# 2. PERSIAPAN DATA OUT-OF-SCOPE (Testing_Data2.csv)
# ==============================================================
print("Memproses Data Out-of-Scope...")
df_oos = pd.read_csv("data/Testing_Data2.csv", encoding='latin-1')
df_oos['text'] = df_oos['Item Title'].astype(str) + " " + df_oos['Abstract'].astype(str)
x_oos_text = [clean_str(sent).split(" ") for sent in df_oos['text'].tolist()]

# Ubah teks ke angka (Padding)
x_mapped_oos = [[vocabulary.get(word, 0) for word in sentence] for sentence in x_oos_text]
x_oos = pad_sequences(x_mapped_oos, maxlen=sequence_length, padding="post", truncating="post")

# True Label untuk OOS semuanya adalah "OUT OF SCOPE"
true_labels_oos = ["OUT OF SCOPE"] * len(x_oos)

# ==============================================================
# 3. FUNGSI PREDIKSI DOC (Deep Open Classification)
# ==============================================================
def predict_doc(x_data):
    y_pred_prob = model.predict(x_data, verbose=0)
    predicted_labels = []
    statuses = []
    
    for prob in y_pred_prob:
        max_idx = np.argmax(prob) # Cari topik dengan probabilitas tertinggi
        
        # Cek apakah probabilitas tertinggi itu lolos ambang batas (threshold) topiknya
        if prob[max_idx] >= thresholds[max_idx]:
            predicted_labels.append(label_names[max_idx])
            statuses.append("IN SCOPE")
        else:
            predicted_labels.append("OUT OF SCOPE")
            statuses.append("OUT OF SCOPE")
            
    return predicted_labels, statuses

print("\nMelakukan Prediksi pada Data In-Scope...")
pred_in_labels, status_in = predict_doc(x_in)

print("Melakukan Prediksi pada Data Out-of-Scope...")
pred_oos_labels, status_oos = predict_doc(x_oos)

# ==============================================================
# 4. GABUNGKAN KE DATAFRAME `df_result`
# ==============================================================
# Buat DataFrame In-Scope
df_result_in = pd.DataFrame({
    "True Label": true_labels_in,
    "Predicted Label": pred_in_labels,
    "Status": status_in
})

# Buat DataFrame Out-of-Scope
df_result_oos = pd.DataFrame({
    "True Label": true_labels_oos,
    "Predicted Label": pred_oos_labels,
    "Status": status_oos
})

# Gabungkan Keduanya
df_result = pd.concat([df_result_in, df_result_oos], ignore_index=True)


# ==============================================================
# 5. EVALUASI LENGKAP (KODE PANDAS MILIKMU)
# ==============================================================

# ================================
# 1. IN-SCOPE EVALUATION
# ================================

df_eval = df_result[df_result["True Label"] != "OUT OF SCOPE"].copy()

df_eval["Correct"] = df_eval["True Label"] == df_eval["Predicted Label"]

summary_in = df_eval.groupby("True Label").agg(
    total_data=("True Label", "count"),
    correct=("Correct", "sum")
)

summary_in["wrong"] = summary_in["total_data"] - summary_in["correct"]
summary_in["accuracy"] = summary_in["correct"] / summary_in["total_data"]


# ================================
# 2. OUT-OF-SCOPE EVALUATION
# ================================

df_out_eval = df_result[df_result["True Label"] == "OUT OF SCOPE"].copy()

out_total = len(df_out_eval)

out_correct = len(df_out_eval[
    df_out_eval["Status"] == "OUT OF SCOPE"
])

out_wrong = out_total - out_correct

out_accuracy = out_correct / out_total if out_total > 0 else 0

summary_out = pd.DataFrame({
    "total_data": [out_total],
    "correct": [out_correct],
    "wrong": [out_wrong],
    "accuracy": [out_accuracy]
}, index=["OUT OF SCOPE"])


# ================================
# 3. GABUNGKAN
# ================================

final_summary = pd.concat([summary_in, summary_out])


# ================================
# 4. PRINT HASIL
# ================================

print("\n" + "="*50)
print("=== RINGKASAN PER TOPIK (IN-SCOPE) ===")
print("="*50)
print(summary_in.to_string(formatters={'accuracy': '{:.2%}'.format}))

print("\n" + "="*50)
print("=== RINGKASAN OUT-OF-SCOPE ===")
print("="*50)
print(summary_out.to_string(formatters={'accuracy': '{:.2%}'.format}))

print("\n" + "="*50)
print("=== RINGKASAN FINAL ===")
print("="*50)
print(final_summary.to_string(formatters={'accuracy': '{:.2%}'.format}))

# Hitung Rata-Rata Akurasi Keseluruhan (Macro Average)
mean_accuracy = final_summary["accuracy"].mean()
print(f"\n=> RATA-RATA AKURASI KESELURUHAN (Macro Average): {mean_accuracy:.2%}")

# ================================
# 5. SIMPAN KE CSV
# ================================

summary_in.to_csv("summary_in_scope.csv")
summary_out.to_csv("summary_out_scope.csv")
final_summary.to_csv("summary_final.csv")
df_result.to_csv("detail_prediksi_lengkap.csv", index=False) # Bonus: Simpan detail prediksi per baris naskah

print("\n✅ Summary dan Detail Prediksi berhasil disimpan ke CSV.")