import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.utils import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Model dan Aset DOC
print("Memuat Model dan Threshold DOC...")
model = load_model("model_desk_evaluation.keras")

with open("doc_assets.pkl", "rb") as f:
    assets = pickle.load(f)
    vocabulary = assets["vocab"]
    label_names = assets["labels"]
    thresholds = assets["thresholds"]

sequence_length = model.input_shape[1]

# 2. Load Data Uji Validasi
print("Membaca data uji validasi...")
df_test = pd.read_csv("data_uji_validasi.csv")

# GABUNGKAN Judul dan Abstrak dengan pemisah spasi
df_test['Teks_Gabungan'] = df_test['Judul'].astype(str) + " " + df_test['Abstrak'].astype(str)

# Gunakan kolom teks yang sudah digabung
teks_gabungan = df_test['Teks_Gabungan'].tolist()
label_asli = df_test['Label_Asli'].tolist()

# 3. Proses Prediksi DOC
label_prediksi = []

# Ubah variabel looping menjadi teks_gabungan
for teks in teks_gabungan:
    # Preprocessing
    teks_bersih = teks.lower().split()
    
    x_mapped = [[vocabulary.get(word, 0) for word in teks_bersih]]
    
    # Filter OOV (Cegah Bahasa Asing)
    kata_dikenali = [angka for angka in x_mapped[0] if angka != 0]
    persentase_dikenali = len(kata_dikenali) / len(teks_bersih) * 100 if len(teks_bersih) > 0 else 0
    
    if persentase_dikenali < 20.0:
        label_prediksi.append("OOS")
        continue
        
    x_final = pad_sequences(x_mapped, maxlen=sequence_length, padding="post", truncating="post")
    probabilitas = model.predict(x_final, verbose=0)[0]
    
    # Deteksi dengan Threshold DOC
    max_score = np.max(probabilitas)
    pred_class_idx = np.argmax(probabilitas)
    
    if max_score >= thresholds[pred_class_idx]:
        label_prediksi.append(label_names[pred_class_idx])
    else:
        label_prediksi.append("OOS")

# 4. Cetak Matriks Evaluasi
print("\n" + "="*50)
print("LAPORAN EVALUASI MODEL DESK EVALUATION (DOC)")
print("="*50)

# Pastikan OOS masuk ke dalam daftar target kelas
target_names = list(label_names) + ["OOS"]

print(classification_report(label_asli, label_prediksi, labels=target_names, zero_division=0))

# 5. Visualisasi Confusion Matrix (Opsional: akan membuat gambar)
cm = confusion_matrix(label_asli, label_prediksi, labels=target_names)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Desk Evaluation Model')
plt.ylabel('Label Seharusnya (Ground Truth)')
plt.xlabel('Label Prediksi Sistem')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Gambar Confusion Matrix berhasil disimpan sebagai 'confusion_matrix.png'")