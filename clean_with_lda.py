import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. Load Dataset Anda
print("Membaca dataset...")
df = pd.read_csv("data/Soil_Sciences.csv")  # Ganti dengan nama file CSV Anda

# Asumsi kolom teksnya bernama 'Abstract' (sesuaikan jika namanya berbeda)
teks_data = df['Abstract'].fillna('')

# 2. Vektorisasi Teks (Mengubah teks menjadi matriks angka)
# stop_words='english' akan otomatis membuang kata hubung (the, is, and, dll)
print("Memproses teks...")
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(teks_data)

# 3. Jalankan Algoritma LDA
# Kita minta LDA memecah data menjadi 10 sub-topik (bisa diubah nanti)
jumlah_topik = 15
print(f"Menjalankan LDA untuk menemukan {jumlah_topik} sub-topik tersembunyi...")
lda = LatentDirichletAllocation(n_components=jumlah_topik, random_state=42)
lda.fit(X)

# 4. Tampilkan Kata Kunci Utama untuk Tiap Topik
print("\n" + "="*50)
print("HASIL PEMBAGIAN TOPIK (Cari mana yang berisi topik IT/IoT!)")
print("="*50)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    # Ambil 10 kata paling sering muncul di tiap topik
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    print(f"Topik {topic_idx}: {', '.join(top_words)}")

# 5. Pasangkan setiap paper dengan topik dominannya
topic_results = lda.transform(X)
df['LDA_Cluster'] = topic_results.argmax(axis=1)

# 6. Simpan hasil agar bisa dianalisis
file_output = "data/Dataset_Telah_Di_Cluster.csv"
df.to_csv(file_output, index=False)
print("\n" + "="*50)
print(f"Selesai! Data hasil pengelompokan disimpan di: {file_output}")
print("Silakan buka file tersebut di Excel/Pandas untuk menghapus topik yang salah.")