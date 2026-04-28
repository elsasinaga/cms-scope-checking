import pandas as pd

# Load data hasil clustering tadi
print("Membaca data hasil LDA...")
df = pd.read_csv("data/Dataset_Telah_Di_Cluster.csv")

print(f"Jumlah paper SEBELUM dibersihkan: {len(df)}")

# Daftar topik yang teridentifikasi sebagai "Sampah" (Sipil, Software, Internet)
topik_sampah = [0, 2, 4, 5, 6, 8, 10, 11, 13, 16, 17, 18, 19]

# Simpan hanya baris yang BUKAN bagian dari topik_sampah
df_bersih = df[~df['LDA_Cluster'].isin(topik_sampah)]

# Hapus kolom LDA_Cluster agar data kembali bersih seperti semula
df_bersih = df_bersih.drop(columns=['LDA_Cluster'])

print(f"Jumlah paper SETELAH dibersihkan: {len(df_bersih)}")

# Simpan sebagai dataset final yang sudah suci
df_bersih.to_csv("data/Soil_Sciences_Super_Bersih.csv", index=False)
print("Selesai! File 'Dataset_Super_Bersih.csv' siap digunakan untuk training CNN.")