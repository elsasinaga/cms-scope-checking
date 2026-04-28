import numpy as np
import tensorflow as tf
import pickle
import data_helpers
from keras.utils import pad_sequences

sequence_length = 350 # Harus sama dengan saat training

# 1. Load Model dan Aset
print("Memuat sistem Desk Evaluation...")
model = tf.keras.models.load_model("model_desk_evaluation.keras")

with open("doc_assets.pkl", "rb") as f:
    assets = pickle.load(f)
    
vocabulary_inv = assets["vocab"]
label_names = assets["labels"]
thresholds = assets["thresholds"]

# Buat ulang dictionary vocabulary
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

# 2. Fungsi Prediksi Paper Baru
def evaluasi_paper(judul, abstrak):
    # Gabungkan dan bersihkan teks
    teks_gabungan = judul + " " + abstrak
    teks_bersih = data_helpers.clean_str(teks_gabungan).split(" ")
    
    # Ubah ke angka (indexing) & padding
    x_mapped = [[vocabulary.get(word, 0) for word in teks_bersih]]
    x_final = pad_sequences(x_mapped, maxlen=sequence_length, padding="post", truncating="post")
    
    # Prediksi dengan model
    probabilitas = model.predict(x_final)[0]
    
    print("\n" + "="*40)
    print("HASIL DESK EVALUATION")
    print("="*40)
    print(f"Judul: {judul}")
    
    # Cek apakah DITOLAK (Out-of-Scope)
    is_out_of_scope = True
    for i in range(len(label_names)):
        print(f"-> Skor {label_names[i]}: {probabilitas[i]:.4f} (Batas: {thresholds[i]:.4f})")
        if probabilitas[i] >= thresholds[i]:
            is_out_of_scope = False
            
    if is_out_of_scope:
        print("\nKESIMPULAN: ❌ OUT-OF-SCOPE (Topik tidak dikenali/Ditolak)")
    else:
        idx_tertinggi = np.argmax(probabilitas)
        print(f"\nKESIMPULAN: ✅ DITERIMA. Topik: {label_names[idx_tertinggi]}")

# ==========================================
# 3. MARI KITA UJI!
# ==========================================

# ==============================================================================
# BAGIAN 1: PENGUJIAN IN-SCOPE (HARUS DITERIMA OLEH SISTEM)
# Topik Latih: Artificial Intelligence, Energy Engineering, Geology
# ==============================================================================

# ---------------------------------------------------------
# IN-SCOPE EASY (Sangat Jelas) - Topik: Geology
# Alasan: Menggunakan kosakata murni Geologi tanpa tumpang tindih.
# ---------------------------------------------------------
judul_in_easy = "Tectonic Plate Movement and Sedimentary Rock Formation in the Alpine Fault"
abstrak_in_easy = "This research investigates the geological processes driving tectonic plate movement along the Alpine Fault. Through extensive field surveys and stratigraphy analysis, we examined the layers of sedimentary rock and crustal deformation. The findings provide a clear understanding of mineral composition and earth crust dynamics during the early Paleozoic era."

print("\n[TEST IN-SCOPE - EASY] Murni Geology")
evaluasi_paper(judul_in_easy, abstrak_in_easy)


# ---------------------------------------------------------
# IN-SCOPE MEDIUM (Sedikit Multidisiplin) - Topik: Energy Engineering
# Alasan: Membahas energi, namun ada sedikit singgungan tentang infrastruktur 
# dan efisiensi mekanik. Model harus tetap yakin ini adalah Energy.
# ---------------------------------------------------------
judul_in_medium = "Efficiency Analysis of Wind Turbine Aerodynamics and Power Grid Integration"
abstrak_in_medium = "Energy engineering requires continuous optimization of renewable sources. We analyze the aerodynamic performance of offshore wind turbines and their integration into the national power grid. The study measures thermodynamic efficiency, voltage conversion, and electrical load distribution. Results indicate that modifying the turbine blade angles increases power generation output by 12%."

print("\n[TEST IN-SCOPE - MEDIUM] Energy Engineering (Angin & Grid)")
evaluasi_paper(judul_in_medium, abstrak_in_medium)


# ---------------------------------------------------------
# IN-SCOPE HARD (Sangat Menjebak) - Topik: Artificial Intelligence
# Alasan: Paper ini menerapkan AI untuk menganalisis data Geologi. 
# Terdapat banyak kata Geologi ("seismic", "rock"), namun kontribusi 
# UTAMA paper ini adalah algoritma CNN-nya. Model harus bisa menebak 
# ini sebagai AI, bukan Geology.
# ---------------------------------------------------------
judul_in_hard = "A Deep Learning Framework for Autonomous Fault Detection in Seismic Data"
abstrak_in_hard = "Artificial intelligence is transforming geological surveys. In this paper, we propose a novel Convolutional Neural Network (CNN) to automatically classify seismic fault lines. While the dataset consists of geological rock patterns and tectonic shifts, the core contribution is our novel gradient descent optimization algorithm. This machine learning approach reduces model training loss and improves prediction accuracy to 96%."

print("\n[TEST IN-SCOPE - HARD] AI diterapkan di Geologi")
evaluasi_paper(judul_in_hard, abstrak_in_hard)



# ==============================================================================
# BAGIAN 2: PENGUJIAN OUT-OF-SCOPE (HARUS DITOLAK OLEH SISTEM)
# ==============================================================================

# ---------------------------------------------------------
# OOS EASY (Sangat Jelas Ditolak) - Topik: Kedokteran Anak (Pediatrics)
# Alasan: Kosakata medis murni. Pasti ditolak karena *Out-of-Vocabulary* (OOV).
# ---------------------------------------------------------
judul_oos_easy = "Dietary Interventions for Childhood Obesity and Metabolic Health"
abstrak_oos_easy = "This clinical study evaluates the impact of nutritional diets on childhood obesity. We monitored pediatric patients over twelve months, measuring body mass index, blood sugar levels, and insulin resistance. The medical results demonstrate that strict dietary interventions significantly improve the overall metabolic health of children."

print("\n[TEST OOS - EASY] Kedokteran Anak")
evaluasi_paper(judul_oos_easy, abstrak_oos_easy)


# ---------------------------------------------------------
# OOS MEDIUM (Mulai Mengecoh) - Topik: Ekonomi Bisnis / Keuangan
# Alasan: Menggunakan kata "Energy" dan "Artificial Intelligence", TAPI 
# dalam konteks investasi saham dan pasar uang.
# ---------------------------------------------------------
judul_oos_medium = "Economic Forecasting of Global Energy Markets and Tech Investments"
abstrak_oos_medium = "This paper analyzes the financial fluctuations in global energy markets and the economic impact of investing in artificial intelligence startups. We use econometric models to predict stock market trends and corporate profitability. Although energy sectors and AI technologies are growing, the monetary risks and geopolitical policies play a heavier role in determining market stability."

print("\n[TEST OOS - MEDIUM] Ekonomi / Keuangan Saham")
evaluasi_paper(judul_oos_medium, abstrak_oos_medium)


# ---------------------------------------------------------
# OOS HARD (Jebakan Maksimal) - Topik: Arsitektur / Tata Kota
# Alasan: Ini sangat bahaya. Terdapat kata "Energy consumption" (memancing kelas Energy), 
# "Soil mechanics" (memancing kelas Geology), dan "Smart sensors" (memancing kelas AI). 
# Tapi fokus aslinya adalah perencanaan tata ruang kota dan kebijakan bangunan.
# ---------------------------------------------------------
judul_oos_hard = "Sustainable Urban Planning: Smart Infrastructure and Subsurface Foundation Design"
abstrak_oos_hard = "Rapid urbanization demands smart city infrastructure. This paper evaluates the architectural design of subsurface building foundations in urban areas. We assess the energy consumption of these green buildings and the soil mechanics required to support them. Despite the use of smart artificial sensors and geothermal heating concepts, the focus of this research is strictly on urban architectural planning, zoning policies, and civil construction aesthetics."

print("\n[TEST OOS - HARD] Arsitektur / Tata Kota")
evaluasi_paper(judul_oos_hard, abstrak_oos_hard)