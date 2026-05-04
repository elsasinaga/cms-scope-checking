import pandas as pd
from sklearn.utils import shuffle
import os

# Pastikan folder data tersedia
if not os.path.exists("data"):
    os.makedirs("data")

# 1. Baca dataset utamamu
# Sesuaikan nama file ini jika berbeda
df = pd.read_csv("data/Data_Final.csv") 

train_list = []
val_list = []
test_list = []

# 2. Ambil daftar label unik 
labels = df['label'].unique()

for label in labels:
    df_label = df[df['label'] == label]
    
    # Acak urutan datanya
    df_label = shuffle(df_label, random_state=42)
    
    # Bagi berdasarkan jumlah yang diinginkan (600 Train, 100 Val, 300 Test)
    train_data = df_label.iloc[:600]        
    val_data = df_label.iloc[600:700]       
    test_data = df_label.iloc[700:1000]      
    
    train_list.append(train_data)
    val_list.append(val_data)
    test_list.append(test_data)

# 3. Gabungkan dan acak lagi agar antar label bercampur
df_train = pd.concat(train_list).sample(frac=1, random_state=42).reset_index(drop=True)
df_val = pd.concat(val_list).sample(frac=1, random_state=42).reset_index(drop=True)
df_test = pd.concat(test_list).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Simpan menjadi 3 file berbeda
df_train.to_csv("data/Train_Data.csv", index=False)
df_val.to_csv("data/Validation_Data.csv", index=False)
df_test.to_csv("data/Testing_Data.csv", index=False)

print("Split data berhasil! File tersimpan di folder 'data/'")