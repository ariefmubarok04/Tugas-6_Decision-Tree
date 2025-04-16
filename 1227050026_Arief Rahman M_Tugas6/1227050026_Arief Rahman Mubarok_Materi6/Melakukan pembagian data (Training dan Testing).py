# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("1227050026_Arief Rahman Mubarok_Materi6/WineQT.csv")

# Tampilkan pairplot untuk visualisasi awal
# Pastikan jumlah data tidak terlalu besar, kalau berat bisa di-skip atau subset
sns.pairplot(df.sample(200), hue='quality', palette='Set1')  # gunakan sample jika dataset besar
plt.show()

# %%
# Pisahkan fitur dan target
x = df.drop('quality', axis=1)   # 'quality' sebagai target
y = df['quality']

# Lakukan pembagian data 70% training dan 30% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Tampilkan jumlah data training
print(f"Jumlah data training: {len(x_train)}")
print(f"Jumlah data testing : {len(x_test)}")
