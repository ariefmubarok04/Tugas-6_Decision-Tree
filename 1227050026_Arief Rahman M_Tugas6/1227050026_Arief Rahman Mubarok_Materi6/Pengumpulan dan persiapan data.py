# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load data from CSV
df = pd.read_csv("1227050026_Arief Rahman Mubarok_Materi6/WineQT.csv")  # Sesuaikan nama file jika berbeda

# %%
# Tampilkan info dasar dan statistik deskriptif
print(df.head(10))            # Menampilkan 10 data pertama
print(df.describe().T)        # Statistik ringkasan transpos
print(df.info())              # Info tipe data
