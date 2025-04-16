# %%
# Import library yang diperlukan
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset (gunakan dataset yang sama seperti sebelumnya)
df = pd.read_csv("1227050026_Arief Rahman Mubarok_Materi6/WineQT.csv")

# Hapus kolom 'Id' jika ada dalam dataset
df = df.drop(columns=['Id'], errors='ignore')

# Pisahkan fitur dan target
x = df.drop("quality", axis=1)
y = df["quality"]

# Bagi data: 70% training, 30% testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Buat model dan latih
model = DecisionTreeClassifier(random_state=10)
model.fit(x_train, y_train)

# Example of creating a single Wine data point as a dictionary
wine_test_data = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

# Ensure the order of features matches the training data
feature_order = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
    'pH', 'sulphates', 'alcohol'
]

# Convert the test data into a DataFrame
prediction_input_df = pd.DataFrame([wine_test_data])

# Prediksi menggunakan model yang sudah dilatih
prediction = model.predict(prediction_input_df[feature_order])

# Tampilkan hasil prediksi
print(f"Hasil Prediksi untuk data yang diuji: {prediction}")
