# %%
# Import library yang diperlukan
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("1227050026_Arief Rahman Mubarok_Materi6/WineQT.csv")


# Pisahkan fitur dan target
x = df.drop("quality", axis=1)
y = df["quality"]

# Bagi data: 70% training, 30% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

# Buat model dan latih
model = DecisionTreeClassifier(random_state=10)
model.fit(x_train, y_train)

# Prediksi
y_pred = model.predict(x_test)

# Classification Report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))

sns.set(font_scale=1.4)  # for label size
sns.heatmap(cm, ax=ax, annot=True, annot_kws={"size": 16}, cmap="Blues")  # font size

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
