# %%
# Import library yang diperlukan
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

# Tentukan fitur untuk visualisasi (sesuaikan dengan dataset yang kamu gunakan)
features = x.columns

# %%
# Visualize tree
fig, ax = plt.subplots(figsize=(25, 20))
plot_tree(model, feature_names=features, filled=True, class_names=[str(c) for c in sorted(y.unique())])
plt.show()
