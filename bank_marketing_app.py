import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Fungsi untuk mengganti nilai 'unknown' dengan modus
def replace_unknown_with_mode(column):
    mode = column[column != 'unknown'].mode()[0]
    return column.replace('unknown', mode)

# Fungsi untuk evaluasi model
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Fungsi untuk plot confusion matrix
def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# Aplikasi Streamlit
st.title("Bank Marketing Analysis")
st.write("Aplikasi ini melakukan analisis, preprocessing, dan evaluasi model untuk dataset Bank Marketing.")

# 1. Load Dataset
st.header("1. Load Dataset")
url = "https://raw.githubusercontent.com/msaifulhuda/bank-marketing/main/dataset/bank.csv"
data = pd.read_csv(url, sep=';')
st.write("Dataset Loaded:")
st.dataframe(data.head())

# 2. Preprocessing
st.header("2. Preprocessing")
data_cleaned = data.copy()

# Replace 'unknown' values
for col in data_cleaned.columns:
    if data_cleaned[col].dtype == 'object':
        data_cleaned[col] = replace_unknown_with_mode(data_cleaned[col])

st.write("Dataset setelah preprocessing (unknown diganti):")
st.dataframe(data_cleaned.head())

# Encoding
st.write("Encoding kolom kategorikal:")
categorical_columns = data_cleaned.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    data_cleaned[col] = le.fit_transform(data_cleaned[col])
st.write("Dataset setelah encoding:")
st.dataframe(data_cleaned.head())

# Split dataset
X = data_cleaned.drop('y', axis=1)
y = data_cleaned['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Building
st.header("3. Model Building")

# KNN
st.subheader("KNN")
k = st.slider("Pilih nilai k untuk KNN:", min_value=1, max_value=15, value=7, step=1)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
st.write(f"Akurasi model KNN (k={k}):")
accuracy_knn, precision_knn, recall_knn, f1_knn = evaluate_model(y_test, y_pred_knn)

# Random Forest
st.subheader("Random Forest")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
st.write("Hasil evaluasi Random Forest:")
accuracy_rf, precision_rf, recall_rf, f1_rf = evaluate_model(y_test, y_pred_rf)

# 4. Visualisasi Hasil
st.header("4. Visualisasi Hasil")
st.write("Confusion Matrix KNN:")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_knn), "Confusion Matrix - KNN")

st.write("Confusion Matrix Random Forest:")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), "Confusion Matrix - Random Forest")

# 5. Perbandingan Model
st.header("5. Perbandingan Model")
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "KNN": [accuracy_knn, precision_knn, recall_knn, f1_knn],
    "Random Forest": [accuracy_rf, precision_rf, recall_rf, f1_rf]
})
st.write("Tabel Perbandingan:")
st.dataframe(comparison)

# Visualisasi
fig, ax = plt.subplots()
comparison.set_index("Metric").plot(kind='bar', ax=ax, figsize=(10, 6))
plt.title("Perbandingan Performa Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
st.pyplot(fig)
