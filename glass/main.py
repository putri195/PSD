import streamlit as st
import numpy as np

# Title of the app
st.title('Prediksi Kaca')

# Description of the app
st.write("""
### Deskripsi:
Aplikasi ini memprediksi jenis kaca berdasarkan indeks bias dan komposisi elemen.
Masukkan nilai-nilai di bawah ini untuk mendapatkan prediksi jenis kaca.
""")

# Input fields for the features
ri = st.number_input('RI: Refractive Index', format="%.5f")
na = st.number_input('Na: Sodium (weight percent)', format="%.2f")
mg = st.number_input('Mg: Magnesium (weight percent)', format="%.2f")
al = st.number_input('Al: Aluminum (weight percent)', format="%.2f")
si = st.number_input('Si: Silicon (weight percent)', format="%.2f")
k = st.number_input('K: Potassium (weight percent)', format="%.2f")
ca = st.number_input('Ca: Calcium (weight percent)', format="%.2f")
ba = st.number_input('Ba: Barium (weight percent)', format="%.2f")
fe = st.number_input('Fe: Iron (weight percent)', format="%.2f")

# Mapping of output numbers to glass types
output_mapping = {
    1: 'building_windows_float_processed',
    2: 'building_windows_non_float_processed',
    3: 'vehicle_windows_float_processed',
    4: 'vehicle_windows_non_float_processed',
    5: 'containers',
    6: 'tableware',
    7: 'headlamps'
}

def dummy_prediction(input_data):
    # Dummy logic for demonstration
    # Here we use a simple rule-based approach just for demonstration purposes
    if input_data[0][0] < 1.515:  # Example rule based on RI
        return 1
    elif input_data[0][1] > 13:  # Example rule based on Na
        return 2
    elif input_data[0][2] > 3:  # Example rule based on Mg
        return 3
    elif input_data[0][3] > 1:  # Example rule based on Al
        return 4
    elif input_data[0][4] > 72:  # Example rule based on Si
        return 5
    elif input_data[0][5] > 0.5:  # Example rule based on K
        return 6
    else:
        return 7  # Default case

# Placeholder for prediction logic (to be implementpiped)
if st.button('Predict'):
    # Create a numpy array with the input values
    input_data = np.array([[ri, na, mg, al, si, k, ca, ba, fe]])

    # Make a prediction using dummy logic
    predicted_class = dummy_prediction(input_data)

    # Display the prediction
    st.write("**Hasil Prediksi:**")
    st.write(f"Tipe kaca yang diprediksi: {output_mapping[predicted_class]}")
