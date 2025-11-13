
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------
# Load files
# -----------------------
test_file = "C:/OneDrive/Desktop/NEWP/Data/nsl-kdd/KDDTest_fixed.csv"  # your fixed test file
model_file = "bilstm_ids.keras"
scaler_file = "scaler.pkl"
label_encoder_file = "label_encoder.pkl"
columns_file = "training_columns.pkl"

# Load test dataset
df_test = pd.read_csv(test_file)
df_test.reset_index(drop=True, inplace=True)

# Drop difficulty column if it exists
if "difficulty" in df_test.columns:
    df_test = df_test.drop(columns=["difficulty"])

# Load saved objects
model = load_model(model_file)
scaler = joblib.load(scaler_file)
label_encoder = joblib.load(label_encoder_file)
training_columns = joblib.load(columns_file)

# -----------------------
# Prediction function
# -----------------------
def predict_from_row(row_number):
    # Extract the row
   #40 row = df_test.iloc[row_number].copy()
    # Extract the row
    row = df_test.iloc[row_number - 2].copy()   # subtract 1 so row 40 means actual 40th data row


    # Separate features and actual label
    actual_label = row["label"]
    row = row.drop("label")

    # One-hot encode categorical features
    row_df = pd.DataFrame([row], columns=df_test.columns[:-1])  # exclude label
    row_df = pd.get_dummies(row_df, columns=["protocol_type", "service", "flag"])

    # Match training columns
    row_df = row_df.reindex(columns=training_columns, fill_value=0)

    # Scale
    row_scaled = scaler.transform(row_df)

    # Reshape for BiLSTM
    row_scaled = np.expand_dims(row_scaled, axis=1)

    # Predict
    preds = model.predict(row_scaled)
    pred_class = np.argmax(preds, axis=1)
    pred_label = label_encoder.inverse_transform(pred_class)[0]

    print(f"\nRow {row_number}:")
    print(f"✅ Predicted: {pred_label}")
    print(f"🎯 Actual: {actual_label}")

# -----------------------
# Example usage
# -----------------------
row_number = int(input("Enter row number from test dataset: "))
predict_from_row(row_number)


