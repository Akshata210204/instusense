import joblib
from preprocessing import map_attack
import pandas as pd

train_file = "C:/OneDrive/Desktop/NEWP/Data/nsl-kdd/KDDTrain+.csv"
df = pd.read_csv(train_file)

# Apply map_attack like training
df["mapped"] = df["label"].apply(map_attack)

encoder = joblib.load("label_encoder.pkl")
print("Saved encoder classes:", encoder.classes_)
print("Mapped labels in CSV:", df["mapped"].unique())
