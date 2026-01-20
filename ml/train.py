import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from preprocessing import load_and_preprocess
from model import build_bilstm   # ✅ Use your new Attention-based BiLSTM

# =======================================
# 1️⃣ Load and Preprocess
# =======================================
train_file = "C:/OneDrive/Desktop/NEWP/Data/nsl-kdd/KDDTrain+.csv"

X_train, y_train, scaler, label_encoder, training_columns = load_and_preprocess(
    train_file, training=True, has_header=True
)

num_classes = len(label_encoder.classes_)
print(f"✅ Loaded training data with {X_train.shape[0]} samples and {num_classes} classes.")

# =======================================
# 2️⃣ Reshape input for BiLSTM
# =======================================
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
input_shape = (X_train.shape[1], X_train.shape[2])

# =======================================
# 3️⃣ Build Model (with Attention)
# =======================================
model = build_bilstm(input_shape, num_classes)
model.summary()

# =======================================
# 4️⃣ Handle Class Imbalance
# =======================================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("\n📊 Class Weights:", class_weights)

# =======================================
# 5️⃣ Train Model
# =======================================
print("\n🚀 Starting training ...\n")
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(sss.split(X_train, y_train))

history = model.fit(
    X_train[train_idx], y_train[train_idx],
    validation_data=(X_train[val_idx], y_train[val_idx]),
    epochs=5,
    batch_size=32,
    class_weight=class_weights
)


# =======================================
# 6️⃣ Save Artifacts
# =======================================
model.save_weights("bilstm_ids.weights.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(training_columns, "training_columns.pkl")

print("\n✅ Model and preprocessing objects saved successfully!")

# =======================================
# 7️⃣ Evaluate Model
# =======================================
y_train_pred = model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

print("\n=== Training Evaluation Metrics ===")
print(f"✅ Accuracy: {accuracy_score(y_train, y_train_pred_classes):.4f}")
print("✅ Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred_classes))
print("\n✅ Classification Report:\n", classification_report(
    y_train, y_train_pred_classes,
    target_names=label_encoder.classes_,
    zero_division=0
))
