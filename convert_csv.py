import pandas as pd

# ✅ 1. Convert KDDTrain+.txt to proper CSV
df_train = pd.read_csv(
    r"C:\OneDrive\Desktop\NEWP\Data\nsl-kdd\KDDTrain+.txt",
    header=None,
    sep=","
)
df_train.to_csv(
    r"C:\OneDrive\Desktop\NEWP\Data\nsl-kdd\KDDTrain+.csv",
    index=False
)
print("✅ Converted to CSV: KDDTrain+.csv")

# ✅ 2. Convert KDDTest+.txt to proper CSV
df_test = pd.read_csv(
    r"C:\OneDrive\Desktop\NEWP\Data\nsl-kdd\KDDTest+.txt",
    header=None,
    sep=","
)
df_test.to_csv(
    r"C:\OneDrive\Desktop\NEWP\Data\nsl-kdd\KDDTest+.csv",
    index=False
)
print("✅ Converted to CSV: KDDTest+.csv")
