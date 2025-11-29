import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your extracted features
df = pd.read_csv("train_features.csv")

# ------------------------------
# 1. Basic Overview
# ------------------------------
print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe())

# ------------------------------
# 2. Check class distribution
# ------------------------------
plt.figure(figsize=(5,4))
sns.countplot(x=df['label'])
plt.title("Class Distribution (Real vs Fake)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# ------------------------------
# 3. Histogram of sample features
# ------------------------------
sample_features = df.columns[2:7]  # f1 to f5 (change if needed)

df[sample_features].hist(figsize=(12, 8), bins=30)
plt.suptitle("Histogram of Sample Extracted Features (f1 - f5)")
plt.show()

# ------------------------------
# 4. Boxplots of sample features
# ------------------------------
plt.figure(figsize=(12, 6))
df[sample_features].boxplot()
plt.title("Boxplot of Sample Features (Checking Outliers)")
plt.xlabel("Features")
plt.ylabel("Value")
plt.show()

# ------------------------------
# 5. Correlation Heatmap (subset to avoid huge size)
# ------------------------------
subset = df.iloc[:, 2:22]  # 20 feature columns (f1–f20)

plt.figure(figsize=(14, 10))
sns.heatmap(subset.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap of Sample Features (20 features)")
plt.show()

# ------------------------------
# 6. Pairplot (optional – slow)
# ------------------------------
# sns.pairplot(df[sample_features])
# plt.show()
