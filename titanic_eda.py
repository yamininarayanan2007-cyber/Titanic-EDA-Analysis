# =====================================
# TITANIC DATASET - FULL EDA (VS CODE)
# =====================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Create plots folder if not exists
if not os.path.exists("plots"):
    os.makedirs("plots")

# Load Dataset
df = pd.read_csv("train.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== SURVIVAL COUNT =====")
print(df["Survived"].value_counts())

print("\n===== GENDER COUNT =====")
print(df["Sex"].value_counts())

print("\n===== PASSENGER CLASS COUNT =====")
print(df["Pclass"].value_counts())


# ==============================
# VISUALIZATIONS (Saved as Images)
# ==============================

# 1. Survival Count
plt.figure()
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("plots/survival_count.png")
plt.close()

# 2. Survival by Gender
plt.figure()
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.savefig("plots/survival_by_gender.png")
plt.close()

# 3. Survival by Passenger Class
plt.figure()
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.savefig("plots/survival_by_class.png")
plt.close()

# 4. Age Distribution
plt.figure()
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("plots/age_distribution.png")
plt.close()

# 5. Boxplot - Age vs Survival
plt.figure()
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.savefig("plots/age_vs_survival.png")
plt.close()

# 6. Scatter Plot - Age vs Fare
plt.figure()
plt.scatter(df["Age"], df["Fare"])
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.savefig("plots/age_vs_fare.png")
plt.close()

# 7. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()


# ==============================
# FINAL SUMMARY
# ==============================

print("\n========== FINAL SUMMARY ==========")
print("1. More passengers died than survived.")
print("2. Females had higher survival rate than males.")
print("3. First-class passengers survived more.")
print("4. Most passengers were between age 20-40.")
print("5. Fare and Passenger Class show correlation with survival.")
print("6. Age and Cabin contain missing values.")
print("===================================")

print("\nAll graphs saved inside 'plots' folder successfully ✅")
