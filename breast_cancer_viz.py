import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

#Load the dataset
data = load_breast_cancer()

#create dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)

#Add diagnostics
df["diagnosis"] = data.target_names[data.target]

print(df.head())
print(df.shape)

#Create the first visualization
# Create the first visualization
sns.countplot(
    data=df,
    x="diagnosis",
    hue="diagnosis",
    palette=["red", "blue"],
    legend=False
)

#Boxplot
plt.title("Breast Cancer Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()

sns.boxplot(
    data=df,
    x="diagnosis",
    y="mean radius",
    hue="diagnosis",
    palette=["red", "blue"],
    legend=False
)

plt.title("Tumor Size by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Mean Radius")

plt.tight_layout()
plt.show()

#Mean Radius vs Mean Texture
# Create the third visualization
sns.scatterplot(
    data=df,
    x="mean radius",
    y="mean texture",
    hue="diagnosis"
)

plt.title("Mean Radius vs Mean Texture")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")

plt.tight_layout()
plt.show()

#Calculate Correlation
features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]
correlation = df[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Between Tumor Measurements")
plt.tight_layout()
plt.show()