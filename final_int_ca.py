import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("toyota.csv")

print(df.head())
print(df.describe())
print(df.isnull().sum())

num = df.select_dtypes(include=np.number).columns
for col in num:
    plt.hist(df[col], bins=30)
    plt.title(col)
    plt.show()

target_col = "price" if "price" in df.columns else num[0]
df["target"] = (df[target_col] > df[target_col].median()).astype(int)

X = df.drop(columns=[target_col, "target"])
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())
y = df["target"]

selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
X = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    results[name] = [
        accuracy_score(y_test, p),
        precision_score(y_test, p),
        recall_score(y_test, p),
        f1_score(y_test, p)
    ]
    print(f"\n{name} → Acc:{results[name][0]:.3f}, Prec:{results[name][1]:.3f}, Rec:{results[name][2]:.3f}, F1:{results[name][3]:.3f}")

names = list(results.keys())
acc = [results[m][0] for m in names]
prec = [results[m][1] for m in names]
rec = [results[m][2] for m in names]
f1 = [results[m][3] for m in names]

plt.bar(names, acc)
plt.title("Accuracy Comparison")
plt.show()

x = np.arange(len(names))
w = 0.2
plt.bar(x-w, prec, w)
plt.bar(x, rec, w)
plt.bar(x+w, f1, w)
plt.xticks(x, names)
plt.legend(["Precision", "Recall", "F1"])
plt.title("Precision / Recall / F1 Comparison")
plt.show()
