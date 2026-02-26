import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load("../models/best_model.pkl")

importances = model.named_steps["model"].feature_importances_

feature_names = pd.read_csv(
    "../data/creditcard.csv").drop("Class", axis=1).columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

sns.barplot(x="feature", y="importance", data=importance_df.head(10))
plt.title("Top important features")
plt.show()
