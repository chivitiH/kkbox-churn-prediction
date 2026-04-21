import pandas as pd
import matplotlib.pyplot as plt

feat = pd.read_csv("artifacts/feature_importance.csv", index_col=0)

feat = feat.sort_values(by=feat.columns[0], ascending=False).head(20)

plt.figure()
feat.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top Feature Importance")
plt.tight_layout()

plt.savefig("reports/figures/feature_importance.png")

print("✅ Graph saved in reports/figures/")
