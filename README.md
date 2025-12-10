# Data-Wrangling-Final-Project

## Project Code

The full analysis script is saved in the repository as `data wrangling.qmd`. You can view or download it directly from the repo.

If you'd like the code visible inside this README, expand the block below.

<details>
<summary>View `data wrangling.qmd` (click to expand)</summary>

```qmd
---
title: "Data wrangling"
format: html
---

# Data
```{python}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# Load your dataset
df = pd.read_csv('~/Downloads/diamonds.csv')
df.head()
```

# Looking at Carat size

```{python}
import numpy as np

bins = np.linspace(df['carat'].min(), df['carat'].max(), 20)
df['carat_bin'] = pd.cut(df['carat'], bins)

carat_group = df.groupby('carat_bin')['price'].mean().reset_index()

plt.figure(figsize=(12,6))
plt.plot(carat_group['carat_bin'].astype(str), carat_group['price'], marker='o')
plt.xticks(rotation=45)
plt.title("Average Diamond Price by Carat Size (Smoothed)")
plt.xlabel("Carat (Binned)")
plt.ylabel("Average Price (USD)")
plt.tight_layout()
plt.show()
```

# Looking at price of cut

```{python}
import matplotlib.pyplot as plt
import seaborn as sns

cut_order = ["Ideal", "Good", "Very Good", "Fair", "Premium"]

cut_avg = df.groupby("cut")["price"].mean().reindex(cut_order).reset_index()

colors = ["#FFB6B9", "#A0D2FF", "#B8FFB8", "#FFE066", "#FFCC99"]

plt.figure(figsize=(10,6))
bars = plt.bar(cut_avg["cut"], cut_avg["price"], color=colors)

for bar, value in zip(bars, cut_avg["price"]):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() - 300,
        f"${value:,.0f}",
        ha='center',
        va='bottom',
        fontsize=12,
        color="black"
    )

legend_labels = ["Ideal", "Good", "Very Good", "Fair", "Premium"]
for color, label in zip(colors, legend_labels):
    plt.bar(0, 0, color=color, label=label)

plt.legend(title="Cut Types", loc="upper left")
plt.title("Average Diamond Price by Cut Quality", fontsize=14)
plt.xlabel("Cut Quality", fontsize=12)
plt.ylabel("Average Price (USD)", fontsize=12)
plt.tight_layout()
plt.show()
```

# Price by color

```{python}
import matplotlib.pyplot as plt
import numpy as np

def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_no_outliers = remove_outliers(df, "price")

color_order = ["D", "E", "F", "G", "H", "I", "J"]
price_data = [df_no_outliers[df_no_outliers["color"] == c]["price"] for c in color_order]
colors = ["#FFB6B9", "#A0D2FF", "#B8FFB8", "#FFE066", "#FFCC99", "#C8A2FF", "#FFD6A5"]

plt.figure(figsize=(12,6))
bp = plt.boxplot(price_data, labels=color_order, patch_artist=True, showfliers=False,
                boxprops=dict(linewidth=1.5), medianprops=dict(color="orange", linewidth=2),
                whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

plt.title("Price Distribution by Color Grade (Outliers Removed)", fontsize=14)
plt.xlabel("Color Grade", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.tight_layout()
plt.show()
```

# Price by clarity

```{python}
import matplotlib.pyplot as plt

def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_no_outliers = remove_outliers(df, "price")
clarity_order = ["I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"]
price_data = [df_no_outliers[df_no_outliers["clarity"] == c]["price"] for c in clarity_order]
colors = ["#FFB6B9", "#A0D2FF", "#B8FFB8", "#FFE066", "#FFCC99", "#C8A2FF", "#FFD6A5", "#B5EAD7"]

plt.figure(figsize=(12,6))
bp = plt.boxplot(price_data, labels=clarity_order, patch_artist=True, showfliers=False,
                boxprops=dict(linewidth=1.5), medianprops=dict(color="orange", linewidth=2),
                whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

plt.title("Price Distribution by Clarity Grade (Outliers Removed)", fontsize=14)
plt.xlabel("Clarity Grade", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.tight_layout()
plt.show()
```

# OLS regression supporting size

```{python}
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('~/Downloads/diamonds.csv')
X = df[['carat']]
y = df['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

```

</details>
