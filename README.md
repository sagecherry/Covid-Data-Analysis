#  COVID-19 Data Analysis
---

##  Aim

To perform an end-to-end exploratory data analysis (EDA) on a real-world COVID-19 dataset, extract meaningful insights about the global spread of the virus, and conduct a focused drill-down analysis of India's state-wise case distribution.

---

##  Objective

- Load and preprocess the `covid_19_data.csv` dataset using Pandas.
- Engineer new features such as **Active Cases** from the existing columns.
- Identify and analyse case trends for the **latest available observation date**.
- Visualise the global spread of confirmed COVID-19 cases on an **interactive world choropleth map**.
- Determine the **Top 20 countries** by confirmed and recovered cases.
- Perform a dedicated **India-specific analysis** — find the most and least affected states/UTs.

---

##  Theory

COVID-19 (Coronavirus Disease 2019), caused by the SARS-CoV-2 virus, was declared a global pandemic by the WHO in March 2020. Tracking its spread required continuous ingestion and analysis of large, multi-country datasets. This notebook applies core data science techniques — cleaning, transformation, aggregation, and visualisation — to make sense of that data.

---

##  Code Walkthrough

### Importing Libraries

```python
import pandas as pd
import numpy as np
```

`pandas` is the primary library used throughout for loading, cleaning, filtering, and grouping tabular data. `numpy` provides fast numerical operations and is used implicitly by pandas under the hood for array-level computations.

---

### Loading the Dataset

```python
data = pd.read_csv("/content/covid_19_data.csv")
data.head()
```

`pd.read_csv()` reads the CSV file into a **DataFrame** — a 2D table-like structure. `data.head()` displays the first 5 rows to quickly verify the data was loaded correctly and to inspect column names and sample values before any transformation.

---

### Dropping Irrelevant Columns

```python
data = data.drop(['SNo', 'Last Update'], axis=1)
data.head()
```

`SNo` is just a serial number with no analytical value, and `Last Update` is a redundant timestamp. Dropping them with `axis=1` (column-wise) keeps the DataFrame clean and reduces noise in subsequent operations. The updated DataFrame is confirmed with `head()`.

---

### Checking the Shape

```python
data.shape
```

Returns a tuple `(rows, columns)`. This gives a quick snapshot of how large the dataset is — useful for gauging memory requirements and understanding the scale before processing begins.

---

### Inspecting Data Types and Nulls

```python
data.info()
```

`info()` prints each column's name, its data type (`dtype`), and the count of non-null values. This is a critical first step — it reveals if numeric columns were accidentally read as `object` (string) type, and flags columns with missing values that may need imputation later.

---

### Type Casting

```python
data['ObservationDate'] = data['ObservationDate'].astype('datetime64[ns]')
data['Confirmed']        = data['Confirmed'].astype('int64')
data['Deaths']           = data['Deaths'].astype('int64')
data['Recovered']        = data['Recovered'].astype('int64')

data.info()
```

CSV files store everything as strings by default. This cell explicitly casts:
- `ObservationDate` → `datetime64` so that date comparisons (like `.max()`) work correctly on actual date values rather than alphabetical string comparisons.
- `Confirmed`, `Deaths`, `Recovered` → `int64` so that arithmetic and aggregation produce correct numeric results rather than string concatenation errors.

`data.info()` is called again to confirm the types were applied successfully.

---

### Feature Engineering: Active Cases

```python
data['Active'] = data['Confirmed'] - data['Deaths'] - data['Recovered']
data.head(50)

data.iloc[50:100]
```

A new column `Active` is derived by subtracting Deaths and Recovered from Confirmed cases — representing people currently still infected at any given point. `data.head(50)` shows the first 50 rows to verify the new column, while `data.iloc[50:100]` uses **integer-location based indexing** to slice and inspect rows 50 through 99 — useful for verifying data integrity across different records.

---

### Finding the Latest Date

```python
data['ObservationDate'].max()
```

Since the dataset is a time-series spanning many months, `.max()` on the date column returns the **most recent observation date** in the dataset. This value is used as the reference point to extract the current state of the pandemic in subsequent cells.

---

### Filtering Latest Data

```python
latest_data = data[data['ObservationDate'] == data['ObservationDate'].max()]
latest_data.head()
```

Boolean filtering is applied to extract only rows matching the latest date. This creates a **snapshot DataFrame** (`latest_data`) representing the most current pandemic state across all countries — eliminating historical rows so aggregations don't double-count cumulative case numbers from earlier dates.

---

### Shape of Latest Data

```python
latest_data.shape
```

Checks how many rows and columns remain after the date filter. This tells us exactly how many country/region records exist for the latest observation — a good sanity check before grouping.

---

### Exploring Countries

```python
latest_data['Country/Region'].value_counts()
latest_data['Country/Region'].unique()
latest_data['Country/Region'].nunique()
```

Three exploratory checks on the `Country/Region` column:
- `value_counts()` — shows how many rows exist per country. Countries with multiple province-level entries (like China or Australia) will appear with a count > 1.
- `unique()` — lists every distinct country name, useful for spotting naming inconsistencies (e.g. `"Mainland China"` vs `"China"`).
- `nunique()` — returns the total count of unique countries, confirming the global coverage of the dataset.

---

### Country-Level Aggregation

```python
countries = latest_data.groupby("Country/Region")[["Confirmed", "Deaths", "Recovered", "Active"]].sum()
countries = countries.reset_index()
countries.head()
```

`groupby()` groups all rows sharing the same country name, and `.sum()` collapses multiple province-level rows into a single national total. `reset_index()` then promotes the group key (`Country/Region`) back into a regular column instead of leaving it as the DataFrame index — making filtering and plotting easier downstream.

---

### Spot Checking Specific Countries

```python
countries[countries['Country/Region'] == 'India']
countries[countries['Country/Region'] == 'US']
countries[countries['Country/Region'] == 'Mainland China']
```

Boolean filtering retrieves the aggregated case counts for three of the most-discussed nations — **India**, **US**, and **Mainland China** — for a quick manual sanity check that the groupby aggregation produced plausible totals before moving on to visualisation.

---

### World Map Visualisation

```python
import plotly.express as px

world_map = px.choropleth(
    countries,
    locations              = "Country/Region",
    locationmode           = "country names",
    color                  = "Confirmed",
    color_continuous_scale = "mint",
    range_color            = [0, 10_000_000]
)
world_map.show()
```

`plotly.express.choropleth()` renders an **interactive world map** where each country is shaded according to its confirmed case count. Key parameters:
- `locationmode = "country names"` — tells Plotly to match the string country names to its internal geographic boundaries.
- `color_continuous_scale = "mint"` — a light-to-dark green gradient that intuitively signals intensity.
- `range_color = [0, 10_000_000]` — caps the scale at 10 million so that moderately affected countries still show visible colour differences rather than all collapsing to the low end of the scale.

Hovering over any country on the rendered map shows its exact confirmed case count.

---

### Top 20 Countries by Cases

```python
top = latest_data.groupby("Country/Region")[['Confirmed', 'Recovered']].sum().reset_index()
top = top.sort_values(['Confirmed'], ascending=False)
top_20 = top.head(20)
top_20
```

Countries are grouped, summed, and then sorted in **descending order** by confirmed cases using `sort_values()`. `.head(20)` slices out the top 20 most-affected nations, making it easy to directly compare the hardest-hit countries side by side for both confirmed and recovered counts.

---

### Filtering India's Data

```python
india = data[data['Country/Region'] == 'India']
india
```

The **full time-series** (not just the latest date) is filtered for India alone. Using the complete history (rather than `latest_data`) is important here because it allows state-wise trends to be examined across time if needed, and ensures no information is lost before the India-specific analysis begins.

---

### Exploring Indian States

```python
india['Province/State'].nunique()
india['Province/State'].unique()
```

`nunique()` counts how many distinct state/UT entries appear in India's data, while `unique()` lists all of them. This step is essential for spotting inconsistencies — missing values (`NaN`), placeholder strings like `"Unknown"`, or differently-spelled state names — before aggregation.

---

### Handling Missing State Values

```python
india['Province/State'] = india['Province/State'].fillna(india['Province/State'].mode()[0])
india['Province/State'] = india['Province/State'].replace('Unknown', india['Province/State'].mode()[0])
```

Two imputation steps are applied in sequence:
- `fillna()` replaces `NaN` (completely absent) values with the **mode** — the most frequently occurring state name — as a neutral, data-driven fallback rather than dropping rows.
- `replace()` then swaps the literal string `"Unknown"` with the same mode value, because `"Unknown"` carries the same meaning as missing data but won't be caught by `fillna()` since it is technically a non-null value.

Using the mode (most frequent category) is preferred over a fixed label to avoid artificially skewing case counts toward any one state.

---

### India's Latest Snapshot

```python
india_latest_data = india[india['ObservationDate'] == india['ObservationDate'].max()]
india_latest_data
```

The same date-filter logic applied globally is now reapplied to India's subset. This ensures the subsequent state-level aggregation uses only the most recent cumulative totals — preventing historical rows from inflating the sums when grouping by state.

---

### State-wise Ranking

```python
top_state = india_latest_data.groupby('Province/State')[['Confirmed', 'Recovered']].sum().reset_index()
top_state = top_state.sort_values(['Confirmed'], ascending=False)
top_state.head(20)
```

States are grouped and their case counts summed, then sorted descending by confirmed cases. The top 20 states are displayed, giving a clear ranked view of which states bore the greatest burden of infections — analogous to the global top-20 analysis done earlier, but at the sub-national level for India.

---

### Most & Least Affected States

```python
# Maximum confirmed cases value
top_state['Confirmed'].max()

# Name of the most affected state
top_state[top_state['Confirmed'] == top_state['Confirmed'].max()]['Province/State'].values[0]

# Name of the least affected state
top_state[top_state['Confirmed'] == top_state['Confirmed'].min()]['Province/State'].values[0]
```

`.max()` and `.min()` retrieve the highest and lowest confirmed case totals across all states. The state name is extracted by filtering the DataFrame to the row where `Confirmed` equals that extreme value, then using `.values[0]` to pull the string from the resulting single-row result. This yields a direct, human-readable answer: the **most** and **least** COVID-affected Indian state by name.

---
##  Conclusion

This analysis demonstrates how Python-based data science tools can be used to derive actionable insights from large-scale epidemiological data:

- **Data Preprocessing** — Type casting, dropping redundant columns, and feature engineering (Active Cases) formed a clean analytical foundation.
- **Global Trends** — The choropleth map clearly highlighted that countries like the **US, India, and Mainland China** recorded the highest confirmed case counts at the latest observation date.
- **Top 20 Countries** — Grouping and sorting by confirmed cases revealed a significant disparity between highly affected and less-affected nations.
- **India's State Analysis** — After handling missing state values with mode-based imputation, the analysis successfully identified the **most and least affected Indian states/UTs**, providing granular visibility into the domestic spread.

---
