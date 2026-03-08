import pandas as pd
df = pd.read_csv('notebooks/Reviews.csv', nrows=5)
print(df.columns)
print(df.head(2))
