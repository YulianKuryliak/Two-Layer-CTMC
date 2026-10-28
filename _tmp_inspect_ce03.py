import pandas as pd

print("=== arrivals.csv ===")
df_arr = pd.read_csv('output/claude_experiments/ce03_beta_ratio/arrivals.csv')
print(df_arr.head(20))
print(f"\nShape: {df_arr.shape}")
print(f"Columns: {list(df_arr.columns)}")
print(f"\nUnique scenarios: {df_arr['scenario'].unique() if 'scenario' in df_arr.columns else 'N/A'}")

print("\n\n=== summary.csv ===")
df_sum = pd.read_csv('output/claude_experiments/ce03_beta_ratio/summary.csv')
print(df_sum)
print(f"\nColumns: {list(df_sum.columns)}")
