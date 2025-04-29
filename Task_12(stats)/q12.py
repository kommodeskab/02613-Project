import pandas as pd
import matplotlib.pyplot as plt

column_names = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

df = pd.read_csv(r'Task 9-10/all_floorplans.csv', header=None, names=column_names)
print(df.head())

plt.figure(figsize=(8, 5))
plt.hist(df['mean_temp'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (ºC)')
plt.ylabel('Number of Buildings')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'Task_12(stats)/mean_temperature_histogram.png')
plt.close()

# average mean temperature
avg_mean_temp = df['mean_temp'].mean()
print(f"Average Mean Temperature: {avg_mean_temp:.2f} ºC")

# average temperature standard deviation
avg_std_temp = df['std_temp'].mean()
print(f"Average Temperature Standard Deviation: {avg_std_temp:.2f} ºC")

# 50% area above 18ºC
count_above_18 = (df['pct_above_18'] >= 50).sum()
print(f"Number of buildings with ≥50% area above 18ºC: {count_above_18}")

# 50% area below 15ºC
count_below_15 = (df['pct_below_15'] >= 50).sum()
print(f"Number of buildings with ≥50% area below 15ºC: {count_below_15}")