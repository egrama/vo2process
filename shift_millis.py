import pandas as pd

# Read the CSV file
df = pd.read_csv('esala2_p2.csv')
print(df.columns)
print(df.head())

# Subtract 26454 from column 'R' only for numeric values
df.loc[df['R'].notna() & df['R'].apply(lambda x: str(x).replace('.','',1).isdigit()), 'R'] -= 26454

# Save the modified dataframe back to CSV
df.to_csv('esala2_p2_shifted.csv', index=False)

print("Processing complete. Modified file saved.") 