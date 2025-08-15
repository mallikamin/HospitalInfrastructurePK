import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Load the data
df = pd.read_csv(r'C:\Users\Malik\Desktop\PKData.csv')

# Define Tier 1 cities
tier1_cities = ['Lahore', 'Karachi', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Peshawar']

# Create tier classification
df['Tier'] = np.where(df['CITY'].isin(tier1_cities), 'Tier 1', 
                     np.where(df['HOSPITAL_COUNT'] > 100, 'Tier 2',
                             np.where(df['HOSPITAL_COUNT'] > 20, 'Tier 3', 'Tier 4')))

# Check the data
print(df.head())
print(df.info())




# Filter Tier 1 cities
tier1_df = df[df['Tier'] == 'Tier 1']

# Basic statistics
print("\nBasic Statistics for Tier 1 Cities:")
print(tier1_df[['CITY', 'HOSPITAL_COUNT', 'POPULATION_2024_2025', 'AREA_SQ_KM', 
                'Hospitals per sqkm', 'Pop. Per sq km', 'Hospitals per person', 
                'Person served per hospital']].describe())

# City-specific stats
print("\nCity-specific Statistics:")
print(tier1_df[['CITY', 'HOSPITAL_COUNT', 'POPULATION_2024_2025', 'AREA_SQ_KM', 
                'Hospitals per sqkm', 'Pop. Per sq km', 'Hospitals per person', 
                'Person served per hospital']].sort_values('HOSPITAL_COUNT', ascending=False))




plt.figure(figsize=(6, 6))
sns.barplot(x='CITY', y='HOSPITAL_COUNT', data=tier1_df.sort_values('HOSPITAL_COUNT', ascending=False))
plt.title('Number of Hospitals in Tier 1 Cities')
plt.xlabel('City')
plt.ylabel('Number of Hospitals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




plt.figure(figsize=(12, 6))
plt.scatter(tier1_df['Pop. Per sq km'], tier1_df['Hospitals per sqkm'], s=tier1_df['HOSPITAL_COUNT']*2)
for i, row in tier1_df.iterrows():
    plt.text(row['Pop. Per sq km'], row['Hospitals per sqkm'], row['CITY'], fontsize=9)
plt.title('Population Density vs Hospital Density in Tier 1 Cities')
plt.xlabel('Population per sq km')
plt.ylabel('Hospitals per sq km')
plt.grid(True)
plt.tight_layout()
plt.show()




plt.figure(figsize=(6, 6))
sns.barplot(x='CITY', y='Person served per hospital', 
            data=tier1_df.sort_values('Person served per hospital', ascending=False))
plt.title('Persons Served per Hospital in Tier 1 Cities')
plt.xlabel('City')
plt.ylabel('Persons per Hospital')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




tier_stats = df.groupby('Tier').agg({
    'HOSPITAL_COUNT': 'sum',
    'POPULATION_2024_2025': 'sum',
    'AREA_SQ_KM': 'sum',
    'Hospitals per sqkm': 'mean',
    'Pop. Per sq km': 'mean',
    'Hospitals per person': 'mean',
    'Person served per hospital': 'mean'
}).reset_index()

print("\nTier-wise Statistics:")
print(tier_stats)
