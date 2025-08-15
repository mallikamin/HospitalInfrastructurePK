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





# Hospital distribution by tier
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
sns.barplot(x='Tier', y='HOSPITAL_COUNT', data=tier_stats)
plt.title('Total Hospitals by Tier')

plt.subplot(2, 2, 2)
sns.barplot(x='Tier', y='Hospitals per sqkm', data=tier_stats)
plt.title('Average Hospital Density by Tier')

plt.subplot(2, 2, 3)
sns.barplot(x='Tier', y='Hospitals per person', data=tier_stats)
plt.title('Average Hospitals per Person by Tier')

plt.subplot(2, 2, 4)
sns.barplot(x='Tier', y='Person served per hospital', data=tier_stats)
plt.title('Average Persons Served per Hospital by Tier')

plt.tight_layout()
plt.show()




province_stats = df.groupby('PROVINCE').agg({
    'HOSPITAL_COUNT': 'sum',
    'POPULATION_2024_2025': 'sum',
    'AREA_SQ_KM': 'sum'
}).reset_index()

province_stats['Hospitals per million'] = (province_stats['HOSPITAL_COUNT'] / province_stats['POPULATION_2024_2025']) * 1000000
province_stats['Hospitals per 1000 sqkm'] = (province_stats['HOSPITAL_COUNT'] / province_stats['AREA_SQ_KM']) * 1000

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='PROVINCE', y='Hospitals per million', data=province_stats.sort_values('Hospitals per million', ascending=False))
plt.title('Hospitals per Million Population by Province')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='PROVINCE', y='Hospitals per 1000 sqkm', data=province_stats.sort_values('Hospitals per 1000 sqkm', ascending=False))
plt.title('Hospitals per 1000 sq km by Province')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()




corr_matrix = tier1_df[['HOSPITAL_COUNT', 'POPULATION_2024_2025', 'AREA_SQ_KM', 
                       'Hospitals per sqkm', 'Pop. Per sq km', 
                       'Hospitals per person', 'Person served per hospital']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix for Tier 1 Cities')
plt.tight_layout()
plt.show()




# Cities with highest and lowest hospital density
print("\nTop 5 Cities by Hospital Density:")
print(df.nlargest(5, 'Hospitals per sqkm')[['CITY', 'Hospitals per sqkm']])

print("\nBottom 5 Cities by Hospital Density:")
print(df.nsmallest(5, 'Hospitals per sqkm')[['CITY', 'Hospitals per sqkm']])

# Cities serving most and least people per hospital
print("\nTop 5 Cities by Persons Served per Hospital:")
print(df.nlargest(5, 'Person served per hospital')[['CITY', 'Person served per hospital']])

print("\nBottom 5 Cities by Persons Served per Hospital:")
print(df.nsmallest(5, 'Person served per hospital')[['CITY', 'Person served per hospital']])



plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Person served per hospital', hue='Tier', fill=True, common_norm=False, alpha=0.5)
plt.title('Distribution of Persons Served per Hospital by Tier')
plt.xlabel('Persons Served per Hospital')
plt.ylabel('Density')
plt.xlim(0, 30000)
plt.tight_layout()
plt.show()




plt.figure(figsize=(8, 5))
ax = sns.barplot(x='CITY', y='HOSPITAL_COUNT', data=df[df['CITY'].isin(['Faisalabad', 'Lahore'])], 
                palette=['#1f77b4', '#ff7f0e'], width=0.5)
plt.title('Hospital Count Comparison: Faisalabad vs Lahore', pad=15)
plt.xlabel('')
plt.ylabel('Number of Hospitals')

for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", 
               (p.get_x() + p.get_width()/2., p.get_height()),
               ha='center', va='center', 
               xytext=(0,7), textcoords='offset points')

plt.tight_layout()
plt.show()





plt.figure(figsize=(8, 5))
ax = sns.barplot(x='CITY', y='Person served per hospital', data=df[df['CITY'].isin(['Faisalabad', 'Lahore'])], 
                palette=['#1f77b4', '#ff7f0e'], width=0.5)
plt.title('Healthcare Accessibility: Persons Served per Hospital', pad=15)
plt.xlabel('')
plt.ylabel('Persons per Hospital')

for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", 
               (p.get_x() + p.get_width()/2., p.get_height()),
               ha='center', va='center', 
               xytext=(0,7), textcoords='offset points')

plt.tight_layout()
plt.show()
