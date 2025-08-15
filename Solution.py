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


