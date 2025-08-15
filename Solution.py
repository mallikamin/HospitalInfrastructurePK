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
