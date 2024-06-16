import pandas as pd

path2csv = r"C:\Users\gabma\Dropbox\PC\Downloads\trace_data_limit_10000_gas_co2e_100yr_subsectors_aluminum,cement,chemicals,other-manufacturing,petrochemicals,pulp-and-paper,steel_to_2022_since_2022.csv"

df = pd.read_csv(path2csv)
print(df.head())
print(df.columns)

# get the data frame correspinding to "lime_production" for source_type
df_lime_production = df[df['source_type'] == 'lime_production']

# Now get the sub dataframe containing only the source_name, lat and lon columns for the lime_production source_type
df_lime_production = df_lime_production[['source_name', 'lat', 'lon']]

# Get the unique values of the source_name column in the lime_production sub dataframe
unique_sources = df_lime_production['source_name'].unique()

# Get the new dataframe based only on the unique sources
df_lime_production_unique = df_lime_production.drop_duplicates(subset='source_name')
print(df_lime_production_unique)
print(len(df_lime_production_unique))

for i in range(len(df_lime_production_unique)):
    print(df_lime_production_unique.iloc[i]['source_name'], df_lime_production_unique.iloc[i]['lat'], df_lime_production_unique.iloc[i]['lon'])
