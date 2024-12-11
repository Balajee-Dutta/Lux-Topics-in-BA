#Cleaning and merging the Data files

import pandas as pd

# Load and read the datasets simultaneously
anti_tobacco_campaign_df = pd.read_excel(r'C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\lux-Anti-tobacco mass media campaign.xlsx')
ban_tobacco_advertising_df = pd.read_excel(r'C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\lux-Enforce bans on tobacco advertising.xlsx')
health_warnings_df = pd.read_excel(r'C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\lux-Health warnings on cigarette packets.xlsx')
quit_tobacco_help_df = pd.read_csv(r'C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\lux-Offer help to quit tobacco use.csv')
smoke_free_df = pd.read_excel(r'C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\lux-smoke free data.xlsx')

# Filter the Luxembourg data from each dataset
anti_tobacco_campaign_lux = anti_tobacco_campaign_df[anti_tobacco_campaign_df['Year'].notnull()]
ban_tobacco_advertising_lux = ban_tobacco_advertising_df[ban_tobacco_advertising_df['Countries, territories and areas'] == 'Luxembourg']
health_warnings_lux = health_warnings_df[health_warnings_df['Countries, territories and areas'] == 'Luxembourg']
quit_tobacco_help_lux = quit_tobacco_help_df[quit_tobacco_help_df['Countries, territories and areas'] == 'Luxembourg']
smoke_free_lux = smoke_free_df[smoke_free_df['Countries, territories and areas'] == 'Luxembourg']

# Merge all datasets on the "Year" column
merged_df = pd.merge(anti_tobacco_campaign_lux, ban_tobacco_advertising_lux, on='Year', suffixes=('_anti', '_ban'))
merged_df = pd.merge(merged_df, health_warnings_lux, on='Year', suffixes=('', '_health'))
merged_df = pd.merge(merged_df, quit_tobacco_help_lux, on='Year', suffixes=('', '_quit'))
merged_df = pd.merge(merged_df, smoke_free_lux, on='Year', suffixes=('', '_smoke'))

# Save the cleaned dataframe as CSV
#merged_df.to_excel(r'C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\merged_tobacco_control_data.xlsx', index=False)

# Display the cleaned dataframe
print(merged_df.head())

#Creation of Correlation-Heatmap

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'lux- co-relation file.xlsx'
df = pd.read_excel(file_path)

# Rename columns for easier access
df.rename(columns={
    'Affordability of cigarettes: percentage of GDP per capita required to purchase 2000 cigarettes of the most sold brand': 'Affordability',
    'Most sold brand of cigarettes - Taxes as a % of price: specific excise': 'Taxes_Percentage'
}, inplace=True)

# Filter only the relevant columns
correlation_df = df[['Affordability', 'Taxes_Percentage']]

# Drop any rows with missing values
correlation_df.dropna(inplace=True)

# Calculate the correlation matrix
heatmap_df = correlation_df.corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Heatmap of Correlation between Affordability and Taxes as % of Price')
plt.show()


#creation of Line chart for exploratory analysis

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'lux- co-relation file.xlsx'
df = pd.read_excel(file_path)

# Filter the data for Luxembourg
lux_affordability_data = df[df['Countries, territories and areas'] == 'Luxembourg']

# Rename columns for easier access
lux_affordability_data.rename(columns={
    'Affordability of cigarettes: percentage of GDP per capita required to purchase 2000 cigarettes of the most sold brand': 'Affordability',
    'Most sold brand of cigarettes - Taxes as a % of price: specific excise': 'Taxes_Percentage'
}, inplace=True)

# Extract relevant columns for the graph
years = lux_affordability_data['Year']
affordability = lux_affordability_data['Affordability']
tax_percentage = lux_affordability_data['Taxes_Percentage']

# Create a line chart for affordability and taxes as a percentage of price with data labels
plt.figure(figsize=(10, 6))

# Plot the affordability line
plt.plot(years, affordability, marker='o', color='green', label='Affordability', linestyle='-', linewidth=2)
for i, txt in enumerate(affordability):
    plt.text(years.iloc[i], affordability.iloc[i] + 0.01, f'{txt:.2f}', color='green', fontsize=10, ha='center')

# Plot the taxes as a percentage of price line
plt.plot(years, tax_percentage, marker='s', color='red', label='Taxes as % of Price', linestyle='-', linewidth=2)
for i, txt in enumerate(tax_percentage):
    plt.text(years.iloc[i], tax_percentage.iloc[i] + 0.01, f'{txt:.2f}', color='red', fontsize=10, ha='center')

# Set labels, title, and legend
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Value', fontweight='bold')
plt.title('Affordability and Taxes as % of Price (2012-2020)')
plt.xticks(years, fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend()

# Remove gridlines
plt.grid(False)

# Display the plot
plt.show()

#Line chart for Combined deaths Data in Luxembourgh

import pandas as pd
import matplotlib.pyplot as plt

# List of colors as per the provided color categories
custom_colors = [
    # Reds and Pinks
    '#DC143C', '#FF66CC', '#FF9999', '#990000', '#F4C2C2',
    # Oranges and Browns
    '#CC5500', '#FFB347', '#C68E17', '#DA7445', '#954535',
    # Yellows and Golds
    '#FFD300', '#DAA520', '#FFDB58', '#FFF44F', '#FFBF00',
    # Greens
    '#50C878', '#98FF98', '#808000', '#9FE2BF', '#228B22',
    # Blues
    '#87CEEB', '#4169E1', '#000080', '#00FFFF', '#0F52BA',
    # Purples and Violets
    '#E6E6FA', '#DA70D6', '#9966CC', '#FF00FF', '#8F00FF',
    # Grays and Silvers
    '#36454F', '#BCC6CC', '#B2BEB5', '#E5E4E2', '#71797E',
    # Whites and Creams
    '#FFFFF0', '#F0EAD6', '#F8F6F1', '#FAFAFA', '#F3E5AB',
    # Blacks and Deep Shades
    '#343434', '#353839', '#0F0F0F', '#555D50', '#0A0A0A',
    # Uncommon and Rare Shades
    '#CCCCFF', '#008080', '#40E0D0', '#800000', '#7FFF00'
]

# File path to the Excel file
file_path = r"C:\Users\Jeedu\OneDrive\Desktop\Lux-Topics in BA\Combined_Deaths_Data- lux.xlsx"

# Load the data from the specified file path
data = pd.read_excel(file_path, sheet_name='Deaths Data')

# Pivot the data to have 'Year' as the index and 'Death Type' as columns
pivot_df = data.pivot(index='Year', columns='Death Type', values='Deaths')

# Convert the entire pivot table to numeric, forcing errors to NaN
pivot_df_numeric = pivot_df.apply(pd.to_numeric, errors='coerce')

# Extract columns and years for plotting
columns_to_plot = pivot_df_numeric.columns
years = pivot_df_numeric.index

# Create the plot using the specified color palette
plt.figure(figsize=(12, 6))

# Plot all the lines, and highlight 'Smoking' and 'Secondhand smoke'
for i, column in enumerate(columns_to_plot):
    color = custom_colors[i % len(custom_colors)]  # Cycle through the custom color list
    if column == 'Smoking':
        plt.plot(years, pivot_df_numeric[column], label=column, linewidth=3, marker='o', color=color)
        for x, y in zip(years, pivot_df_numeric[column]):
            plt.text(x, y + 10, f'{y}', fontsize=9, ha='center', va='bottom')  # Data labels above the line
    elif column == 'Secondhand smoke':
        plt.plot(years, pivot_df_numeric[column], label=column, linewidth=2.5, linestyle='--', color=color)
    else:
        plt.plot(years, pivot_df_numeric[column], label=column, linewidth=1, color=color)

# Set chart details
plt.title('Deaths by Cause Over Time', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Number of Deaths in millions', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Position the legend on the extreme right
plt.xticks(years)

# Set y-axis intervals and ensure the origin starts at 0
plt.yticks(range(0, int(pivot_df_numeric.max().max()) + 100, 100))
plt.ylim(0, int(pivot_df_numeric.max().max()) + 100)  # Set the y-axis to start from 0

# Enable the grid lines
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()


#visualisation of Polynomial Regression (Degree 4) for predictive analysis
import matplotlib.pyplot as plt
import numpy as np

# Define the future years
future_years = [2022, 2024, 2026, 2028, 2030]

# Placeholder values for each model
# Polynomial Regression
poly_aff_future_seasonal = [0.5214, 0.5064, 0.4825, 0.4480, 0.4013]
poly_tax_future_seasonal = [0.0757, 0.0835, 0.0942, 0.1078, 0.1242]

# STL with Linear Regression
stl_aff_future_dynamic = [0.5375, 0.5350, 0.5325, 0.5300, 0.5375]
stl_tax_future_dynamic = [0.0550, 0.0600, 0.0650, 0.0700, 0.0550]

# Exponential Smoothing
exp_smoothing_aff_future = [0.5317, 0.5300, 0.5317, 0.5300, 0.5317]
exp_smoothing_tax_future = [0.0689, 0.0705, 0.0689, 0.0705, 0.0689]

# Visualization for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.plot(future_years, poly_aff_future_seasonal, label='Polynomial Regression (Degree 4) - Affordability', linestyle='-', marker='o', color='purple')
plt.plot(future_years, poly_tax_future_seasonal, label='Polynomial Regression (Degree 4) - Tax Percentage', linestyle='-', marker='o', color='brown')

# Add data labels above the line
for i, txt in enumerate(poly_aff_future_seasonal):
    plt.text(future_years[i], poly_aff_future_seasonal[i] + 0.008, f'{txt:.4f}', fontweight='bold', fontsize=7, ha='center')

for i, txt in enumerate(poly_tax_future_seasonal):
    plt.text(future_years[i], poly_tax_future_seasonal[i] + 0.008, f'{txt:.4f}', fontweight='bold', fontsize=7, ha='center')

# Customize the plot
plt.xlabel('Years', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.title('Polynomial Regression (Degree 4)', fontweight='bold', pad=20)
plt.xticks(future_years, rotation=45, fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend on the extreme right
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Visualization for STL with Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(future_years, stl_aff_future_dynamic, label='STL with Linear Regression - Affordability', linestyle='-', marker='o', color='green')
plt.plot(future_years, stl_tax_future_dynamic, label='STL with Linear Regression - Tax Percentage', linestyle='-', marker='o', color='orange')

# Add data labels above the line
for i, txt in enumerate(stl_aff_future_dynamic):
    plt.text(future_years[i], stl_aff_future_dynamic[i] + 0.007, f'{txt:.4f}', fontweight='bold', fontsize=7, ha='center')

for i, txt in enumerate(stl_tax_future_dynamic):
    plt.text(future_years[i], stl_tax_future_dynamic[i] + 0.007, f'{txt:.4f}', fontweight='bold', fontsize=7, ha='center')

# Customize the plot
plt.xlabel('Years', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.title('STL with Linear Regression (Dynamic Seasonality)', fontweight='bold', pad=20)
plt.xticks(future_years, rotation=45, fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend on the extreme right
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Visualization for Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(future_years, exp_smoothing_aff_future, label='Exponential Smoothing - Affordability', linestyle='-', marker='o', color='blue')
plt.plot(future_years, exp_smoothing_tax_future, label='Exponential Smoothing - Tax Percentage', linestyle='-', marker='o', color='red')

# Add data labels above the line
for i, txt in enumerate(exp_smoothing_aff_future):
    plt.text(future_years[i], exp_smoothing_aff_future[i] + 0.007, f'{txt:.4f}', fontweight='bold', fontsize=7, ha='center')

for i, txt in enumerate(exp_smoothing_tax_future):
    plt.text(future_years[i], exp_smoothing_tax_future[i] + 0.007, f'{txt:.4f}', fontweight='bold', fontsize=7, ha='center')

# Customize the plot
plt.xlabel('Years', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.title('Exponential Smoothing with Seasonality', fontweight='bold', pad=20)
plt.xticks(future_years, rotation=45, fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Legend on the extreme right
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#Visualisation of WHO Tobacco Control Measures
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file from the specified path
file_path = 'WHO_Tobacco_Control_Measures.csv'  # Use raw string to handle backslashes
df = pd.read_csv(file_path)

# Step 2: Extract categories and effectiveness scores from the DataFrame
categories = df['Control Measure'].tolist()
effectiveness = df['Effectiveness (%)'].tolist()

# Step 3: Create distinct colors for each bar
colors = ['#FF5733', '#33FF57', '#3357FF', '#F39C12', '#8E44AD', '#E74C3C']  # Add more colors if needed

# Step 4: Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, effectiveness, color=colors[:len(categories)])  # Adjust colors to match the number of categories

# Step 5: Add data labels inside each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval - 5, f'{yval}%', ha='center', va='top', color='white', fontweight='bold')

# Step 6: Customize the chart
plt.title('Effectiveness of WHO-Recommended Tobacco Control Measures', fontweight='bold', pad=10)
plt.xlabel('Control Measures', fontweight='bold', labelpad=-8)  # Move the X-axis label inside the graph
plt.ylabel('Effectiveness (%)', fontweight='bold', labelpad=-45)  # Move the Y-axis label inside the graph
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.yticks(fontweight='bold')

# Remove grid lines
plt.grid(False)

# Adjust layout to bring the X-axis and Y-axis labels inside the chart
plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)

# Display the chart
plt.show()


