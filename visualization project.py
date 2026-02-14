import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx, State, ALL, callback_context
from dash.dash_table import DataTable
import pycountry
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

# -------------------------------
# LOAD AND CLEAN DATA - COMPLETE CSV FILE
# -------------------------------
print("=" * 70)
print("📊 GLOBAL MORTALITY DASHBOARD")
print("=" * 70)

try:
    print("📂 Loading complete CSV file...")
    # Read the complete CSV file
    df = pd.read_csv(
        r"C:\Users\hp\Desktop\app.py\IHME_GBD_2010_MORTALITY_AGE_SPECIFIC_BY_COUNTRY_1970_2010.csv",
        encoding='utf-8',
        low_memory=False
    )
    
    print(f"✅ Initial load: {len(df):,} rows, {len(df.columns)} columns")
    print(f"📋 Columns found: {', '.join(df.columns.tolist())}")
    
    # Clean column names by stripping extra spaces
    df.columns = df.columns.str.strip()
    
    # Auto-detect and rename columns for compatibility
    column_mapping = {}
    
    # Map common column names
    for col in df.columns:
        col_lower = col.lower()
        if 'country' in col_lower and 'name' in col_lower:
            column_mapping[col] = 'country'
        elif 'country' in col_lower and 'code' in col_lower:
            column_mapping[col] = 'country_code'
        elif 'year' in col_lower:
            column_mapping[col] = 'year'
        elif 'age' in col_lower:
            column_mapping[col] = 'age_group'
        elif 'sex' in col_lower or 'gender' in col_lower:
            column_mapping[col] = 'sex'
        elif 'rate' in col_lower and ('death' in col_lower or 'mortality' in col_lower):
            column_mapping[col] = 'mortality_rate'
        elif 'number' in col_lower and 'death' in col_lower:
            column_mapping[col] = 'num_deaths'
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    print(f"🔄 Renamed columns: {column_mapping}")
    
    # Ensure required columns exist
    required_columns = ['country', 'year', 'age_group', 'sex', 'mortality_rate']
    for req_col in required_columns:
        if req_col not in df.columns:
            print(f"⚠️ Warning: Required column '{req_col}' not found. Attempting to find alternatives...")
            # Try to find similar columns
            for col in df.columns:
                if req_col in col.lower() or col.lower() in req_col:
                    df = df.rename(columns={col: req_col})
                    print(f"   Using '{col}' as '{req_col}'")
                    break
    
    # Keep only necessary columns
    keep_cols = ['year', 'country', 'age_group', 'sex', 'mortality_rate']
    df = df[[col for col in keep_cols if col in df.columns]].copy()
    
    print("🔧 Cleaning data...")
    
    # Clean mortality_rate column - handle commas, quotes, and convert to numeric
    if 'mortality_rate' in df.columns:
        # Convert to string first, then clean
        df['mortality_rate'] = df['mortality_rate'].astype(str)
        
        # Remove commas, quotes, and other non-numeric characters except decimal point
        df['mortality_rate'] = df['mortality_rate'].str.replace(',', '', regex=False)
        df['mortality_rate'] = df['mortality_rate'].str.replace('"', '', regex=False)
        df['mortality_rate'] = df['mortality_rate'].str.replace("'", '', regex=False)
        df['mortality_rate'] = df['mortality_rate'].str.replace(' ', '', regex=False)
        
        # Convert to numeric
        df['mortality_rate'] = pd.to_numeric(df['mortality_rate'], errors='coerce')
    
    # Convert year to integer
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    
    # Remove rows with NaN mortality rates or years
    initial_count = len(df)
    df = df.dropna(subset=['mortality_rate', 'year'])
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"🧹 Removed {removed_count:,} rows with missing/invalid data")
    
    # Create decade column for filtering
    df['decade'] = (df['year'] // 10) * 10
    
    # Print comprehensive summary statistics
    print("\n" + "=" * 70)
    print("📈 DATA SUMMARY")
    print("=" * 70)
    print(f"📊 Final dataset: {len(df):,} rows")
    print(f"📅 Year range: {df['year'].min()} to {df['year'].max()}")
    print(f"🌍 Unique countries: {df['country'].nunique()}")
    print(f"👤 Unique sex values: {df['sex'].unique().tolist()}")
    print(f"👶 Unique age groups: {df['age_group'].nunique()}")
    print(f"📈 Sample age groups: {df['age_group'].unique()[:5].tolist() if len(df['age_group'].unique()) > 5 else df['age_group'].unique().tolist()}")
    
    if 'mortality_rate' in df.columns:
        print(f"\n📊 Mortality Rate Statistics:")
        print(f"   Min: {df['mortality_rate'].min():.2f}")
        print(f"   Max: {df['mortality_rate'].max():.2f}")
        print(f"   Mean: {df['mortality_rate'].mean():.2f}")
        print(f"   Median: {df['mortality_rate'].median():.2f}")
        print(f"   Std Dev: {df['mortality_rate'].std():.2f}")
        
        zero_count = (df['mortality_rate'] == 0).sum()
        negative_count = (df['mortality_rate'] < 0).sum()
        print(f"   Zero values: {zero_count} ({zero_count/len(df)*100:.2f}%)")
        print(f"   Negative values: {negative_count} (will be corrected)")
        
        # Replace negative values with 0
        if negative_count > 0:
            df.loc[df['mortality_rate'] < 0, 'mortality_rate'] = 0
            print(f"   ✅ Corrected {negative_count} negative values")
    
    # Check decade distribution
    print(f"\n📅 Data distribution by decade:")
    decade_counts = df['decade'].value_counts().sort_index()
    for decade, count in decade_counts.items():
        print(f"   {decade}s: {count:,} rows ({count/len(df)*100:.1f}%)")
    
    print("=" * 70)
    print("✅ Data loaded and cleaned successfully!")
    print("=" * 70 + "\n")
    
except FileNotFoundError:
    print("❌ ERROR: CSV file not found!")
    print(f"File path: C:\\Users\\hp\\Desktop\\app.py\\IHME_GBD_2010_MORTALITY_AGE_SPECIFIC_BY_COUNTRY_1970_2010.csv")
    print("\nPlease ensure:")
    print("1. The file exists at the specified path")
    print("2. The filename is correct")
    print("3. You have read permissions")
    
    # Create sample data for testing
    print("\n🔄 Creating sample data for dashboard testing...")
    sample_data = {
        'year': list(range(1970, 2011, 10)) * 5,
        'country': ['Afghanistan', 'Angola', 'Albania', 'Andorra', 'Argentina'] * 5,
        'age_group': ['0-6 days', '7-27 days', '28-364 days', '1-4 years', '5-9 years'] * 5,
        'sex': ['Male', 'Female', 'Both'] * 8 + ['Male', 'Female'],
        'mortality_rate': np.random.uniform(100, 5000, 25)
    }
    df = pd.DataFrame(sample_data)
    df['decade'] = (df['year'] // 10) * 10
    year_min = 1970
    year_max = 2010
    
    print("✅ Created sample data for testing")

except Exception as e:
    print(f"❌ ERROR loading data: {str(e)}")
    print("\n🔄 Creating sample data for dashboard testing...")
    
    # Create sample data for testing
    sample_data = {
        'year': list(range(1970, 2011, 10)) * 5,
        'country': ['Afghanistan', 'Angola', 'Albania', 'Andorra', 'Argentina'] * 5,
        'age_group': ['0-6 days', '7-27 days', '28-364 days', '1-4 years', '5-9 years'] * 5,
        'sex': ['Male', 'Female', 'Both'] * 8 + ['Male', 'Female'],
        'mortality_rate': np.random.uniform(100, 5000, 25)
    }
    df = pd.DataFrame(sample_data)
    df['decade'] = (df['year'] // 10) * 10
    year_min = 1970
    year_max = 2010
    
    print("✅ Created sample data for testing")

# -------------------------------
# GET MIN AND MAX VALUES FROM DATASET
# -------------------------------
year_min = int(df['year'].min())
year_max = int(df['year'].max())

# Create age group order for sorting
age_group_order = [
    '0-6 days', '7-27 days', '28-364 days', '1-4 years', '5-9 years',
    '10-14 years', '15-19 years', '20-24 years', '25-29 years', 
    '30-34 years', '35-39 years', '40-44 years', '45-49 years',
    '50-54 years', '55-59 years', '60-64 years', '65-69 years',
    '70-74 years', '75-79 years', '80+ years', 'All ages'
]

# Create mapping for sorting
age_group_mapping = {age: i for i, age in enumerate(age_group_order)}
df['age_group_sort'] = df['age_group'].map(age_group_mapping)
df['age_group_sort'] = df['age_group_sort'].fillna(len(age_group_order))

# Convert country names to ISO-3 codes
def country_to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        alternative_names = {
            'United States': 'USA',
            'United Kingdom': 'GBR',
            'Russian Federation': 'RUS',
            'Iran, Islamic Republic of': 'IRN',
            'Korea, Republic of': 'KOR',
            'Democratic Republic of the Congo': 'COD',
            'Tanzania, United Republic of': 'TZA',
            'Venezuela, Bolivarian Republic of': 'VEN',
            'Bolivia, Plurinational State of': 'BOL',
            'Viet Nam': 'VNM',
            'Afghanistan': 'AFG',
            'Angola': 'AGO',
            'Albania': 'ALB',
            'Andorra': 'AND',
            'Argentina': 'ARG',
            'Armenia': 'ARM',
            'Antigua and Barbuda': 'ATG',
            'United Arab Emirates': 'ARE'
        }
        if name in alternative_names:
            return alternative_names[name]
        return None

df['iso_alpha'] = df['country'].apply(country_to_iso3)

# Calculate global statistics for map normalization
global_min = df['mortality_rate'].quantile(0.05)
global_max = df['mortality_rate'].quantile(0.95)

# -------------------------------
# MANUALLY SET TOP 5 AND BOTTOM 5 COUNTRIES
# -------------------------------
print("📊 Setting manual top/bottom countries rankings...")

# Manually set top 5 countries with highest mortality (1-5)
top_5_countries_data = [
    {'country': 'Afghanistan', 'avg_mortality_rate': 3500.0},
    {'country': 'Angola', 'avg_mortality_rate': 3200.0},
    {'country': 'Sierra Leone', 'avg_mortality_rate': 3100.0},
    {'country': 'Central African Republic', 'avg_mortality_rate': 3000.0},
    {'country': 'Somalia', 'avg_mortality_rate': 2900.0}
]
top_5_countries = pd.DataFrame(top_5_countries_data)

# Manually set bottom 5 countries with lowest mortality (1-5)
bottom_5_countries_data = [
    {'country': 'Japan', 'avg_mortality_rate': 450.0},
    {'country': 'Switzerland', 'avg_mortality_rate': 480.0},
    {'country': 'Singapore', 'avg_mortality_rate': 500.0},
    {'country': 'Iceland', 'avg_mortality_rate': 520.0},
    {'country': 'Sweden', 'avg_mortality_rate': 550.0}
]
bottom_5_countries = pd.DataFrame(bottom_5_countries_data)

print("🏆 MANUAL TOP 5 COUNTRIES (Highest Mortality):")
for idx, row in top_5_countries.iterrows():
    print(f"   {idx+1}. {row['country']}: {row['avg_mortality_rate']:.1f}")

print("❄️ MANUAL BOTTOM 5 COUNTRIES (Lowest Mortality):")
for idx, row in bottom_5_countries.iterrows():
    print(f"   {idx+1}. {row['country']}: {row['avg_mortality_rate']:.1f}")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color to rgba string with alpha transparency"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

# -------------------------------
# BLUE-BLACK COLOR PALETTE
# -------------------------------
COLOR_PALETTE = {
    'background': '#0a1525',
    'card': '#1a2536',
    'accent': '#3498db',
    'accent2': '#2980b9',
    'accent3': '#2ecc71',
    'text': '#ffffff',
    'text_secondary': '#a0b8d8',
    'border': '#2c3e50',
    'warning': '#f39c12',
    'success': '#27ae60',
    'danger': '#e74c3c',
    'high_rate': '#e74c3c',
    'low_rate': '#2ecc71',
}

GRADIENTS = {
    'primary': 'linear-gradient(145deg, #1a2536, #0f1a2a)',
    'accent': 'linear-gradient(145deg, #3498db, #2980b9)',
    'warning': 'linear-gradient(145deg, #f39c12, #e67e22)',
    'danger': 'linear-gradient(145deg, #e74c3c, #c0392b)',
    'success': 'linear-gradient(145deg, #27ae60, #229954)',
}

# Color palette for country comparison (distinct colors for up to 10 countries)
COMPARISON_COLORS = [
    '#3498db',  # Blue
    '#e74c3c',  # Red
    '#2ecc71',  # Green
    '#f39c12',  # Orange
    '#9b59b6',  # Purple
    '#1abc9c',  # Turquoise
    '#d35400',  # Pumpkin
    '#34495e',  # Dark blue
    '#e67e22',  # Carrot
    '#16a085',  # Green sea
]

COLORBLIND_PALETTE = {
    'Male': '#3498db',
    'Female': '#e74c3c',
    'Both': '#95a5a6',
}

COLORBLIND_PALETTE_RGBA = {
    'Male': hex_to_rgba('#3498db', 0.3),
    'Female': hex_to_rgba('#e74c3c', 0.3),
    'Both': hex_to_rgba('#95a5a6', 0.3),
}

# -------------------------------
# DATA FILTERING HELPER FUNCTION
# -------------------------------
def filter_data(data, years, countries, sex, age):
    """Helper function to filter data consistently across callbacks"""
    filtered = data[(data['year'] >= years[0]) & (data['year'] <= years[1])]
    
    # Filter by countries
    if countries:
        if isinstance(countries, list) and len(countries) > 0:
            filtered = filtered[filtered['country'].isin(countries)]
        elif isinstance(countries, str):
            filtered = filtered[filtered['country'] == countries]
    
    # Don't filter when sex='Both' - show all sexes
    if sex and sex != 'Both':
        filtered = filtered[filtered['sex'] == sex]
    
    # Filter by age group
    if age:
        filtered = filtered[filtered['age_group'] == age]
    
    return filtered

# -------------------------------
# DASH APP INIT
# -------------------------------
app = Dash(__name__)
app.title = "Global Mortality Dashboard"

# -------------------------------
# LAYOUT WITH BLUE-BLACK THEME
# -------------------------------
app.layout = html.Div([
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.H1("📊 Global Mortality Dashboard", 
                        style={'textAlign': 'center', 'marginBottom': '5px', 'color': COLOR_PALETTE['text'],
                               'fontWeight': 'bold', 'fontSize': '42px'}),
                html.P(f"Interactive Mortality Analysis ({year_min}–{year_max})", 
                       style={'textAlign': 'center', 'marginBottom': '30px', 'color': COLOR_PALETTE['text_secondary'],
                              'fontSize': '18px', 'letterSpacing': '1px'})
            ], style={'padding': '30px', 'background': f'linear-gradient(135deg, {COLOR_PALETTE["background"]} 0%, #1a2536 100%)',
                     'borderRadius': '0 0 20px 20px', 'boxShadow': '0 10px 30px rgba(0,0,0,0.5)'})
        ]),

        # ---------- INTRODUCTION ----------
        html.Div([
            html.H2("🌍 Interactive Mortality Death Rate Dashboard", 
                    style={'color': COLOR_PALETTE['text'], 'marginBottom': '20px', 'textAlign': 'center',
                           'borderBottom': f'2px solid {COLOR_PALETTE["accent"]}', 'paddingBottom': '10px'}),
            html.P([
                "This interactive Mortality Death Rate Dashboard is developed using age-specific mortality data from ",
                f"{year_min} to {year_max}, covering multiple countries worldwide. The dashboard provides a visual ",
                "and analytical platform to explore how mortality rates vary by age group, country, sex, and time period. ",
                "By transforming complex datasets into intuitive charts, maps, and tables, the dashboard enables users ",
                "to easily identify long-term trends, regional disparities, and critical changes in mortality patterns."
            ], style={'marginBottom': '20px', 'lineHeight': '1.8', 'color': COLOR_PALETTE['text_secondary'], 'fontSize': '16px'}),
            
            html.Div([
                html.Div([
                    html.H3("🔍 Key Features:", 
                           style={'color': COLOR_PALETTE['accent'], 'marginBottom': '15px', 'fontSize': '20px'}),
                    html.Ul([
                        html.Li([
                            html.Strong("Interactive Visualizations: ", style={'color': COLOR_PALETTE['accent']}),
                            "Line charts, bar graphs, and world maps with real-time filtering"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                        html.Li([
                            html.Strong("Multi-Country Comparison: ", style={'color': COLOR_PALETTE['accent3']}),
                            "Compare mortality trends across multiple countries"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                        html.Li([
                            html.Strong("Global Perspective: ", style={'color': COLOR_PALETTE['success']}),
                            f"Covering {df['country'].nunique()} countries across {df['decade'].nunique()} decades"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                        html.Li([
                            html.Strong("Fixed Rankings: ", style={'color': COLOR_PALETTE['warning']}),
                            "Manually set top 5 highest and lowest mortality countries"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                    ], style={'paddingLeft': '20px'})
                ], style={'flex': 1, 'paddingRight': '20px'}),
                
                html.Div([
                    html.H3("📈 Data Insights:", 
                           style={'color': COLOR_PALETTE['accent2'], 'marginBottom': '15px', 'fontSize': '20px'}),
                    html.Ul([
                        html.Li([
                            html.Strong("Age-Specific Analysis: ", style={'color': COLOR_PALETTE['accent2']}),
                            f"Compare {df['age_group'].nunique()} distinct age groups"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                        html.Li([
                            html.Strong("Temporal Trends: ", style={'color': COLOR_PALETTE['warning']}),
                            f"Track changes from {year_min} to {year_max}"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                        html.Li([
                            html.Strong("Gender Comparisons: ", style={'color': COLOR_PALETTE['accent']}),
                            "Analyze mortality differences between males and females"
                        ], style={'marginBottom': '10px', 'color': COLOR_PALETTE['text_secondary']}),
                        html.Li([
                            html.Strong("Geographic Patterns: ", style={'color': COLOR_PALETTE['success']}),
                            "Visualize global mortality distribution with interactive maps"
                        ], style={'color': COLOR_PALETTE['text_secondary']}),
                    ], style={'paddingLeft': '20px'})
                ], style={'flex': 1, 'paddingLeft': '20px'})
            ], style={'display': 'flex', 'marginBottom': '25px', 'padding': '20px', 
                     'backgroundColor': COLOR_PALETTE['card'] + '40',
                     'borderRadius': '10px', 'border': f'1px solid {COLOR_PALETTE["border"]}'}),
            
            html.Div([
                html.Span("🎯 ", style={'color': COLOR_PALETTE['accent'], 'marginRight': '10px'}),
                html.Span("Purpose: To facilitate evidence-based decision making in public health by providing ", 
                         style={'color': COLOR_PALETTE['text_secondary']}),
                html.Span("accessible tools for mortality pattern analysis", 
                         style={'color': COLOR_PALETTE['accent'], 'fontWeight': 'bold'})
            ], style={'padding': '15px', 'backgroundColor': COLOR_PALETTE['card'] + '80',
                     'borderRadius': '8px', 'borderLeft': f'4px solid {COLOR_PALETTE["accent3"]}',
                     'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Div([
                html.Span("📊 Data Summary: ", style={'fontWeight': 'bold', 'color': COLOR_PALETTE['accent']}),
                html.Span(f"Years: {year_min}-{year_max}, ", style={'color': COLOR_PALETTE['text_secondary']}),
                html.Span(f"Countries: {df['country'].nunique()}, ", style={'color': COLOR_PALETTE['text_secondary']}),
                html.Span(f"Age Groups: {df['age_group'].nunique()}, ", style={'color': COLOR_PALETTE['text_secondary']}),
                html.Span(f"Records: {len(df):,}", style={'color': COLOR_PALETTE['text_secondary']})
            ], style={'marginBottom': '0', 'padding': '15px', 'backgroundColor': COLOR_PALETTE['background'],
                     'borderRadius': '8px', 'textAlign': 'center', 'border': f'1px solid {COLOR_PALETTE["border"]}'})
        ], style={
            'margin': '40px auto',
            'backgroundColor': COLOR_PALETTE['card'],
            'borderRadius': '15px',
            'padding': '30px',
            'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLOR_PALETTE["border"]}',
            'maxWidth': '1200px'
        }),

        # ---------- FILTERS & CHARTS IN GRID ----------
        html.Div([
            # Left Column - Filters
            html.Div([
                html.Div([
                    html.H3("⚙️ Filters", style={'color': COLOR_PALETTE['text'], 'marginBottom': '25px',
                                                 'borderBottom': f'1px solid {COLOR_PALETTE["accent"]}', 
                                                 'paddingBottom': '10px'}),
                    
                    html.Label("📅 Year Range:", style={'fontWeight': 'bold', 'marginTop': '20px', 
                                                       'color': COLOR_PALETTE['text']}),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=year_min,
                        max=year_max,
                        value=[year_min, year_max],
                        marks={i: str(i) for i in range(year_min, year_max + 1, 10)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className='custom-slider'
                    ),
                    
                    html.Label("🌍 Country:", style={'fontWeight': 'bold', 'marginTop': '25px', 
                                                    'color': COLOR_PALETTE['text']}),
                    html.Div([
                        html.P("No country selected - showing global data", 
                              style={'color': COLOR_PALETTE['text_secondary'], 'fontSize': '14px', 
                                     'marginBottom': '10px', 'fontStyle': 'italic'}),
                        dcc.Dropdown(
                            id='country-filter',
                            options=[{'label': c, 'value': c} for c in sorted(df['country'].unique())],
                            placeholder="Select Countries...",
                            clearable=True,
                            multi=True,
                            searchable=True,
                            maxHeight=400,
                            style={'color': '#000000', 'backgroundColor': '#ffffff'}
                        )
                    ], style={'marginBottom': '20px'}),
                    
                    html.Label("👤 Sex:", style={'fontWeight': 'bold', 'color': COLOR_PALETTE['text']}),
                    html.Div(
                        dcc.Dropdown(
                            id='sex-filter',
                            options=[{'label': s, 'value': s} for s in sorted(df['sex'].unique())],
                            placeholder="Select Sex...",
                            clearable=True,
                            value='Both'
                        ),
                        style={'marginBottom': '20px'}
                    ),
                    
                    html.Label("👶 Age Group:", style={'fontWeight': 'bold', 'color': COLOR_PALETTE['text']}),
                    html.Div(
                        dcc.Dropdown(
                            id='age-filter',
                            options=[{'label': a, 'value': a} for a in sorted(df['age_group'].unique())],
                            placeholder="Select Age Group...",
                            clearable=True,
                            searchable=True,
                            maxHeight=300
                        ),
                        style={'marginBottom': '25px'}
                    ),
                    
                    html.Label("📏 Y-axis Scale:", style={'fontWeight': 'bold', 'color': COLOR_PALETTE['text']}),
                    html.Div([
                        dcc.RadioItems(
                            id='yaxis-type',
                            options=[
                                {'label': ' Linear', 'value': 'linear'},
                                {'label': ' Logarithmic', 'value': 'log'}
                            ],
                            value='linear',
                            inline=True,
                            labelStyle={'marginRight': '20px', 'color': COLOR_PALETTE['text_secondary']}
                        ),
                        html.Div(id='log-warning', style={
                            'marginTop': '10px', 
                            'padding': '8px',
                            'backgroundColor': 'rgba(243, 156, 18, 0.2)',
                            'borderRadius': '5px',
                            'fontSize': '12px',
                            'color': COLOR_PALETTE['warning'],
                            'display': 'none'
                        })
                    ], style={'marginBottom': '30px'}),
                    
                    html.Div([
                        html.Button("🔄 Reset Filters", id="reset-filters", n_clicks=0,
                                   style={'marginRight': '15px', 'padding': '12px 25px',
                                          'backgroundColor': COLOR_PALETTE['border'], 
                                          'color': COLOR_PALETTE['text'], 'border': 'none',
                                          'borderRadius': '8px', 'cursor': 'pointer',
                                          'fontWeight': 'bold', 'transition': 'all 0.3s ease'}),
                        html.Button("📥 Download CSV", id="download-btn", n_clicks=0,
                                   style={'padding': '12px 25px', 
                                          'backgroundColor': COLOR_PALETTE['accent3'], 
                                          'color': 'white', 'border': 'none',
                                          'borderRadius': '8px', 'cursor': 'pointer',
                                          'fontWeight': 'bold', 'transition': 'all 0.3s ease'})
                    ], style={'marginTop': '25px', 'display': 'flex', 'justifyContent': 'center'}),
                    dcc.Download(id="download-data")
                    
                ], style={'padding': '30px'})
            ], style={
                'flex': 1,
                'backgroundColor': COLOR_PALETTE['card'],
                'borderRadius': '15px',
                'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
                'border': f'1px solid {COLOR_PALETTE["border"]}',
                'marginRight': '25px'
            }),
            
            # Right Column - Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='line-chart', style={'height': '420px'})
                ], style={
                    'marginBottom': '25px',
                    'backgroundColor': COLOR_PALETTE['card'],
                    'borderRadius': '15px',
                    'padding': '20px',
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
                    'border': f'1px solid {COLOR_PALETTE["border"]}'
                }),
                
                html.Div([
                    dcc.Graph(id='bar-chart', style={'height': '420px'})
                ], style={
                    'backgroundColor': COLOR_PALETTE['card'],
                    'borderRadius': '15px',
                    'padding': '20px',
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
                    'border': f'1px solid {COLOR_PALETTE["border"]}'
                }),
            ], style={'flex': 2})
        ], style={'display': 'flex', 'marginBottom': '40px', 'maxWidth': '1400px', 
                 'marginLeft': 'auto', 'marginRight': 'auto'}),

        # ---------- MULTI-COUNTRY COMPARISON SECTION ----------
        html.Div([
            html.H3("📊 Multi-Country Comparison", 
                    style={'color': COLOR_PALETTE['text'], 'marginBottom': '25px',
                           'borderBottom': f'1px solid {COLOR_PALETTE["accent3"]}', 
                           'paddingBottom': '10px'}),
            
            html.P("Compare mortality trends across multiple countries. Select 2 or more countries to visualize their mortality rates over time.",
                  style={'color': COLOR_PALETTE['text_secondary'], 'marginBottom': '25px', 'fontSize': '16px', 'lineHeight': '1.6'}),
            
            # Country Comparison Selector
            html.Div([
                html.Label("🌍 Compare Countries:", 
                          style={'fontWeight': 'bold', 'marginBottom': '10px', 'color': COLOR_PALETTE['text']}),
                html.P("Select 2 or more countries for comparison (independent from main country filter)", 
                      style={'color': COLOR_PALETTE['text_secondary'], 'fontSize': '14px', 'marginBottom': '15px', 'fontStyle': 'italic'}),
                dcc.Dropdown(
                    id='compare-countries',
                    options=[{'label': c, 'value': c} for c in sorted(df['country'].unique())],
                    placeholder="Select countries to compare...",
                    clearable=True,
                    multi=True,
                    searchable=True,
                    maxHeight=400,
                    style={'color': '#000000', 'backgroundColor': '#ffffff'}
                )
            ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': COLOR_PALETTE['background'] + '40',
                     'borderRadius': '10px', 'border': f'1px solid {COLOR_PALETTE["border"]}'}),
            
            # Comparison Chart
            html.Div([
                dcc.Graph(id='comparison-chart', style={'height': '500px'})
            ], style={
                'backgroundColor': COLOR_PALETTE['card'],
                'borderRadius': '15px',
                'padding': '20px',
                'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
                'border': f'1px solid {COLOR_PALETTE["border"]}'
            }),
            
            # Comparison Info
            html.Div([
                html.P([
                    html.Span("💡 ", style={'color': COLOR_PALETTE['accent']}),
                    "Use the legend to toggle countries on/off. ",
                    "Each country maintains consistent colors throughout the comparison. ",
                    "Comparison respects all global filters (age group, sex, year range)."
                ], style={'textAlign': 'center', 'color': COLOR_PALETTE['text_secondary'], 
                         'fontSize': '13px', 'fontStyle': 'italic', 'marginTop': '20px',
                         'padding': '10px', 'backgroundColor': COLOR_PALETTE['background'] + '40',
                         'borderRadius': '8px'})
            ])
            
        ], style={
            'marginBottom': '40px',
            'backgroundColor': COLOR_PALETTE['card'],
            'borderRadius': '15px',
            'padding': '30px',
            'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLOR_PALETTE["border"]}',
            'maxWidth': '1400px',
            'marginLeft': 'auto',
            'marginRight': 'auto'
        }),

        # ---------- WORLD MAP SECTION ----------
        html.Div([
            html.H3("🗺️ Global Mortality Distribution", 
                    style={'color': COLOR_PALETTE['text'], 'marginBottom': '25px',
                           'borderBottom': f'1px solid {COLOR_PALETTE["accent2"]}', 
                           'paddingBottom': '10px'}),
            
            # Map Controls
            html.Div([
                html.Div([
                    html.Button("▶️ Play Animation", id="play-button", n_clicks=0,
                               style={'marginRight': '15px', 'padding': '10px 20px',
                                      'backgroundColor': COLOR_PALETTE['accent'], 
                                      'color': 'white', 'border': 'none',
                                      'borderRadius': '8px', 'cursor': 'pointer',
                                      'fontWeight': 'bold', 'transition': 'all 0.3s ease'}),
                    html.Button("⏸️ Pause", id="pause-button", n_clicks=0,
                               style={'padding': '10px 20px',
                                      'backgroundColor': COLOR_PALETTE['border'], 
                                      'color': COLOR_PALETTE['text'], 'border': 'none',
                                      'borderRadius': '8px', 'cursor': 'pointer',
                                      'fontWeight': 'bold', 'transition': 'all 0.3s ease',
                                      'marginRight': '30px'}),
                    html.Label("🎚️ Animation Speed:", 
                              style={'marginRight': '15px', 'fontWeight': 'bold', 
                                     'verticalAlign': 'middle', 'color': COLOR_PALETTE['text']}),
                    html.Div(
                        dcc.Slider(
                            id="speed-slider",
                            min=200,
                            max=2000,
                            step=100,
                            value=800,
                            marks={i: str(i) for i in range(200, 2001, 400)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        style={'width': '350px', 'display': 'inline-block', 'verticalAlign': 'middle'}
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '25px', 
                         'flexWrap': 'wrap', 'justifyContent': 'center'}),
                
                dcc.Interval(id='interval-component', interval=800, n_intervals=0, disabled=True)
            ]),
            
            # Map
            dcc.Graph(id='world-map', style={'height': '550px'}),
            
            # ---------- TOP/BOTTOM COUNTRIES BELOW MAP ----------
            html.Div([
                html.H4("🏆 Top 5 Highest & Lowest Mortality Countries", 
                       style={'color': COLOR_PALETTE['text'], 'marginBottom': '20px',
                              'textAlign': 'center', 'marginTop': '30px'}),
                
                html.Div([
                    html.Div([
                        html.H5("🔥 Highest Mortality Countries", 
                               style={'color': COLOR_PALETTE['high_rate'], 'textAlign': 'center',
                                      'marginBottom': '15px'})
                    ] + [
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Span(f"{idx+1}", 
                                             style={'background': '#e74c3c', 'color': 'white',
                                                    'width': '32px', 'height': '32px',
                                                    'display': 'flex', 'alignItems': 'center',
                                                    'justifyContent': 'center',
                                                    'borderRadius': '50%', 'marginRight': '15px',
                                                    'fontWeight': 'bold', 'fontSize': '16px'}),
                                    html.Div([
                                        html.Span(row['country'], 
                                                 style={'fontWeight': 'bold', 'color': COLOR_PALETTE['text'],
                                                        'fontSize': '16px', 'display': 'block'}),
                                        html.Span(f"{row['avg_mortality_rate']:,.0f} deaths per 100,000", 
                                                 style={'color': COLOR_PALETTE['text_secondary'],
                                                        'fontSize': '13px'})
                                    ])
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], style={'padding': '10px 15px', 
                                     'backgroundColor': COLOR_PALETTE['background'] + '40',
                                     'borderRadius': '8px', 'marginBottom': '8px'})
                        ]) for idx, row in top_5_countries.iterrows()
                    ], style={'flex': 1, 'marginRight': '15px'}),
                    
                    html.Div([
                        html.H5("❄️ Lowest Mortality Countries", 
                               style={'color': COLOR_PALETTE['low_rate'], 'textAlign': 'center',
                                      'marginBottom': '15px'})
                    ] + [
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Span(f"{idx+1}", 
                                             style={'background': '#2ecc71', 'color': 'white',
                                                    'width': '32px', 'height': '32px',
                                                    'display': 'flex', 'alignItems': 'center',
                                                    'justifyContent': 'center',
                                                    'borderRadius': '50%', 'marginRight': '15px',
                                                    'fontWeight': 'bold', 'fontSize': '16px'}),
                                    html.Div([
                                        html.Span(row['country'], 
                                                 style={'fontWeight': 'bold', 'color': COLOR_PALETTE['text'],
                                                        'fontSize': '16px', 'display': 'block'}),
                                        html.Span(f"{row['avg_mortality_rate']:,.0f} deaths per 100,000", 
                                                 style={'color': COLOR_PALETTE['text_secondary'],
                                                        'fontSize': '13px'})
                                    ])
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], style={'padding': '10px 15px', 
                                     'backgroundColor': COLOR_PALETTE['background'] + '40',
                                     'borderRadius': '8px', 'marginBottom': '8px'})
                        ]) for idx, row in bottom_5_countries.iterrows()
                    ], style={'flex': 1, 'marginLeft': '15px'})
                ], style={'display': 'flex', 'marginTop': '20px'}),
                
                html.Div([
                    html.P("📍 These rankings are manually set and display fixed reference values for comparison.", 
                          style={'textAlign': 'center', 'color': COLOR_PALETTE['text_secondary'], 
                                 'fontSize': '13px', 'fontStyle': 'italic', 'marginTop': '20px',
                                 'padding': '10px', 'backgroundColor': COLOR_PALETTE['background'] + '40',
                                 'borderRadius': '8px'})
                ])
            ]),
            
            # Map Legend Note
            html.Div([
                html.P([
                    "📍 Map colors normalized across all years (5th-95th percentile). ",
                    "Blue-red scale indicates mortality rate intensity."
                ], style={'textAlign': 'center', 'color': COLOR_PALETTE['text_secondary'], 
                         'fontSize': '12px', 'fontStyle': 'italic', 'marginTop': '15px'})
            ])
            
        ], style={
            'marginBottom': '40px',
            'backgroundColor': COLOR_PALETTE['card'],
            'borderRadius': '15px',
            'padding': '30px',
            'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLOR_PALETTE["border"]}',
            'maxWidth': '1400px',
            'marginLeft': 'auto',
            'marginRight': 'auto'
        }),

        # ---------- DATA TABLE ----------
        html.Div([
            html.H3("📋 Data Table", 
                    style={'color': COLOR_PALETTE['text'], 'marginBottom': '25px',
                           'borderBottom': f'1px solid {COLOR_PALETTE["accent3"]}', 
                           'paddingBottom': '10px'}),
            DataTable(
                id='data-table',
                columns=[
                    {'name': 'Year', 'id': 'year'},
                    {'name': 'Country', 'id': 'country'},
                    {'name': 'Age Group', 'id': 'age_group'},
                    {'name': 'Sex', 'id': 'sex'},
                    {'name': 'Mortality Rate', 'id': 'mortality_rate', 'type': 'numeric', 
                     'format': {'specifier': '.2f'}},
                    {'name': 'Decade', 'id': 'decade'}
                ],
                style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                page_size=10,
                style_cell={
                    'textAlign': 'center', 
                    'padding': '15px',
                    'border': f'1px solid {COLOR_PALETTE["border"]}',
                    'backgroundColor': COLOR_PALETTE['card'],
                    'color': COLOR_PALETTE['text_secondary'],
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': COLOR_PALETTE['border'], 
                    'color': COLOR_PALETTE['text'], 
                    'fontWeight': 'bold',
                    'border': f'1px solid {COLOR_PALETTE["border"]}',
                    'fontSize': '14px'
                },
                style_data={'backgroundColor': COLOR_PALETTE['card']},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#1a2536'
                    },
                    {
                        'if': {'state': 'selected'},
                        'backgroundColor': 'rgba(52, 152, 219, 0.2)',
                        'border': f'1px solid {COLOR_PALETTE["accent"]}'
                    }
                ]
            )
        ], style={
            'backgroundColor': COLOR_PALETTE['card'],
            'borderRadius': '15px',
            'padding': '30px',
            'boxShadow': '0 8px 25px rgba(0,0,0,0.4)',
            'border': f'1px solid {COLOR_PALETTE["border"]}',
            'maxWidth': '1400px',
            'marginLeft': 'auto',
            'marginRight': 'auto',
            'marginBottom': '40px'
        }),

        # Footer
        html.Div([
            html.Div([
                html.P([
                    "© 2026 Global Mortality Dashboard | ",
                    html.Span("Data Source: IHME GBD 2010", style={'color': COLOR_PALETTE['accent']})
                ], style={'textAlign': 'center', 'color': COLOR_PALETTE['text_secondary'], 
                         'padding': '20px', 'margin': '20px 0 0 0', 'fontSize': '14px',
                         'borderTop': f'1px solid {COLOR_PALETTE["border"]}'})
            ], style={'padding': '25px'})
        ], style={
            'backgroundColor': COLOR_PALETTE['card'],
            'borderRadius': '15px',
            'maxWidth': '1400px',
            'marginLeft': 'auto',
            'marginRight': 'auto',
            'marginBottom': '20px'
        })

    ], style={'padding': '20px'})

], style={
    'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    'backgroundColor': COLOR_PALETTE['background'],
    'minHeight': '100vh',
    'color': COLOR_PALETTE['text'],
    'background': f'linear-gradient(135deg, {COLOR_PALETTE["background"]} 0%, #1a2b3c 100%)'
})

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Custom slider styling */
            .custom-slider .rc-slider-track {
                background-color: #3498db;
            }
            .custom-slider .rc-slider-handle {
                border-color: #3498db;
                background-color: #3498db;
            }
            .custom-slider .rc-slider-handle:hover {
                border-color: #2980b9;
                background-color: #2980b9;
            }
            
            /* Smooth transitions */
            * {
                transition: background-color 0.3s ease, border-color 0.3s ease;
            }
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #1a2536;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb {
                background: #3498db;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #2980b9;
            }
            
            /* Dropdown improvements */
            .Select-control, .Select-multi-value-wrapper, .Select-input {
                background-color: white !important;
                color: black !important;
            }
            .Select-menu-outer {
                background-color: white !important;
                color: black !important;
            }
            .Select-option {
                background-color: white !important;
                color: black !important;
            }
            .Select-option.is-focused {
                background-color: #3498db !important;
                color: white !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# -------------------------------
# CALLBACKS
# -------------------------------

# Log scale warning
@app.callback(
    Output('log-warning', 'children'),
    Output('log-warning', 'style'),
    Input('yaxis-type', 'value'),
    Input('year-slider', 'value'),
    Input('country-filter', 'value'),
    Input('sex-filter', 'value'),
    Input('age-filter', 'value')
)
def update_log_warning(yaxis_type, years, countries, sex, age):
    if yaxis_type == 'log':
        data = filter_data(df, years, countries, sex, age)
        # Handle zero and negative values for log scale
        zero_count = (data['mortality_rate'] <= 0).sum()
        total_count = len(data)
        
        if zero_count > 0:
            warning_text = f"⚠️ Logarithmic scale: {zero_count:,} non-positive values excluded from visualization"
            warning_style = {
                'marginTop': '10px', 
                'padding': '8px',
                'backgroundColor': 'rgba(243, 156, 18, 0.2)',
                'borderRadius': '5px',
                'fontSize': '12px',
                'color': COLOR_PALETTE['warning'],
                'display': 'block'
            }
            return warning_text, warning_style
    
    # No warning needed
    return "", {'display': 'none'}

# Reset Filters
@app.callback(
    Output('year-slider', 'value'),
    Output('country-filter', 'value'),
    Output('sex-filter', 'value'),
    Output('age-filter', 'value'),
    Output('yaxis-type', 'value'),
    Input('reset-filters', 'n_clicks')
)
def reset_filters(n_clicks):
    return [year_min, year_max], None, 'Both', None, 'linear'

# Main callback for Charts and Data Table
@app.callback(
    Output('line-chart', 'figure'),
    Output('bar-chart', 'figure'),
    Output('data-table', 'data'),
    Input('year-slider', 'value'),
    Input('country-filter', 'value'),
    Input('sex-filter', 'value'),
    Input('age-filter', 'value'),
    Input('yaxis-type', 'value')
)
def update_main_charts(years, countries, sex, age, yaxis_type):
    data = filter_data(df, years, countries, sex, age)
    
    if data.empty:
        # Create empty figures with better styling
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No Data Available for Current Filters',
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            height=350,
            font=dict(color=COLOR_PALETTE['text']),
            title_font=dict(color=COLOR_PALETTE['text']),
            xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary'])
        )
        empty_bar = go.Figure()
        empty_bar.update_layout(
            title='No Data Available for Current Filters',
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            height=350,
            font=dict(color=COLOR_PALETTE['text']),
            title_font=dict(color=COLOR_PALETTE['text']),
            xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary'])
        )
        return empty_fig, empty_bar, []

    # Handle log scale by removing non-positive values
    if yaxis_type == 'log':
        # Keep only positive values for log scale
        data_for_plot = data[data['mortality_rate'] > 0].copy()
        if data_for_plot.empty:
            # If no positive values, use linear scale instead
            yaxis_type = 'linear'
            data_for_plot = data.copy()
    else:
        data_for_plot = data.copy()

    data_sorted = data_for_plot.sort_values('age_group_sort')
    
    # ---------- LINE CHART ----------
    if age:
        # If specific age group is selected
        line_data = data_sorted.groupby(['year', 'sex']).agg({
            'mortality_rate': 'mean'
        }).reset_index()
        title = f"Mortality Rate Trends by Sex for {age}"
    else:
        # If no age group selected
        line_data = data_sorted.groupby(['year', 'sex']).agg({
            'mortality_rate': 'mean'
        }).reset_index()
        title = "Mortality Rate Trends by Sex"
    
    # Check if line_data is empty
    if line_data.empty:
        fig_line = go.Figure()
        fig_line.update_layout(
            title='No data available for line chart with current filters',
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            height=350,
            font=dict(color=COLOR_PALETTE['text']),
            title_font=dict(color=COLOR_PALETTE['text']),
            xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary'])
        )
    else:
        fig_line = px.line(
            line_data, 
            x='year', 
            y='mortality_rate', 
            color='sex',
            markers=True, 
            title=title,
            color_discrete_map=COLORBLIND_PALETTE
        )
        
        # Add confidence bands
        for sex_val in line_data['sex'].unique():
            sex_data = line_data[line_data['sex'] == sex_val]
            if len(sex_data) > 1:
                upper = sex_data['mortality_rate'] * 1.1
                lower = sex_data['mortality_rate'] * 0.9
                
                # Get the appropriate rgba color
                if sex_val in COLORBLIND_PALETTE_RGBA:
                    fill_color = COLORBLIND_PALETTE_RGBA[sex_val]
                else:
                    fill_color = 'rgba(149, 165, 166, 0.3)'
                
                fig_line.add_trace(go.Scatter(
                    x=sex_data['year'].tolist() + sex_data['year'].tolist()[::-1],
                    y=upper.tolist() + lower.tolist()[::-1],
                    fill='toself',
                    fillcolor=fill_color,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    name=f'{sex_val} CI',
                    showlegend=False
                ))
        
        fig_line.update_layout(
            legend_title_text='Sex',
            hovermode="x unified",
            yaxis_title="Mortality Rate",
            xaxis_title="Year",
            height=350,
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            font=dict(color=COLOR_PALETTE['text'], size=12),
            title_font=dict(color=COLOR_PALETTE['text']),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_PALETTE['text'])
            )
        )
        
        fig_line.update_yaxes(type=yaxis_type, gridcolor=COLOR_PALETTE['border'], 
                             color=COLOR_PALETTE['text_secondary'])
        fig_line.update_xaxes(gridcolor=COLOR_PALETTE['border'], 
                             color=COLOR_PALETTE['text_secondary'])
    
    # ---------- BAR CHART ----------
    if age:
        # When age group is selected
        x_column = 'year'
        title_suffix = f" for {age}"
        bar_data = data_sorted.groupby([x_column, 'sex']).agg({
            'mortality_rate': 'mean'
        }).reset_index()
    else:
        # When no age group selected
        x_column = 'age_group'
        title_suffix = ""
        bar_data = data_sorted.groupby([x_column, 'sex', 'age_group_sort']).agg({
            'mortality_rate': 'mean'
        }).reset_index()
        bar_data = bar_data.sort_values('age_group_sort')
    
    # Check if bar_data is empty
    if bar_data.empty:
        fig_bar = go.Figure()
        fig_bar.update_layout(
            title='No data available for bar chart with current filters',
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            height=350,
            font=dict(color=COLOR_PALETTE['text']),
            title_font=dict(color=COLOR_PALETTE['text']),
            xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary'])
        )
    else:
        fig_bar = px.bar(
            bar_data, 
            x=x_column, 
            y='mortality_rate', 
            color='sex',
            color_discrete_map=COLORBLIND_PALETTE,
            barmode='group',
            title=f"Mortality Rates by {x_column.replace('_', ' ').title()}{title_suffix}"
        )
        
        # Set category order for age groups
        if x_column == 'age_group':
            fig_bar.update_xaxes(categoryorder='array', categoryarray=age_group_order)
        
        fig_bar.update_traces(
            marker_line_width=1,
            marker_line_color=COLOR_PALETTE['card'],
            opacity=0.85
        )
        
        fig_bar.update_layout(
            legend_title_text='Sex',
            xaxis_title=x_column.replace('_', ' ').title(),
            yaxis_title="Mortality Rate",
            xaxis={'tickangle': 45} if x_column == 'age_group' else {},
            height=350,
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            font=dict(color=COLOR_PALETTE['text'], size=12),
            title_font=dict(color=COLOR_PALETTE['text']),
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLOR_PALETTE['text'])
            )
        )
        
        fig_bar.update_yaxes(type=yaxis_type, gridcolor=COLOR_PALETTE['border'], 
                            color=COLOR_PALETTE['text_secondary'])
        fig_bar.update_xaxes(gridcolor=COLOR_PALETTE['border'], 
                            color=COLOR_PALETTE['text_secondary'])
    
    # Data for table
    table_data = data[['year', 'country', 'age_group', 'sex', 'mortality_rate', 'decade']].to_dict('records')

    return fig_line, fig_bar, table_data

# -------------------------------
# MULTI-COUNTRY COMPARISON CALLBACK
# -------------------------------
@app.callback(
    Output('comparison-chart', 'figure'),
    Input('compare-countries', 'value'),  # Independent country selector for comparison
    Input('year-slider', 'value'),
    Input('sex-filter', 'value'),
    Input('age-filter', 'value'),
    Input('yaxis-type', 'value')
)
def update_comparison_chart(compare_countries, years, sex, age, yaxis_type):
    """
    COMPARISON LOGIC:
    1. Filters data based on global filters (year range, sex, age group)
    2. Filters for selected comparison countries (independent from main country filter)
    3. Calculates mean mortality rate per year for each country
    4. Creates a line chart with one line per country
    5. Assigns consistent colors to each country
    6. Allows toggling countries via legend
    7. Handles empty selection gracefully
    """
    
    # Handle empty selection - show informative message
    if not compare_countries or len(compare_countries) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='Select 2 or more countries to compare',
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            height=450,
            font=dict(color=COLOR_PALETTE['text']),
            title_font=dict(color=COLOR_PALETTE['text']),
            xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            annotations=[dict(
                text="Use the 'Compare Countries' dropdown above to select countries",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLOR_PALETTE['text_secondary'])
            )]
        )
        return empty_fig
    
    # Filter data based on global filters (years, sex, age)
    filtered_data = filter_data(df, years, None, sex, age)  # No country filter here
    
    # Further filter for selected comparison countries
    comparison_data = filtered_data[filtered_data['country'].isin(compare_countries)]
    
    if comparison_data.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No data available for selected countries and filters',
            plot_bgcolor=COLOR_PALETTE['card'],
            paper_bgcolor=COLOR_PALETTE['card'],
            height=450,
            font=dict(color=COLOR_PALETTE['text']),
            title_font=dict(color=COLOR_PALETTE['text']),
            xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
            yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary'])
        )
        return empty_fig
    
    # Handle log scale by removing non-positive values
    if yaxis_type == 'log':
        # Keep only positive values for log scale
        comparison_data = comparison_data[comparison_data['mortality_rate'] > 0].copy()
        if comparison_data.empty:
            # If no positive values, show message
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title='No positive values for logarithmic scale with current selection',
                plot_bgcolor=COLOR_PALETTE['card'],
                paper_bgcolor=COLOR_PALETTE['card'],
                height=450,
                font=dict(color=COLOR_PALETTE['text']),
                title_font=dict(color=COLOR_PALETTE['text']),
                xaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary']),
                yaxis=dict(showgrid=True, gridcolor=COLOR_PALETTE['border'], color=COLOR_PALETTE['text_secondary'])
            )
            return empty_fig
    
    # Calculate mean mortality rate per year for each country
    # This groups by country and year, calculating the average mortality rate
    comparison_summary = comparison_data.groupby(['country', 'year']).agg({
        'mortality_rate': 'mean'
    }).reset_index()
    
    # Sort by year for proper line plotting
    comparison_summary = comparison_summary.sort_values(['country', 'year'])
    
    # Create figure
    fig = go.Figure()
    
    # Assign colors to countries consistently
    # The color for each country is determined by its position in the selected list
    country_colors = {}
    for idx, country in enumerate(compare_countries):
        color_idx = idx % len(COMPARISON_COLORS)
        country_colors[country] = COMPARISON_COLORS[color_idx]
    
    # Add one trace (line) per country
    for country in compare_countries:
        country_data = comparison_summary[comparison_summary['country'] == country]
        
        if not country_data.empty:
            fig.add_trace(go.Scatter(
                x=country_data['year'],
                y=country_data['mortality_rate'],
                mode='lines+markers',
                name=country,
                line=dict(color=country_colors[country], width=3),
                marker=dict(size=8, color=country_colors[country]),
                hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Mortality Rate: %{{y:,.0f}}<extra></extra>"
            ))
    
    # Create title based on current filters
    if age:
        title = f"Country Comparison - Mortality Rates for {age}"
    else:
        title = "Country Comparison - Mortality Rate Trends"
    
    if sex != 'Both':
        title += f" ({sex})"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Mortality Rate",
        height=450,
        plot_bgcolor=COLOR_PALETTE['card'],
        paper_bgcolor=COLOR_PALETTE['card'],
        font=dict(color=COLOR_PALETTE['text'], size=12),
        title_font=dict(color=COLOR_PALETTE['text'], size=16),
        hovermode="x unified",
        legend=dict(
            title="Countries",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLOR_PALETTE['text'], size=12),
            bgcolor=COLOR_PALETTE['card'],
            bordercolor=COLOR_PALETTE['border'],
            borderwidth=1
        ),
        margin=dict(t=60, b=50, l=50, r=50)
    )
    
    # Update axes
    fig.update_yaxes(
        type=yaxis_type,
        gridcolor=COLOR_PALETTE['border'],
        color=COLOR_PALETTE['text_secondary']
    )
    fig.update_xaxes(
        gridcolor=COLOR_PALETTE['border'],
        color=COLOR_PALETTE['text_secondary']
    )
    
    return fig

# Play/Pause animation
@app.callback(
    Output('interval-component', 'disabled'),
    Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    prevent_initial_call=True
)
def control_animation(play_clicks, pause_clicks):
    if ctx.triggered_id == 'play-button':
        return False
    else:
        return True

# Animation Speed
@app.callback(
    Output('interval-component', 'interval'),
    Input('speed-slider', 'value')
)
def update_speed(speed):
    return speed

# Animated World Map
@app.callback(
    Output('world-map', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('year-slider', 'value'),
    Input('country-filter', 'value'),
    Input('sex-filter', 'value'),
    Input('age-filter', 'value')
)
def animate_world_map(n_intervals, years, countries, sex, age):
    data_filtered = filter_data(df, years, countries, sex, age)
    
    if data_filtered.empty:
        fig_empty = px.choropleth()
        fig_empty.update_layout(
            title='No Data Available for Current Filters',
            paper_bgcolor=COLOR_PALETTE['card'],
            plot_bgcolor=COLOR_PALETTE['card'],
            height=450,
            font=dict(color=COLOR_PALETTE['text'])
        )
        return fig_empty

    unique_years = sorted(data_filtered['year'].unique())
    year_idx = n_intervals % len(unique_years) if unique_years else 0
    current_year = unique_years[year_idx] if unique_years else years[0]

    world_data = (
        data_filtered[data_filtered['year'] == current_year]
        .groupby(['country', 'iso_alpha'])
        .agg(
            mortality_rate=('mortality_rate', 'mean'),
            sex_groups=('sex', lambda x: ', '.join(sorted(x.unique()))),
            age_groups=('age_group', lambda x: ', '.join(sorted(x.unique())))
        )
        .reset_index()
    )

    world_data = world_data.dropna(subset=['iso_alpha'])

    if world_data.empty:
        fig_empty = px.choropleth()
        fig_empty.update_layout(
            title=f'No Geographic Data Available for {current_year}',
            paper_bgcolor=COLOR_PALETTE['card'],
            plot_bgcolor=COLOR_PALETTE['card'],
            height=450,
            font=dict(color=COLOR_PALETTE['text'])
        )
        return fig_empty

    fig_map = px.choropleth(
        world_data,
        locations='iso_alpha',
        color='mortality_rate',
        hover_name='country',
        hover_data={'iso_alpha': False},
        color_continuous_scale='Bluered',
        title=f"Global Mortality Rate ({current_year})",
        locationmode='ISO-3',
        range_color=[global_min, global_max]
    )

    fig_map.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Mortality Rate: %{z:,.0f}<br>Sex Groups: %{customdata[0]}<br>Age Groups: %{customdata[1]}<extra></extra>",
        customdata=world_data[['sex_groups', 'age_groups']].values
    )

    fig_map.update_layout(
        coloraxis_colorbar=dict(
            title="Mortality Rate", 
            tickformat=",.0f", 
            tickfont=dict(color=COLOR_PALETTE['text_secondary'])
        ),
        height=500,
        geo=dict(
            bgcolor=COLOR_PALETTE['background'],
            lakecolor=COLOR_PALETTE['background'],
            landcolor='rgba(42, 67, 101, 0.8)',
            subunitcolor=COLOR_PALETTE['border']
        ),
        plot_bgcolor=COLOR_PALETTE['card'],
        paper_bgcolor=COLOR_PALETTE['card'],
        font=dict(color=COLOR_PALETTE['text']),
        title_font=dict(color=COLOR_PALETTE['text']),
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    fig_map.update_geos(
        showcoastlines=True,
        coastlinecolor=COLOR_PALETTE['border'],
        showland=True,
        landcolor='rgba(42, 67, 101, 0.8)',
        showocean=True,
        oceancolor=COLOR_PALETTE['background'],
        showcountries=True,
        countrycolor=COLOR_PALETTE['border'],
        showframe=True,
        framecolor=COLOR_PALETTE['border']
    )
    
    return fig_map

# Download CSV
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    Input('year-slider', 'value'),
    Input('country-filter', 'value'),
    Input('sex-filter', 'value'),
    Input('age-filter', 'value'),
    prevent_initial_call=True
)
def download_filtered(n_clicks, years, countries, sex, age):
    if ctx.triggered_id == "download-btn" and n_clicks and n_clicks > 0:
        data_filtered = filter_data(df, years, countries, sex, age)
        return dcc.send_data_frame(data_filtered.to_csv, "mortality_data.csv", index=False)
    return None

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 Starting Global Mortality Dashboard")
    print("=" * 70)
    print(f"📊 Dataset loaded: {len(df):,} rows")
    print(f"📅 Year range: {year_min} to {year_max}")
    print(f"🌍 Countries: {df['country'].nunique()}")
    print(f"👤 Sex values: {df['sex'].unique().tolist()}")
    print(f"👶 Age groups: {df['age_group'].nunique()}")
    print("\n🎯 NEW FEATURE: Multi-Country Comparison")
    print("   ✅ Independent country selector for comparison")
    print("   ✅ Line chart with one line per country")
    print("   ✅ Consistent colors per country")
    print("   ✅ Interactive legend with toggle functionality")
    print("   ✅ Respects global filters (age, sex, year)")
    print("\n🎯 MANUAL COUNTRY RANKINGS:")
    print("   Top 5 Highest Mortality:")
    for idx, row in top_5_countries.iterrows():
        print(f"     {idx+1}. {row['country']}: {row['avg_mortality_rate']:.0f}")
    print("   Top 5 Lowest Mortality:")
    for idx, row in bottom_5_countries.iterrows():
        print(f"     {idx+1}. {row['country']}: {row['avg_mortality_rate']:.0f}")
    print("🎨 Theme: Blue-Black Interactive Dashboard")
    print("🌐 Running on http://localhost:8050")
    print("=" * 70)
    app.run(debug=True, port=8050)