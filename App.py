import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import numpy as np
import json
import requests
import matplotlib.cm as cm  # For Viridis color map
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


@st.cache_data
def calculate_key_metrics(country_df):
    # Global temperature change since pre-industrial times (using 1850-1900 as baseline)
    baseline_period = country_df[(country_df['Year'] >= 1850) & (country_df['Year'] <= 1900)]
    baseline_temp = baseline_period['AverageTemperature'].mean()
    current_period = country_df[country_df['Year'] >= 2010]
    current_temp = current_period['AverageTemperature'].mean()
    temp_change = current_temp - baseline_temp
    
    # Fastest warming regions (last 50 years)
    recent_data = country_df[country_df['Year'] >= (datetime.now().year - 50)]
    warming_rates = recent_data.groupby('Country').apply(
        lambda x: np.polyfit(x['Year'], x['AverageTemperature'], 1)[0]
    ).sort_values(ascending=False)
    fastest_warming = warming_rates.head(3).to_dict()
    
    # Most extreme recorded temperatures
    max_temp = country_df['AverageTemperature'].max()
    max_temp_country = country_df.loc[country_df['AverageTemperature'].idxmax(), 'Country']
    min_temp = country_df['AverageTemperature'].min()
    min_temp_country = country_df.loc[country_df['AverageTemperature'].idxmin(), 'Country']
    
    return {
        'temp_change': temp_change,
        'baseline_temp': baseline_temp,
        'current_temp': current_temp,
        'fastest_warming': fastest_warming,
        'max_temp': max_temp,
        'max_temp_country': max_temp_country,
        'min_temp': min_temp,
        'min_temp_country': min_temp_country
    }
# Load datasets
@st.cache_data
def load_data():
    # Main temperature datasets
    country_df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
    country_df['dt'] = pd.to_datetime(country_df['dt'], format='mixed')
    country_df['Year'] = country_df['dt'].dt.year

    state_df = pd.read_csv("GlobalLandTemperaturesByState.csv")
    state_df['dt'] = pd.to_datetime(state_df['dt'], format='mixed')
    state_df['Year'] = state_df['dt'].dt.year

    # Urban-rural and elevation data
    urban_rural_df = pd.read_csv("urban_rural_temp_comparison.csv")
    elevation_df = pd.read_csv("elevation_temp_relationship.csv")
    
    return country_df, state_df, urban_rural_df, elevation_df

country_df, state_df, urban_rural_df, elevation_df = load_data()


# Sidebar for user inputs
st.sidebar.title("Filters")
countries = st.sidebar.multiselect(
    "Select countries",
    country_df['Country'].unique(),
    default=["United States of America", "India", "Greenland", "New Zealand"]
)

# Filter data based on user inputs
filtered_country_df = country_df[country_df['Country'].isin(countries)]
filtered_state_df = state_df[state_df['Country'].isin(countries)]

# Title and description
st.title("Global and Regional Climate Change Visualization")
st.write("Explore trends in global temperature and regional variations.")



st.header("🌦 Seasonal Temperature Analysis")

# Add seasonal analysis
if not country_df.empty:
    # Extract month from date
    country_df['Month'] = country_df['dt'].dt.month
    
    # Define seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    country_df['Season'] = country_df['Month'].map(season_map)
    
    # Let user select country
    selected_country_season = st.selectbox(
        "Select country for seasonal analysis",
        country_df['Country'].unique(),
        key='season_country'
    )
    
    # Filter data
    seasonal_df = country_df[country_df['Country'] == selected_country_season]
    
    if not seasonal_df.empty:
        # Calculate seasonal averages by decade
        seasonal_df['Decade'] = (seasonal_df['Year'] // 10) * 10
        seasonal_avg = seasonal_df.groupby(['Decade', 'Season'])['AverageTemperature'].mean().reset_index()
        
        # Create line chart
        seasonal_chart = alt.Chart(seasonal_avg).mark_line().encode(
            x='Decade:O',
            y='AverageTemperature:Q',
            color='Season:N',
            tooltip=['Decade', 'Season', 'AverageTemperature']
        ).properties(
            width=800,
            height=400,
            title=f"Seasonal Temperature Trends in {selected_country_season}"
        ).interactive()
        
        st.altair_chart(seasonal_chart, use_container_width=True)
        
        # Calculate seasonal changes
        earliest = seasonal_avg['Decade'].min()
        latest = seasonal_avg['Decade'].max()
        seasonal_changes = seasonal_avg.pivot(index='Decade', columns='Season', values='AverageTemperature')
        seasonal_changes = seasonal_changes.loc[latest] - seasonal_changes.loc[earliest]
        
        st.subheader(f"Temperature Change by Season ({earliest}s to {latest}s)")
        st.bar_chart(seasonal_changes)
    else:
        st.warning("No seasonal data available for selected country.")
else:
    st.warning("No country data available for seasonal analysis.")

st.divider()

# Add this after the seasonal analysis section
st.header("🔮 Temperature Projection Simulator")

# Add temperature simulator
if not country_df.empty:
    # Let user select country
    selected_country_sim = st.selectbox(
        "Select country for projection",
        country_df['Country'].unique(),
        key='sim_country'
    )
    
    # Get historical data for selected country
    country_hist = country_df[country_df['Country'] == selected_country_sim]
    
    if not country_hist.empty:
        # Calculate historical trend - remove NaN values
        yearly_avg = country_hist.groupby('Year')['AverageTemperature'].mean().reset_index()
        yearly_avg = yearly_avg.dropna()  # Remove rows with NaN values
        
        # Check if we have enough data after cleaning
        if len(yearly_avg) < 2:
            st.warning(f"Not enough valid temperature data for {selected_country_sim} to create projection.")
        else:
            # Fit linear regression
            X = yearly_avg['Year'].values.reshape(-1, 1)
            y = yearly_avg['AverageTemperature'].values
            
            try:
                model = LinearRegression().fit(X, y)
                trend_slope = model.coef_[0]
                trend_intercept = model.intercept_
                
                # User inputs for simulation
                col1, col2 = st.columns(2)
                with col1:
                    future_years = st.slider(
                        "Project how many years into the future?",
                        10, 100, 50
                    )
                with col2:
                    scenario_multiplier = st.select_slider(
                        "Climate scenario",
                        options=['Low (0.5x)', 'Medium (1x)', 'High (2x)'],
                        value='Medium (1x)'
                    )
                
                # Calculate multiplier
                multiplier = 0.5 if 'Low' in scenario_multiplier else (2 if 'High' in scenario_multiplier else 1)
                
                # Generate projection
                last_year = yearly_avg['Year'].max()
                future_years_range = range(last_year, last_year + future_years + 1)
                projected_temp = [trend_intercept + trend_slope * (year - X.min()) * multiplier for year in future_years_range]
                
                # Create combined dataframe
                history_df = yearly_avg.copy()
                history_df['Type'] = 'Historical'
                projection_df = pd.DataFrame({
                    'Year': future_years_range,
                    'AverageTemperature': projected_temp,
                    'Type': 'Projected'
                })
                combined_df = pd.concat([history_df, projection_df])
                
                # Create chart
                sim_chart = alt.Chart(combined_df).mark_line().encode(
                    x='Year:O',
                    y='AverageTemperature:Q',
                    color='Type:N',
                    strokeDash='Type:N',
                    tooltip=['Year', 'AverageTemperature', 'Type']
                ).properties(
                    width=800,
                    height=400,
                    title=f"Temperature Projection for {selected_country_sim}"
                ).interactive()
                
                # Add vertical line at present
                rule = alt.Chart(pd.DataFrame({'Year': [last_year]})).mark_rule(
                    color='red',
                    strokeWidth=2
                ).encode(x='Year:O')
                
                st.altair_chart(sim_chart + rule, use_container_width=True)
                
                # Show key metrics
                temp_change = projected_temp[-1] - yearly_avg['AverageTemperature'].iloc[-1]
                st.metric(
                    label=f"Projected Temperature Change in {future_years} years",
                    value=f"{temp_change:.2f}°C",
                    delta=f"From {yearly_avg['AverageTemperature'].iloc[-1]:.1f}°C to {projected_temp[-1]:.1f}°C"
                )
                
            except Exception as e:
                st.error(f"Error creating projection: {str(e)}")
    else:
        st.warning("No historical data available for selected country.")
else:
    st.warning("No country data available for simulation.")

# Urban vs Rural Comparison
st.header("🏙️ Urban Temperature Comparison")
urban_rural_df['UHI_Intensity'] = urban_rural_df['Urban_Temp_C'] - urban_rural_df['Rural_Temp_C']

tab1, tab2 = st.tabs(["Temperature Difference", "Population Relationship"])

with tab1:
    st.subheader("Urban Heat Island Intensity")
    chart = alt.Chart(urban_rural_df).mark_bar().encode(
        x='City:N',
        y='UHI_Intensity:Q',
        color=alt.condition(
            alt.datum.UHI_Intensity > 0,
            alt.value('orange'),
            alt.value('blue')
        ),
        tooltip=['City', 'Urban_Temp_C', 'Rural_Temp_C', 'UHI_Intensity']
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

with tab2:
    st.subheader("Urban Heat Island vs Population")
    scatter = alt.Chart(urban_rural_df).mark_circle(size=60).encode(
        x='Population_Millions:Q',
        y='UHI_Intensity:Q',
        color='Country:N',
        tooltip=['City', 'Population_Millions', 'UHI_Intensity']
    ).properties(width=700, height=400)
    st.altair_chart(scatter, use_container_width=True)

st.divider()

# Elevation-Temperature Relationship
st.header("⛰️ Elevation-Temperature Relationship")
elevation_df['Temp_Change_Per_100m'] = (elevation_df['Avg_Temp_C'] - elevation_df.iloc[0]['Avg_Temp_C']) / (elevation_df['Elevation_m']/100)

col1, col2 = st.columns([3, 1])
with col1:
    chart = alt.Chart(elevation_df).mark_circle(size=100).encode(
        x='Elevation_m:Q',
        y='Avg_Temp_C:Q',
        color='Latitude:Q',
        size=alt.value(100),
        tooltip=['Location', 'Elevation_m', 'Avg_Temp_C']
    ).properties(width=600, height=400)
    line = chart.transform_regression('Elevation_m', 'Avg_Temp_C').mark_line(color='red')
    st.altair_chart(chart + line, use_container_width=True)

with col2:
    slope = np.polyfit(elevation_df['Elevation_m'], elevation_df['Avg_Temp_C'], 1)[0]
    st.metric(
        "Environmental Lapse Rate",
        f"{abs(slope * 1000):.1f}°C per 1000m",
        "Typical range: 5-10°C per 1000m"
    )

st.divider()



# Bar chart for average temperature by country
st.header("Average Temperature by Country (Bar Chart)")
if not filtered_country_df.empty:
    avg_temp_by_country = filtered_country_df.groupby('Country')['AverageTemperature'].mean().reset_index()
    bar_chart = alt.Chart(avg_temp_by_country).mark_bar().encode(
        x='Country:N',
        y='AverageTemperature:Q',
        color=alt.Color('Country:N', scale=alt.Scale(scheme='blues')),
        tooltip=['Country', 'AverageTemperature']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.warning("No data available for selected countries.")


# State-level visualizations for USA and India
st.header("State-Level Visualizations")

# Filter state data for USA and India
usa_state_df = filtered_state_df[filtered_state_df['Country'] == "United States of America"]
india_state_df = filtered_state_df[filtered_state_df['Country'] == "India"]

# Bar chart for average temperature by state (USA)
st.subheader("Average Temperature by State (USA)")
if not usa_state_df.empty:
    avg_temp_by_state_usa = usa_state_df.groupby('State')['AverageTemperature'].mean().reset_index()
    bar_chart_usa = alt.Chart(avg_temp_by_state_usa).mark_bar().encode(
        x='State:N',
        y='AverageTemperature:Q',
        color=alt.Color('State:N', scale=alt.Scale(scheme='blues')),  # Use Blues color scheme
        tooltip=['State', 'AverageTemperature']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(bar_chart_usa, use_container_width=True)
else:
    st.warning("No state-level data available for the USA.")

# Bar chart for average temperature by state (India)
st.subheader("Average Temperature by State (India)")
if not india_state_df.empty:
    avg_temp_by_state_india = india_state_df.groupby('State')['AverageTemperature'].mean().reset_index()
    bar_chart_india = alt.Chart(avg_temp_by_state_india).mark_bar().encode(
        x='State:N',
        y='AverageTemperature:Q',
        color=alt.Color('State:N', scale=alt.Scale(scheme='blues')),  # Use Blues color scheme
        tooltip=['State', 'AverageTemperature']
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(bar_chart_india, use_container_width=True)
else:
    st.warning("No state-level data available for India.")

# Choropleth map for temperature by state (USA)
st.subheader("Temperature by State (USA - Choropleth Map)")
if not usa_state_df.empty:
    # Aggregate data by state
    usa_state_avg_temp = usa_state_df.groupby('State')['AverageTemperature'].mean().reset_index()

    # Load GeoJSON data for USA states
    usa_geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    usa_geojson_data = requests.get(usa_geojson_url).json()

    # Generate colors using Viridis color map
    viridis = cm.get_cmap('viridis', len(usa_state_avg_temp))
    usa_state_avg_temp['color'] = usa_state_avg_temp.index.map(lambda x: [int(c * 255) for c in viridis(x)[:3]] + [180])

    # Merge temperature data with GeoJSON data
    for feature in usa_geojson_data['features']:
        state_name = feature['properties']['name']
        temp_value = usa_state_avg_temp.loc[usa_state_avg_temp['State'] == state_name, 'AverageTemperature']
        feature['properties']['AverageTemperature'] = float(temp_value) if not temp_value.empty else "No Data"
        color_data = usa_state_avg_temp.loc[usa_state_avg_temp['State'] == state_name, 'color']
        if not color_data.empty:
            feature['properties']['color'] = color_data.values[0]
        else:
            feature['properties']['color'] = [200, 200, 200, 160]  # Default color for missing data

    # Create choropleth map using PyDeck
    usa_layer = pdk.Layer(
        "GeoJsonLayer",
        data=usa_geojson_data,
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="properties.color",
        get_line_color=[255, 255, 255],
        pickable=True,
        auto_highlight=True,
    )

    usa_view_state = pdk.ViewState(latitude=37.0902, longitude=-95.7129, zoom=3)
    usa_tooltip = {
        "html": "<b>State:</b> {name}<br/><b>Average Temperature:</b> {AverageTemperature}°C",
        "style": {
            "backgroundColor": "black",
            "color": "white",
            "padding": "5px",
            "borderRadius": "5px",
        },
    }

    usa_r = pdk.Deck(
        layers=[usa_layer],
        initial_view_state=usa_view_state,
        tooltip=usa_tooltip,
    )

    st.pydeck_chart(usa_r)
else:
    st.warning("No state-level data available for the USA.")

# Choropleth map for temperature by state (India)
st.subheader("Temperature by State (India - Choropleth Map)")
if not india_state_df.empty:
    # Aggregate data by state
    india_state_avg_temp = india_state_df.groupby('State')['AverageTemperature'].mean().reset_index()

    # Load GeoJSON data for India states
    india_geojson_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
    india_geojson_data = requests.get(india_geojson_url).json()

    # Generate colors using Viridis color map
    viridis = cm.get_cmap('viridis', len(india_state_avg_temp))
    india_state_avg_temp['color'] = india_state_avg_temp.index.map(lambda x: [int(c * 255) for c in viridis(x)[:3]] + [180])

    # Merge temperature data with GeoJSON data
    for feature in india_geojson_data['features']:
        state_name = feature['properties']['NAME_1']
        temp_value = india_state_avg_temp.loc[india_state_avg_temp['State'] == state_name, 'AverageTemperature']
        feature['properties']['AverageTemperature'] = float(temp_value) if not temp_value.empty else "No Data"
        color_data = india_state_avg_temp.loc[india_state_avg_temp['State'] == state_name, 'color']
        if not color_data.empty:
            feature['properties']['color'] = color_data.values[0]
        else:
            feature['properties']['color'] = [200, 200, 200, 160]  # Default color for missing data

    # Create choropleth map using PyDeck
    india_layer = pdk.Layer(
        "GeoJsonLayer",
        data=india_geojson_data,
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="properties.color",
        get_line_color=[255, 255, 255],
        pickable=True,
        auto_highlight=True,
    )

    india_view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=4)
    india_tooltip = {
        "html": "<b>State:</b> {NAME_1}<br/><b>Average Temperature:</b> {AverageTemperature}°C",
        "style": {
            "backgroundColor": "black",
            "color": "white",
            "padding": "5px",
            "borderRadius": "5px",
        },
    }

    india_r = pdk.Deck(
        layers=[india_layer],
        initial_view_state=india_view_state,
        tooltip=india_tooltip,
    )

    st.pydeck_chart(india_r)
else:
    st.warning("No state-level data available for India.")

# Country-specific temperature trends over years
st.header("Temperature Trends for a Specific Country Over Years")
if not filtered_country_df.empty:
    # Select a specific country
    selected_country = st.selectbox(
        "Select a country to see its temperature trends over the years",
        filtered_country_df['Country'].unique()
    )

    # Filter data for the selected country
    country_trend_df = filtered_country_df[filtered_country_df['Country'] == selected_country]

    # Group by year and calculate average temperature
    country_trend_df = country_trend_df.groupby('Year')['AverageTemperature'].mean().reset_index()

    # Line chart for temperature trends over years
    trend_chart = alt.Chart(country_trend_df).mark_line().encode(
        x='Year:O',
        y='AverageTemperature:Q',
        tooltip=['Year', 'AverageTemperature']
    ).properties(
        width=800,
        height=400,
        title=f"Temperature Trends for {selected_country} Over Years"
    ).interactive()
    st.altair_chart(trend_chart, use_container_width=True)
else:
    st.warning("No data available for selected countries.")