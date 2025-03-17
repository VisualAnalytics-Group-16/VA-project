import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import numpy as np
import json
import requests
import matplotlib.cm as cm  # For Viridis color map

# Load datasets
@st.cache_data  # Cache data for better performance
def load_data():
    # Load Global Land Temperatures by Country
    country_df = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
    country_df['dt'] = pd.to_datetime(country_df['dt'], format='mixed')  # Convert to datetime
    country_df['Year'] = country_df['dt'].dt.year  # Extract year for aggregation

    # Load Global Land Temperatures by State
    state_df = pd.read_csv("GlobalLandTemperaturesByState.csv")
    state_df['dt'] = pd.to_datetime(state_df['dt'], format='mixed')  # Convert to datetime
    state_df['Year'] = state_df['dt'].dt.year  # Extract year for aggregation

    return country_df, state_df

country_df, state_df = load_data()

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

# Line chart for global average temperature over time
st.header("Global Average Temperature Over Time")
if not country_df.empty:
    global_avg_temp = country_df.groupby('dt')['AverageTemperature'].mean().reset_index()
    line_chart = alt.Chart(global_avg_temp).mark_line().encode(
        x='dt:T',
        y='AverageTemperature:Q',
        tooltip=['dt', 'AverageTemperature']
    ).properties(
        width=800,
        height=400
    ).interactive()
    st.altair_chart(line_chart, use_container_width=True)
else:
    st.warning("No country data available for global average temperature.")

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

# Choropleth map for temperature by country
st.header("Temperature by Country (Choropleth Map)")
if not country_df.empty:
    # Aggregate data by country
    country_avg_temp = country_df.groupby('Country')['AverageTemperature'].mean().reset_index()

    # Load GeoJSON data for countries
    geojson_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    response = requests.get(geojson_url)
    geojson_data = response.json()

    # Ensure 'name' property exists for each country
    for feature in geojson_data['features']:
        if 'properties' in feature:
            feature['properties']['name'] = feature['properties'].get('ADMIN', 'Unknown')

    # Convert GeoJSON to DataFrame
    geojson_df = pd.json_normalize(geojson_data['features'])

    # Merge temperature data with GeoJSON data
    merged_data = geojson_df.merge(country_avg_temp, left_on='properties.name', right_on='Country', how='left')

    # Generate distinct colors for different countries using Viridis color map
    viridis = cm.get_cmap('viridis', len(merged_data))
    merged_data['color'] = merged_data.index.map(lambda x: [int(c * 255) for c in viridis(x)[:3]] + [180])

    # Create choropleth map using PyDeck
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_data,  # Use correctly modified GeoJSON
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="properties.color",
        get_line_color=[255, 255, 255],
        pickable=True,  # Enable tooltips
        auto_highlight=True,
    )

    # Assign colors to GeoJSON properties
    for feature in geojson_data["features"]:
        country_name = feature["properties"].get("name", "Unknown")
        temp_value = country_avg_temp.loc[country_avg_temp["Country"] == country_name, "AverageTemperature"]

        # Assign average temperature to GeoJSON properties
        feature["properties"]["AverageTemperature"] = float(temp_value) if not temp_value.empty else "No Data"

        # Assign a unique color (handle missing data)
        color_data = merged_data.loc[merged_data['Country'] == country_name, 'color']
        if not color_data.empty:
            feature["properties"]["color"] = color_data.values[0]
        else:
            feature["properties"]["color"] = [200, 200, 200, 160]  # Default color for missing data

    # Set the initial view state for the map
    view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)

    # Tooltip to display country name and average temperature
    tooltip = {
        "html": "<b>Country:</b> {name}<br/><b>Average Temperature:</b> {AverageTemperature}°C",
        "style": {
            "backgroundColor": "black",
            "color": "white",
            "padding": "5px",
            "borderRadius": "5px",
        },
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,  # Use the corrected tooltip
    )

    st.pydeck_chart(r)

else:
    st.warning("No country data available for choropleth map.")


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