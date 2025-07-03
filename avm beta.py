import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium, folium_static
from sqlalchemy import create_engine
from folium.plugins import Draw
from shapely.geometry import shape
from helper import visualize_poi_by_wkt
from shapely import wkt
import  json
import requests
from shapely.ops import transform 
import plotly.express as px
import pyproj

# --- Database Engine ---
engine = create_engine('postgresql://kjpprhr:rhrdatascience@13.229.240.204:5432/geodb')

# --- Load Cached Grid ---

# --- UI Tabs ---


# st.title("Point Analysis Beta")



# st.write("This app is a private application for KJPP RHR. It is designed to assist in property valuation by allowing users to select a point on the map and analyze the surrounding area. The app provides insights into property prices, amenities, and other relevant factors that can influence property value.")
# st.write("Please input the password to access the app:")
# st.warning("Please contact inderaihsan@gmail.com for further information")
# password = st.text_input("Password", type="password", placeholder="Enter password here")  
# if password == st.secrets["ADMIN_PASSWORD"]: 
#     st.session_state['password_approved'] = True
# else:
#     st.session_state['password_approved'] = None
# if st.session_state['password_approved']:
# st.write("Click on map to view. analysis")
# st.success("Password accepted! You can now use the app.")
st.title("How to Use This App")
st.write("Click on the map to select a point for property analysis. The app will then provide insights into the selected area, including property prices, amenities, and other relevant factors.") 
st.write("Use the draw marker point tool to select a point on the map. After placing the point, you can fill out the property input form to get predictions and analysis.")
with st.expander() : 
    m = folium.Map(location=[-6.2, 106.8], zoom_start=11)
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "rectangle": False,
            "circlemarker": False,
            "marker": True  # Only allow points
        },
        # edit_options={"edit": False}  # Optional: disable editing after placing
    )
    draw.add_to(m)
    folium.LayerControl().add_to(m)

    click_data = st_folium(m, width=1350, height=500)

if click_data and 'last_active_drawing' in click_data and click_data['last_active_drawing']:
    geom_geojson = click_data['last_active_drawing']['geometry']
    shapely_geom = shape(geom_geojson)
    selected_geom_wkt = shapely_geom.wkt
    clicked_point = wkt.loads(selected_geom_wkt) 
        # Define CRS
    wgs84 = pyproj.CRS("EPSG:4326")
    utm = pyproj.CRS("EPSG:32749")  # UTM zone 49S (correct for Jakarta/Bogor area)

    # Create transformers
    to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

    # Transform point to UTM
    utm_point = transform(to_utm, clicked_point)

    # Buffer in meters
    buffer_utm = utm_point.buffer(1000)

    # Transform buffer back to WGS84
    buffer_wgs84 = transform(to_wgs84, buffer_utm)

    # Get WKT of buffer in WGS84
    buffer_1k = buffer_wgs84.wkt

    st.subheader("Property Input Form")

    # Input fields
    latitude = st.number_input("Latitude", format="%.6f", value = clicked_point.y)
    longitude = st.number_input("Longitude", format="%.6f", value = clicked_point.x)
    if click_data and 'last_active_drawing' in click_data and click_data['last_active_drawing']:
        macro_analysis = requests.get("https://apiavm.rhr.co.id/geo/get-administrative-area", params = {"lon" : longitude, "lat" : latitude}, verify = False) 
        st.spinner("Loading spatial analysis results...")
        if (macro_analysis.status_code ==200) : 
            st.subheader("Spatial Analysis Results : ")
            macro_analysis_response = macro_analysis.json() 
            # Extract POI name and distance pairs
            poi_data = []
                            
            # --- Group 1: Administrative Location ---
            with st.expander("📍 Administrative Location"):
                st.write(pd.DataFrame({
                    "Provinsi": macro_analysis_response["Provinsi"],
                    "Kota/Kabupaten": macro_analysis_response["Kota/Kabupaten"],
                    "Kecamatan": macro_analysis_response["Kecamatan"],
                    "Kelurahan/Desa": macro_analysis_response["Kelurahan/Desa"],
                }, index=[0])) 
            
        
            # --- Group 2: Amenity Within ---
            with st.expander("🏢 Amenity Within"):
                amenity_tab , transport_tab, demography_tabs , negative_tab = st.tabs(["🏢 Amenity Within", "🚉 Access to Public Transportation","Demography Analysis" , "⚠️ Negative Factor"])
                with amenity_tab:
                    amenity_data = {
                        "School": (macro_analysis_response["Nearest School"], macro_analysis_response["Distance to Nearest School (m)"]),
                        "Retail": (macro_analysis_response["Nearest Retail"], macro_analysis_response["Distance to Nearest Retail (m)"]),
                        "Hotel": (macro_analysis_response["Nearest Hotel"], macro_analysis_response["Distance to Nearest Hotel (m)"]),
                        "Restaurant": (macro_analysis_response["Nearest Restaurant"], macro_analysis_response["Distance to Nearest Restaurant (m)"]),
                        "Cafe/Resto": (macro_analysis_response["Nearest Cafe/Resto"], macro_analysis_response["Distance to Nearest Cafe/Resto (m)"]),
                        "Mall": (macro_analysis_response["Nearest Mall"], macro_analysis_response["Distance to Nearest Mall (m)"]),
                        "Government Institution": (macro_analysis_response["Nearest Government Institution"], macro_analysis_response["Distance to Nearest Government (m)"]),
                        "Convenience Store": (macro_analysis_response["Nearest Retail"], macro_analysis_response["Distance to Nearest Retail (m)"]),  # assuming same as Retail
                    }
                    df_amenities = pd.DataFrame([
                        {"Amenity": k, "Name": v[0], "Distance (m)": v[1]}
                        for k, v in amenity_data.items()
                    ])
                    st.dataframe(df_amenities)

                # --- Group 3: Access to Public Transportation ---
                with transport_tab:
                    transport_data = {
                        "Train Station": (macro_analysis_response["Nearest Train Station"], macro_analysis_response["Distance to Nearest Train Station (m)"]),
                        "Airport": (macro_analysis_response["Nearest Airport"], macro_analysis_response["Distance to Nearest Airport (m)"]),
                        "Bus Stop": (macro_analysis_response["Nearest Bus Stop"], macro_analysis_response["Distance to Nearest Bus Stop (m)"]),
                    }
                    df_transport = pd.DataFrame([
                        {"Transportation": k, "Name": v[0], "Distance (m)": v[1]}
                        for k, v in transport_data.items()
                    ])
                    st.dataframe(df_transport)

                # --- Group 4: Negative Factor ---

                with demography_tabs:
                    # sosio_data = sosio_analysis.json()["data"]
                    # # Define gender counts
                    # gender_counts = {
                    #     "Pria": sosio_data.get("pria", 0),
                    #     "Wanita": sosio_data.get("wanita", 0)
                    # }
                    # total_gender = sum(gender_counts.values())
                    # # Define new aggregated age categories
                    # age_group_map = {
                    #     "u18": ["u0", "u5", "u10", "u15"],
                    #     "u30": ["u20", "u25"],
                    #     "u50": ["u30", "u35", "u40", "u45"],
                    #     "u50plus": ["u50", "u55", "u60", "u65", "u70", "u75"]
                    # }
                    # # Build sunburst data
                    # sunburst_rows = []
                    # for gender, g_count in gender_counts.items():
                    #     ratio = g_count / total_gender if total_gender > 0 else 0
                    #     for group_name, group_keys in age_group_map.items():
                    #         group_total = sum([sosio_data.get(key, 0) or 0 for key in group_keys])
                    #         estimated = group_total * ratio
                    #         sunburst_rows.append({
                    #             "Gender": gender,
                    #             "Age Group": group_name,
                    #             "Value": estimated
                    #         })

                    # df = pd.DataFrame(sunburst_rows)
                    # # Create sunburst chart
                    # fig = px.sunburst(df, path=["Gender", "Age Group"], values="Value",
                    #                 color="Gender", title="Gender to Aggregated Age Group")
                    # fig.update_traces(textinfo='label+percent entry')
                    # st.plotly_chart(fig, use_container_width=True)
                    st.write("Demography Analysis is not available at the moment. Please check back later.")

                with negative_tab:
                    st.write({
                        "Nearest Cemetery": macro_analysis_response["Nearest Cemetery"],
                        "Distance to Nearest Cemetery (m)": macro_analysis_response["Distance to Nearest Cemetery (m)"]
                    })

        with st.expander("Detailed View for Selected Area") :
        # st.subheader("Detailed View for Selected Area")
            map_viz, bar_fig, land_price_kde, building_price_kde, yearly_price_development, surrounding_environment = visualize_poi_by_wkt(
                buffer_1k, engine
            )
            map_tab , bar_tab, price_tab, demand_tab = st.tabs(["Map visualization","📍 Access (POI Count)", "📊 Price Group", "💰 Demand Analysis"]) 
            with map_tab:
                if map_viz:
                    folium_static(map_viz, width=1400, height=500)
                else:
                    st.warning("No 1000m grid intersects with your selection.")

                # Access group: POI Count & Surrounding Environment 
        
            with bar_tab:
                cols = st.columns(2)
                with cols[0]:
                    if bar_fig:
                        st.plotly_chart(bar_fig, use_container_width=True)
                    else:
                        st.info("POI data is not available.")
                with cols[1]:
                    if surrounding_environment:
                        st.plotly_chart(surrounding_environment, use_container_width=True)
                    else:
                        st.info("Surrounding environment data is not available.")

            # Price group: Land Price KDE & Building Price KDE
            with price_tab:
                cols = st.columns(2)
                with cols[0]:
                    if land_price_kde:
                        st.plotly_chart(land_price_kde, use_container_width=True)
                    else:
                        st.info("Land price data is not available.")
                with cols[1]:
                    if building_price_kde:
                        st.plotly_chart(building_price_kde, use_container_width=True)
                    else:
                        st.info("Building price data is not available.")

            # Demand group: Yearly Median Price Development
            with demand_tab:
                if yearly_price_development:
                    st.plotly_chart(yearly_price_development, use_container_width=True)
                else:
                    st.info("Yearly price trend data is not available.")
    else:
        st.error(f"Please click point on the map. You can use the draw marker point tool to select a point.") 






        

