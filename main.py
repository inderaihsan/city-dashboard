import geopandas as gpd
import pandas as pd
import streamlit as st
import pydeck as pdk
import json
import numpy as np
import plotly.express as px
from shapely import Polygon

# Set page config
st.set_page_config(page_title="Bogor Dalam Peta", layout="wide")

# Load data
@st.cache_data
def load_geojson():
    return gpd.read_file("data/mapid_data.geojson")

@st.cache_data
def load_poi(poi_type, kelurahan_list):
    file_path = f"data/{poi_type}_intersected.geojson"
    df = gpd.read_file(file_path)
    if 'DESA ATAU KELURAHAN' in df.columns:
        df = df[df['DESA ATAU KELURAHAN'].isin(kelurahan_list)]
    return df

gdf = load_geojson()

# Sidebar filters
st.title("Bogor Dalam Peta")

# Tabs
tab_map, tab_dashboard, tab_accessibility, tab_data = st.tabs(["Peta Interaktif", "Dashboard Statistik", "Analisa Akesibilitas", "Data Tabel"])

# Filter Logic
def filter_geodataframe(gdf):
    kecamatan = st.multiselect("Pilih Kecamatan", gdf['KECAMATAN'].unique())
    filter_kelurahan = st.checkbox("Filter berdasarkan kelurahan")
    
    if kecamatan:
        filtered_kelurahan_options = gdf[gdf['KECAMATAN'].isin(kecamatan)]['DESA ATAU KELURAHAN'].unique()
    else:
        filtered_kelurahan_options = gdf['DESA ATAU KELURAHAN'].unique()
    
    kelurahan = st.multiselect("Pilih Kelurahan", filtered_kelurahan_options)

    if kelurahan:
        filtered_gdf = gdf[gdf['DESA ATAU KELURAHAN'].isin(kelurahan)]
    elif kecamatan:
        filtered_gdf = gdf[gdf['KECAMATAN'].isin(kecamatan)]
    else:
        filtered_gdf = gdf.copy()
    
    return filtered_gdf

# === MAP TAB ===
with tab_map:
    st.subheader("Peta Wilayah Bogor")
    
    # Apply filtering
    filtered_gdf = filter_geodataframe(gdf)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Luas Wilayah (KM²)", f"{np.round(filtered_gdf['LUAS WILAYAH (KM²)'].sum(), 3)}")
    with col2:
        st.metric("Jumlah Penduduk", f"{int(filtered_gdf['JUMLAH PENDUDUK'].sum())}")

    radiolayer = st.radio("Layer Peta", ["Satelite", "Default"])

    # GeoJSON Layer
    filtered_gdf['kelurahan'] = filtered_gdf['DESA ATAU KELURAHAN']
    filtered_gdf['luas'] = filtered_gdf['LUAS WILAYAH (KM²)']
    filtered_gdf['kepadatan_penduduk'] = filtered_gdf['KEPADATAN PENDUDUK (JIWA/KM²)']

    centroid = filtered_gdf.unary_union.centroid
    lat, lon = centroid.y, centroid.x

    geojson_dict = json.loads(filtered_gdf.to_json())

    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_dict,
        stroked=True,
        filled=True,
        get_fill_color="[0, 150, 200, 100]",
        get_line_color=[255, 255, 255],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=14,
        pitch=0,
    )

    # POI Layer
    poi_files = {
        "school": "Sekolah",
        "bus_stop": "Halte",
        "graveyard": "Pemakaman",
        "hotel": "Hotel",
        "retail": "Ritel",
        "mall": "Mall",
        "hospital": "Rumah Sakit",
        "clinic": "Klinik / Faskes",
        "doctors": "Praktek Dokter",
    }

    icon_map = {
        "school": "https://img.icons8.com/?size=100&id=nnPbTRXFnQPK&format=png&color=000000 ",
        "bus_stop": "https://img.icons8.com/?size=100&id=bR0KARCUkfgL&format=png&color=000000  ",
        "graveyard": "https://img.icons8.com/?size=100&id=N0goJ2xHdDX7&format=png&color=000000  ",
        "hotel": "https://img.icons8.com/?size=100&id=sG7Ksu4yr5dZ&format=png&color=000000 ",
        "retail": "https://img.icons8.com/?size=100&id=3v9tw8RLozRM&format=png&color=000000 ",
        "mall": "https://img.icons8.com/?size=100&id=uUXumf3lrTEV&format=png&color=000000 ",
        "hospital": "https://img.icons8.com/?size=100&id=bzZ1vLaK5u1A&format=png&color=000000 ",
        "clinic": "https://img.icons8.com/?size=100&id=121193&format=png&color=000000",
        "doctors": "https://img.icons8.com/?size=100&id=PHpIFiaqVPHo&format=png&color=000000 "
    }

    # Checkbox untuk filter POI
    selected_poi_types = st.multiselect("Pilih Jenis POI untuk Ditampilkan", list(poi_files.keys()), default=list(poi_files.keys()))

    poi_dfs = []
    for poi_type in selected_poi_types:
        df = load_poi(poi_type, filtered_gdf['DESA ATAU KELURAHAN'].unique())
        df["type"] = poi_type
        poi_dfs.append(df)

    if poi_dfs:
        poi_combined = pd.concat(poi_dfs, ignore_index=True)
        poi_combined["longitude"] = poi_combined.geometry.centroid.x
        poi_combined["latitude"] = poi_combined.geometry.centroid.y

        # Tambahkan ikon
        poi_combined["icon_data"] = poi_combined["type"].apply(lambda x: {
            "url": icon_map[x],
            "width": 128,
            "height": 128,
            "anchorY": 128
        })

        icon_layer = pdk.Layer(
            type="IconLayer",
            data=poi_combined,
            get_icon="icon_data",
            get_size=2,
            size_scale=15,
            get_position='[longitude, latitude]',
            pickable=True
        )
    else:
        icon_layer = None

    layers = [geojson_layer]
    if icon_layer:
        layers.append(icon_layer)

    map_style = "mapbox://styles/mapbox/satellite-streets-v11" if radiolayer == "Satelite" else None

    tooltip = {
        "html": """
            <div>
                <strong>Kelurahan:</strong> {kelurahan} <br/>
                <strong>Luas Wilayah:</strong> {luas} km²<br/>
                <strong>Kepadatan:</strong> {kepadatan_penduduk} jiwa/km²<br/>
                <hr style='margin:4px 0;'/>
                <strong>Nama POI:</strong> {name}<br/>
                <strong>Tipe:</strong> {type}<br/>
            </div>
        """,
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.7)",
            "color": "white",
            "fontSize": "13px",
            "padding": "10px"
        }
    }

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=map_style
    )

    # Layout: Map + Legend + Chart
    col_map, col_side, col_poi = st.columns(spec=[0.7, 0.1,0.2], vertical_alignment="top")
    with col_map:
        st.pydeck_chart(r)

    with col_side:
        # Legenda
        st.markdown("""
        <style>
        .legend-container {
            background-color: rgba(0,0,0,0);
            padding: 10px;
            border-radius: 8px;
            width: 200px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .legend-icon {
            width: 24px;
            height: 24px;
            margin-right: 10px;
        }
        </style>
        <div class="legend-container">
            <strong>Legenda POI:</strong><br><br>
            """ +
            "".join([
                f"<div class='legend-item'><img class='legend-icon' src='{icon_map[key]}' />{value}</div>"
                for key, value in poi_files.items()
                if key in selected_poi_types
            ]) +
            """
        </div>
        """, unsafe_allow_html=True)

        # Bar chart jumlah POI per jenis
    with col_poi : 
        if poi_dfs and len(poi_combined) > 0:
            poi_count = poi_combined.groupby("type").size().reset_index(name='Jumlah')
            fig_poi_bar = px.bar(
                poi_count,
                x='type',
                y='Jumlah',
                title='Jumlah POI per Jenis',
                labels={'type': 'Jenis POI', 'Jumlah': 'Jumlah'},
                color='type',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_poi_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_poi_bar, use_container_width=True)

# === DASHBOARD TAB ===
with tab_dashboard:
    st.subheader("Informasi Sosio-Ekonomi dan Demografi")
    def prepare_data(gdf):
        pekerjaan_cols = [
            "BELUM/TIDAK BEKERJA", "NELAYAN", "PELAJAR DAN MAHASISWA",
            "PENSIUNAN", "PERDAGANGAN", "MENGURUS RUMAH TANGGA",
            "WIRASWASTA", "GURU", "PERAWAT", "PENGACARA", "PEKERJAAN LAINNYA"
        ]
        gdf[pekerjaan_cols] = gdf[pekerjaan_cols].apply(pd.to_numeric, errors='coerce')
        pekerjaan_data = gdf[pekerjaan_cols].sum().fillna(0).reset_index()
        pekerjaan_data.columns = ['Pekerjaan', 'Jumlah']

        usia_cols = [
            "USIA 0-4 TAHUN", "USIA 5-9 TAHUN", "USIA 10-14 TAHUN",
            "USIA 15-19 TAHUN", "USIA 20-24 TAHUN", "USIA 25-29 TAHUN",
            "USIA 30-34 TAHUN", "USIA 35-39 TAHUN", "USIA 40-44 TAHUN",
            "USIA 45-49 TAHUN", "USIA 50-54 TAHUN", "USIA 55-59 TAHUN",
            "USIA 60-64 TAHUN", "USIA 65-69 TAHUN", "USIA 70-74 TAHUN",
            "USIA 75 TAHUN KE ATAS"
        ]
        usia_data = gdf[usia_cols].sum().reset_index()
        usia_data.columns = ['Kelompok Usia', 'Jumlah']

        pertumbuhan_cols = [
            "PERTUMBUHAN PENDUDUK TAHUN 2020 (%)",
            "PERTUMBUHAN PENDUDUK TAHUN 2021 (%)",
            "PERTUMBUHAN PENDUDUK TAHUN 2022 (%)",
            "PERTUMBUHAN PENDUDUK TAHUN 2023 (%)",
            "PERTUMBUHAN PENDUDUK TAHUN 2024 (%)"
        ]
        pertumbuhan_data = gdf[pertumbuhan_cols].mean().dropna().reset_index()
        pertumbuhan_data.columns = ['Tahun', 'Pertumbuhan (%)']
        pertumbuhan_data['Tahun'] = pertumbuhan_data['Tahun'].str.extract(r'(\d{4})')

        return pekerjaan_data, usia_data, pertumbuhan_data

    pekerjaan_data, usia_data, pertumbuhan_data = prepare_data(filtered_gdf)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_pie = px.pie(pekerjaan_data, names='Pekerjaan', values='Jumlah',
                         title='Distribusi Penduduk Berdasarkan Pekerjaan')
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        fig_bar = px.bar(usia_data, x='Kelompok Usia', y='Jumlah',
                         title='Distribusi Penduduk Berdasarkan Kelompok Usia',
                         labels={'Jumlah': 'Jumlah Penduduk', 'Kelompok Usia': ''})
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col3:
        fig_line = px.line(pertumbuhan_data, x='Tahun', y='Pertumbuhan (%)',
                           markers=True, title='Rata-rata Pertumbuhan Penduduk Per Tahun (%)')
        fig_line.update_xaxes(tickmode='linear')
        st.plotly_chart(fig_line, use_container_width=True)

    # Additional Charts
    st.subheader("Informasi Demografi Tambahan")
    col4, col5, col6 = st.columns(3)
    with col4:
        status_cols = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"]
        status_data = gdf[status_cols].sum().reset_index()
        status_data.columns = ['Status Perkawinan', 'Jumlah']
        fig_status = px.pie(status_data, names='Status Perkawinan', values='Jumlah',
                            title='Distribusi Penduduk Berdasarkan Status Perkawinan')
        fig_status.update_traces(textinfo='percent+label')
        fig_status.update_layout(showlegend=False)
        st.plotly_chart(fig_status, use_container_width=True)
    with col5:
        pendidikan_cols = [
            "TIDAK/BELUM SEKOLAH", "BELUM TAMAT SD", "TAMAT SD",
            "SLTP", "SLTA", "D1 DAN D2", "D3", "S1", "S2", "S3"
        ]
        gdf[pendidikan_cols] = gdf[pendidikan_cols].apply(pd.to_numeric, errors='coerce')
        pendidikan_data = gdf[pendidikan_cols].sum().reset_index()
        pendidikan_data.columns = ['Jenjang Pendidikan', 'Jumlah']
        fig_pendidikan = px.bar(pendidikan_data, x='Jenjang Pendidikan', y='Jumlah',
                                title='Distribusi Penduduk Berdasarkan Jenjang Pendidikan',
                                labels={'Jumlah': 'Jumlah Penduduk'})
        fig_pendidikan.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pendidikan, use_container_width=True)
    with col6:
        jk_cols = ["LAKI-LAKI", "PEREMPUAN"]
        jk_data = gdf[jk_cols].sum().reset_index()
        jk_data.columns = ['Jenis Kelamin', 'Jumlah']
        fig_jk = px.pie(jk_data, names='Jenis Kelamin', values='Jumlah',
                        title='Distribusi Penduduk Berdasarkan Jenis Kelamin')
        fig_jk.update_traces(textinfo='percent+label')
        fig_jk.update_layout(showlegend=False)
        st.plotly_chart(fig_jk, use_container_width=True)

#===Aksesibilitas TAB=== 


# === ACCESSIBILITY TAB ===
with tab_accessibility:
    st.subheader("Analisis Kepadatan POI dengan Grid Heksagonal")

    if 'poi_combined' in locals() and not poi_combined.empty and not filtered_gdf.empty:
        import numpy as np
        from scipy.spatial.distance import cdist
        from shapely.geometry import Polygon
        import geopandas as gpd
        import plotly.express as px

        def create_hex_grid(gdf=None, bounds=None, n_cells=30, crs="EPSG:3857", buffer_size=50):
            if bounds is not None:
                xmin, ymin, xmax, ymax = bounds
            else:
                xmin, ymin, xmax, ymax = gdf.total_bounds

            width = (xmax - xmin) / n_cells
            height = np.sqrt(3) * width / 2

            xmin -= width
            xmax += width
            ymin -= height
            ymax += height

            hexagons = []
            col = 0
            x = xmin
            while x < xmax + width:
                y = ymin if col % 2 == 0 else ymin + height
                while y < ymax + height:
                    hexagon = Polygon([
                        (x, y),
                        (x + width / 2, y + height),
                        (x + 1.5 * width, y + height),
                        (x + 2.0 * width, y),
                        (x + 1.5 * width, y - height),
                        (x + width / 2, y - height),
                    ])
                    hexagons.append(hexagon)
                    y += 2 * height
                x += 1.5 * width
                col += 1

            grid = gpd.GeoDataFrame({'geometry': hexagons}, crs=crs)
            grid["grid_area"] = grid.area
            grid = grid.reset_index().rename(columns={"index": "grid_id"})

            if gdf is not None:
                grid = grid.clip(gdf)

            return grid

        # Konversi ke UTM (EPSG:3857)
        gdf_utm = filtered_gdf.to_crs("EPSG:3857")
        poi_utm = gpd.GeoDataFrame(geometry=poi_combined.geometry, crs="EPSG:4326").to_crs("EPSG:3857")

        # Buat grid
        total_bounds = gdf_utm.total_bounds
        width_meters = 500
        n_cells = int((total_bounds[2] - total_bounds[0]) / width_meters)

        hex_grid = create_hex_grid(gdf=gdf_utm, n_cells=n_cells, buffer_size=50)

        # Hitung centroid
        hex_grid["centroid_x"] = hex_grid.geometry.centroid.x
        hex_grid["centroid_y"] = hex_grid.geometry.centroid.y

        # Hitung jumlah POI per grid berdasarkan jarak <= 500m
        poi_utm['x'] = poi_utm.geometry.centroid.x
        poi_utm['y'] = poi_utm.geometry.centroid.y
        points = np.array([poi_utm['x'], poi_utm['y']]).T
        centroids = np.array([hex_grid['centroid_x'], hex_grid['centroid_y']]).T

        dist = cdist(centroids, points)
        hex_grid["poi_count"] = np.sum(dist <= 500, axis=1)

        # Kembali ke EPSG:4326 untuk ditampilkan di Mapbox
        hex_grid = hex_grid.to_crs(epsg=4326)
        hex_grid["lon"] = hex_grid.geometry.centroid.x
        hex_grid["lat"] = hex_grid.geometry.centroid.y

        # Plot choropleth dengan Plotly
        fig = px.choropleth_mapbox(
            hex_grid,
            geojson=hex_grid.__geo_interface__,
            locations=hex_grid.index,
            color="poi_count",
            hover_name="poi_count",
            color_continuous_scale="RdYlGn_r",
            range_color=(0, hex_grid["poi_count"].max()),
            mapbox_style="carto-positron",
            zoom=12,
            center={"lat": hex_grid["lat"].mean(), "lon": hex_grid["lon"].mean()},
            opacity=0.6,
            labels={'poi_count': 'Jumlah POI'}
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Data POI atau wilayah tidak tersedia.")


# === DATA TAB ===
with tab_data:
    st.subheader("Data Tabel")
    st.write("### Data Wilayah Kelurahan")
    st.dataframe(filtered_gdf)

    if poi_dfs:
        st.write("### Data POI")
        st.dataframe(poi_combined[['name', 'type', 'longitude', 'latitude']])
    else:
        st.info("Tidak ada POI yang dimuat.")
