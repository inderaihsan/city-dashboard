import geopandas as gpd
import pandas as pd
import streamlit as st
import pydeck as pdk
import json
import numpy as np

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Bogor Dalam Peta") 

# "JUMLAH PENDUDUK": 28391, "JUMLAH KK": 8767
# Load batas kelurahan
gdf = gpd.read_file("data/mapid_data.geojson")
kelurahan = st.multiselect("Pilih kelurahan", gdf['DESA ATAU KELURAHAN'].unique())

if kelurahan:
    gdf = gdf[gdf['DESA ATAU KELURAHAN'].isin(kelurahan)]
    col1, col2 = st.columns(2, vertical_alignment="top")

    with col1:
        st.subheader("Jumlah Penduduk")
        st.subheader(np.round(gdf['JUMLAH PENDUDUK'].sum(), 3))

    with col2:
        st.subheader("Kepadatan (jiwa/km²) ")
        st.subheader(np.round(gdf['KEPADATAN PENDUDUK (JIWA/KM²)'].sum() , 3)) 
        
    radiolayer = st.radio("Layer Map", ["Satelite", "None"])

    gdf['kelurahan'] = gdf['DESA ATAU KELURAHAN']
    gdf['luas'] = gdf['LUAS WILAYAH (KM²)']
    gdf['kepadatan_penduduk'] = gdf['KEPADATAN PENDUDUK (JIWA/KM²)']
    
    centroid = gdf.unary_union.centroid
    lat, lon = centroid.y, centroid.x

    geojson_dict = json.loads(gdf.to_json())
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

    # Load semua POI dan tandai jenisnya
    poi_files = {
        "school": "data/school_intersected.geojson",
        "bus_stop": "data/bus_stop_intersected.geojson",
        "graveyard": "data/graveyard_intersected.geojson",
        "hotel": "data/hotel_intersected.geojson",
        "retail": "data/retail_intersected.geojson", 
        "mall" : "data/mall_intersected.geojson", 
    }

    poi_dfs = []
    for poi_type, file_path in poi_files.items():
        df = gpd.read_file(file_path)
        df = df[df['DESA ATAU KELURAHAN'].isin(kelurahan)]
        df["type"] = poi_type
        poi_dfs.append(df)

    # Gabung semua
    poi_combined = pd.concat(poi_dfs, ignore_index=True) 
    
    poi_combined["longitude"] = poi_combined.geometry.centroid.x
    poi_combined["latitude"] = poi_combined.geometry.centroid.y

    # Mapping icon per POI
    icon_map = {
        "school": "https://img.icons8.com/?size=100&id=RWH5eUW9Vr7f&format=png&color=000000",
        "bus_stop": "https://img.icons8.com/?size=100&id=bR0KARCUkfgL&format=png&color=000000",
        "graveyard": "https://img.icons8.com/?size=100&id=N0goJ2xHdDX7&format=png&color=000000",
        "hotel": "https://cdn-icons-png.flaticon.com/512/235/235861.png",
        "retail": "https://img.icons8.com/?size=100&id=18901&format=png&color=000000", 
        "mall" : "https://img.icons8.com/?size=100&id=mlOtoDvNDS6o&format=png&color=000000"
    }

    poi_combined["icon_data"] = poi_combined["type"].apply(lambda x: {
        "url": icon_map[x],
        "width": 128,
        "height": 128,
        "anchorY": 128
    })

    # POI Icon Layer
    icon_layer = pdk.Layer(
        type="IconLayer",
        data=poi_combined,
        get_icon="icon_data",
        get_size=2,
        size_scale=15,
        get_position='[longitude, latitude]',
        pickable=True
    )

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

    map_style = "mapbox://styles/mapbox/satellite-streets-v11" if radiolayer=="Satelite" else None
    r = pdk.Deck(
        layers=[geojson_layer, icon_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=map_style
    )
    
    # Legend manual (bisa diatur di samping peta)
   

    col1, col2 = st.columns(spec =[0.9,0.1])

    with col1:
        st.pydeck_chart(r)

    with col2:
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
        <div class="legend-item">
            <img class="legend-icon" src="https://img.icons8.com/?size=100&id=RWH5eUW9Vr7f&format=png&color=000000" />
            Sekolah
        </div>
        <div class="legend-item">
            <img class="legend-icon" src="https://img.icons8.com/?size=100&id=bR0KARCUkfgL&format=png&color=000000" />
            Halte
        </div>
        <div class="legend-item">
            <img class="legend-icon" src="https://img.icons8.com/?size=100&id=N0goJ2xHdDX7&format=png&color=000000" />
            Pemakaman
        </div>
        <div class="legend-item">
            <img class="legend-icon" src="https://cdn-icons-png.flaticon.com/512/235/235861.png" />
            Hotel
        </div>
        <div class="legend-item">
            <img class="legend-icon" src="https://img.icons8.com/?size=100&id=18901&format=png&color=000000" />
            Ritel
        </div>
           <div class="legend-item">
            <img class="legend-icon" src="https://img.icons8.com/?size=100&id=mlOtoDvNDS6o&format=png&color=000000" />
            Mall
        </div>
    </div>
    """, unsafe_allow_html=True)
    import streamlit as st
    import plotly.express as px
    import pandas as pd

    # Asumsi gdf sudah dimuat sebagai DataFrame/GeoDataFrame
    # Contoh: gdf = pd.read_json('data.json') atau pd.read_csv(...) dsb.

 
    import plotly.express as px
    import pandas as pd

    # Asumsi gdf sudah dimuat sebagai DataFrame/GeoDataFrame
    # Contoh: gdf = pd.read_json('data.json') atau pd.read_csv(...) dsb.

    st.subheader("Informasi Sosio-Ekonomi dan Demografi")

    # --- Pemrosesan Data Pekerjaan ---
    pekerjaan_cols = [
        "BELUM/TIDAK BEKERJA", "NELAYAN", "PELAJAR DAN MAHASISWA",
        "PENSIUNAN", "PERDAGANGAN", "MENGURUS RUMAH TANGGA",
        "WIRASWASTA", "GURU", "PERAWAT", "PENGACARA", "PEKERJAAN LAINNYA"
    ]

    # Konversi ke numerik
    gdf[pekerjaan_cols] = gdf[pekerjaan_cols].apply(pd.to_numeric, errors='coerce')

    # Jumlahkan tiap kolom
    pekerjaan_data = gdf[pekerjaan_cols].sum().fillna(0).reset_index()
    pekerjaan_data.columns = ['Pekerjaan', 'Jumlah']
    pekerjaan_data = pekerjaan_data[pekerjaan_data['Jumlah'] > 0]  # Filter jika 0

    # --- Pemrosesan Data Usia ---
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

    # --- Pemrosesan Pertumbuhan Penduduk ---
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

    # --- Layout Kolom ---
    col1, col2, col3 = st.columns(3)

    # Pie Chart di Kolom 1: Distribusi Pekerjaan
    with col1:
        fig_pie = px.pie(
            pekerjaan_data,
            names='Pekerjaan',
            values='Jumlah',
            title='Distribusi Penduduk Berdasarkan Pekerjaan'
        )
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar Chart di Kolom 2: Distribusi Usia
    with col2:
        fig_bar = px.bar(
            usia_data,
            x='Kelompok Usia',
            y='Jumlah',
            title='Distribusi Penduduk Berdasarkan Kelompok Usia',
            labels={'Jumlah': 'Jumlah Penduduk', 'Kelompok Usia': ''}
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Line Chart di bawah kolom
    with col3 : 
        fig_line = px.line(
            pertumbuhan_data,
            x='Tahun',
            y='Pertumbuhan (%)',
            markers=True,
            title='Rata-rata Pertumbuhan Penduduk Per Tahun (%)'
        )
        fig_line.update_xaxes(tickmode='linear')
        st.plotly_chart(fig_line, use_container_width=True)
    # --- Bagian Tambahan: Status Perkawinan ---
    status_cols = ["BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI"]
    status_data = gdf[status_cols].sum().reset_index()
    status_data.columns = ['Status Perkawinan', 'Jumlah']

    fig_status = px.pie(
        status_data,
        names='Status Perkawinan',
        values='Jumlah',
        title='Distribusi Penduduk Berdasarkan Status Perkawinan',
    )
    fig_status.update_traces(textinfo='percent+label')
    fig_status.update_layout(showlegend=False)

    # --- Bagian Tambahan: Jenjang Pendidikan ---
    pendidikan_cols = [
        "TIDAK/BELUM SEKOLAH", "BELUM TAMAT SD", "TAMAT SD",
        "SLTP", "SLTA", "D1 DAN D2", "D3", "S1", "S2", "S3"
    ]
    gdf[pendidikan_cols] = gdf[pendidikan_cols].apply(pd.to_numeric, errors='coerce')
    pendidikan_data = gdf[pendidikan_cols].sum().reset_index()
    pendidikan_data.columns = ['Jenjang Pendidikan', 'Jumlah']
    pendidikan_data = pendidikan_data[pendidikan_data['Jumlah'] > 0]

    fig_pendidikan = px.bar(
        pendidikan_data,
        x='Jenjang Pendidikan',
        y='Jumlah',
        title='Distribusi Penduduk Berdasarkan Jenjang Pendidikan',
        labels={'Jumlah': 'Jumlah Penduduk'}
    )
    fig_pendidikan.update_xaxes(tickangle=45)

    # --- Bagian Tambahan: Jenis Kelamin ---
    jk_cols = ["LAKI-LAKI", "PEREMPUAN"]
    jk_data = gdf[jk_cols].sum().reset_index()
    jk_data.columns = ['Jenis Kelamin', 'Jumlah']

    fig_jk = px.pie(
        jk_data,
        names='Jenis Kelamin',
        values='Jumlah',
        title='Distribusi Penduduk Berdasarkan Jenis Kelamin'
    )
    fig_jk.update_traces(textinfo='percent+label')
    fig_jk.update_layout(showlegend=False)

    # --- Layout 3 Kolom Baru di Bawah ---
    st.subheader("Informasi Demografi Tambahan")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.plotly_chart(fig_status, use_container_width=True)

    with col5:
        st.plotly_chart(fig_pendidikan, use_container_width=True)

    with col6:
        st.plotly_chart(fig_jk, use_container_width=True) 
    