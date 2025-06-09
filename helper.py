from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import streamlit as st
import pandas as pd
import geopandas as gpd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



def remove_inf(data):
  data.replace([np.inf, -np.inf], np.nan, inplace=True)
  data.dropna(inplace=True, subset = 'geometry')
  return data

from shapely.geometry import Point

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    try: 
        modify = st.checkbox("Add filters")
        if not modify:
            return df
        df = df.copy()
        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()
        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    
                    user_min_input = right.number_input(
                        f"Lower bound for {column}", 
                        # min_value=_min, 
                        # max_value=_max, 
                        value=_min,
                        step=1.0
                    )
                    
                    user_max_input = right.number_input(
                        f"Upper bound for {column}", 
                        # min_value=_min, 
                        # max_value=_max, 
                        value=_max,
                        step=1.0
                    )
                    
                    df = df[df[column].between(user_min_input, user_max_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]
    except:
        st.warning("This column cannot be filtered, might be filled with missing values. To prevent this:")
        st.text("Ensure the column contains values.")
    return df


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers transform numeric columns
    (logarithm, inverse, or squared). The transformations create new columns
    rather than modifying the original column.
    
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: DataFrame with new transformed columns
    """
    modify = st.checkbox("Add transformations")
    if not modify:
        return df
    
    df = df.copy()

    # Create container for transformations
    transformation_container = st.container()
    with transformation_container:
        to_transform_columns = st.multiselect("Select columns to transform", df.columns)
        
        for column in to_transform_columns:
            if is_numeric_dtype(df[column]):
                transformation_type = st.selectbox(
                    f"Choose transformation for {column}",
                    ("None", "Logarithm", "Inverse", "Squared"),
                )
                transformation_name = {
                    'Logarithm' : 'ln',
                    'Inverse' : 'inv', 
                    'Squared' : 'sq', 
                    "None" : " "
                    
                }
                new_column_name = f"{transformation_name[transformation_type]}_{column}"

                if transformation_type == "Logarithm":
                    try:
                        df[new_column_name] = np.log(df[column])
                        st.write(f"Column '{new_column_name}' created.")
                    except ValueError:
                        st.warning(f"Cannot apply logarithm to column {column}. It may contain non-positive values.")
                
                elif transformation_type == "Inverse":
                    try:
                        df[new_column_name] = 1 / df[column]
                        st.write(f"Column '{new_column_name}' created.")
                    except ZeroDivisionError:
                        st.warning(f"Cannot apply inverse to column {column}. It may contain zeros.")
                
                elif transformation_type == "Squared":
                    df[new_column_name] = df[column] ** 2
                    st.write(f"Column '{new_column_name}' created.")
            else:
                st.warning(f"Column {column} is not numeric and cannot be transformed.")
    
    return df

def fsd_2(y_true, y_pred):
  y_true = np.exp(y_true)
  y_pred = np.exp(y_pred)
  pe = (y_true - y_pred)/y_true
  ape = np.abs(pe)
  return np.std(ape)

def evaluate(actual, predicted, squared = False, model = None):
    """
    Calculate various regression evaluation metrics, including FSD (Forecast Standard Deviation).

    Parameters:
    actual (array-like): The actual target values.
    predicted (array-like): The predicted target values.

    Returns:
    dict: A dictionary containing the calculated metrics.
    """
    if (squared == True):
        actual = np.exp(actual)
        predicted = np.exp(predicted)
    #calculate percentage error and absolute percentage error
    pe = ((actual-predicted)/actual)
    ape = np.abs(pe)
    n = len(actual)
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(actual - predicted))
    mae = mean_absolute_error(actual, predicted)
    # Calculate MSE (Mean Squared Error)
    mse = mean_squared_error(actual ,predicted)
    # Calculate R-squared (R2)
    r2 = r2_score(actual, predicted)
    # Calculate MAPE (Median Absolute Percentage Error)
    mape = np.median(ape)
    # Calculate FSD (Forecast Standard Deviation)
    fsd = np.std(ape)
    #pe10 and rt20 :
    r20 = [x for x in ape if x>=0.2]
    r10 = [x for x in ape if x<=0.1]
    rt20 = len(r20)/n
    pe10 = len(r10)/n
    # Create a dictionary to store the metrics
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'FSD': fsd,
        'PE10' : pe10,
        'RT20' : rt20
    }
    return metrics
@st.cache_data
def create_coordinate_2_dots(dataset , cola,colb) :
     geom = []
     #dataset.set_index('ID', inplace=True)
     dataset['ID'] = range(1, len(dataset)+1)
     dataset.set_index('ID', inplace = True)
     i=1
     for j,y in zip(dataset[cola],dataset[colb]) :
       try :
         if(pd.notna(j) & pd.notnull(y) & pd.notnull(j)&pd.notna(y)) :
            j = str(j)
            j = j.replace('°', ' ')
            j = j.replace(',', ' ')
            y = str(y)
            y = y.replace('°', ' ')
            y = y.replace(',', ' ')
            long = float(j)
            lat = float(y)
            geom.append(Point(lat, long))
            i=i+1
         else :
            geom.append(Point(-6.211694439526311, 106.82835921068619))
       except :
          dataset.drop(i, inplace = True)
          i=i+1
     dataset.apply(lambda col: col.drop_duplicates().reset_index(drop=True))
     gdf_tra = gpd.GeoDataFrame(dataset, geometry = geom)
     gdf_tra.set_crs(epsg = 4326, inplace = True) 
     gdf_tra.to_crs(epsg=32749, inplace = True)
     return gdf_tra

@st.cache_data
def transform_data_to_geodataframe(df, lat_col='Y', lon_col='X'):
    # Create a copy to avoid modifying the original dataframe
    dataset = df.copy()

    # Add an ID column if not present
    if 'ID' not in dataset.columns:
        dataset['ID'] = range(1, len(dataset) + 1)

    dataset.set_index('ID', inplace=True)

    def create_point(row):
        try:
            lat, lon = row[lat_col], row[lon_col]
            if pd.notna(lat) and pd.notna(lon):
                # Clean and convert coordinates
                lat = float(str(lat).replace('°', '').replace(',', '.'))
                lon = float(str(lon).replace('°', '').replace(',', '.'))
                return Point(lon, lat)
        except ValueError:
            pass
        # Default point if conversion fails
        return Point(-6.211694439526311, 106.82835921068619)

    # Create geometry column
    dataset['geometry'] = dataset.apply(create_point, axis=1)

    # Remove rows with default geometry
    dataset = dataset[dataset['geometry'] != Point(-6.211694439526311, 106.82835921068619)]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(dataset, geometry='geometry')

    # Set CRS and convert
    gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
    gdf.to_crs(epsg=32749, inplace=True)

    # Convert datetime columns to string
    timestamp_columns = gdf.select_dtypes(include=['datetime64']).columns
    gdf[timestamp_columns] = gdf[timestamp_columns].astype(str)

    return gdf

def clean_invalid_infinite_geometries(gdf):
    gdf_valid = gdf[gdf.is_valid]
    def has_infinite_or_nan(geometry):
        try:
            if geometry.is_empty:
                return True
            coords = np.array(geometry.coords)
            return np.any(np.isnan(coords)) or np.any(np.isinf(coords))
        except:
            return True 
    gdf_clean = gdf_valid[~gdf_valid.geometry.apply(has_infinite_or_nan)]

    return gdf_clean


def GovalMachineLearning(data, X, y, _algorithm) :
  algorithm = _algorithm
  columns = X
  columns.append(y)
#   if data[columns].isna().values.any():
  st.warning("Checking missing/infinity value and  attempting to remove them...")
  data.replace([np.inf, -np.inf], np.nan, inplace=True)
  data.dropna(subset=[*X, y], inplace=True)
  st.write("Missing and infinity values removal successful. Current number of rows:", len(data))
  if y in X :
    loading_bar = st.progress(value = 0, text = "Model training....")
    y_name = y
    X = [x for x in X if (x!=y)]
    data_model = data[columns].dropna()
    X = data_model[X]
    y = data_model[y]
    model = algorithm.fit(X,y)
    prediction_train = model.predict(X)
    data['prediction'] = prediction_train
    st.scatter_chart(
    data,
    x="prediction",
    y=y_name,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mod = algorithm.fit(X_train, y_train)
    prediction_test = mod.predict(X_test)
    test_score = evaluate(y_test, prediction_test, squared = True) 
    train_score = evaluate(y, prediction_train, squared = True) 
    st.session_state['test_score'] = test_score
    st.session_state['train_score'] = train_score
    i=0
    evaluation_result = {
        'R2' : [],
        'Fold' : [],
        'FSD' : [],
        'PE10' : [],
        'RT20' : []
    }
    kf = KFold(n_splits=10, shuffle = True, random_state = 404)
    for train_index, test_index in kf.split(X):
    #   st.write("Fold:", i+1)
      X_train = X.iloc[train_index, :]
      y_train = y.iloc[train_index]
      X_test = X.iloc[test_index]
      y_test = y.iloc[test_index]
      mod = model.fit(X_train, y_train)
      prediction = mod.predict(X_test)
      fold_result = (evaluate(y_test, prediction, squared=True))
      evaluation_result['Fold'].append(i)
      evaluation_result['R2'].append(fold_result['R2'])
      evaluation_result['FSD'].append(fold_result['FSD'])
      evaluation_result['PE10'].append(fold_result['PE10'])
      evaluation_result['RT20'].append(fold_result['RT20'])
      i = i+1
      loading_bar.progress(i*10, text='Model training in cross validation....')
      st.session_state['product'] = True 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Train score:")
        st.dataframe(pd.DataFrame(train_score, index = [0]).transpose())

    with col2:
        st.write("Test score (30%):")
        st.dataframe(pd.DataFrame(test_score, index = [0]).transpose())
    with col3 : 
        st.write("KFOLD evaluation:")
        st.dataframe(pd.DataFrame(evaluation_result))
  return [model, evaluation_result]


@st.cache_data
def load_map(map_obj) : 
    return map_obj.explore()


def get_local_url() : 
    local_url = "http://192.168.90.148:8000/" 
    return local_url


condition ='SERVER'
def get_server_url(condition='LOCAL') : 
    server_url = "http://47.129.216.241/" 
    if condition=='SERVER' :
        return server_url 
    else :
        # return "http://192.168.90.148:8000/" 
        # return "http://192.168.1.155:8000/" 
        return "http://127.0.0.1:8000/" 


