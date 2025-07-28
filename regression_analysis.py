import streamlit as st
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from scipy.stats import anderson
import io
from helper import filter_dataframe, transform_dataframe

# Caching data to prevent reloading

with st.expander("Why do i even create this?") :
   st.write("Before jumping into complex machine learning models, our team usually starts with simple linear regression to understand the relationship between and instead of writing and running the same code over and over again i create this app to help us visualize the data and get the results quickly, wThis is a simple app, but it works for me, so i hope it works for you too :)")   
with st.expander("How to use this?") :
    st.write("1. Upload your file by clicking the 'Upload file' button.")
    st.write("2. Choose the independent and dependent variables from the left sidebar.")
    st.write("3. You can transform the data by clicking the 'Transform Data' button to transform the data into logarithmic or square root scale.") 
    st.write("4. You can filter the data by clicking the 'Filter Data' button.")
    st.write("5. Get the results by clicking the 'Start regression analysis' button.") 
@st.cache_data
def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

@st.cache_resource
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data[vif_data['feature'] != 'const']

def plot_regression_lines(X, y, data):
    st.write("Scatter plot between all dependent variables")
    X = [item for item in X if item in data.columns]
    num_features = len(X)
    num_columns = 4
    num_rows = int(np.ceil(num_features / num_columns))

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, feature in enumerate(X):
        ax = axes[i]
        sns.regplot(x=data[feature], y=data[y], ax=ax)
        ax.set_title(f'Regression: {y} vs {feature}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def regression_analysis(X, y, data):
    if data.isna().values.any():
        st.warning("Warning! Detected missing values. or values with infinity! Attempting to remove them...")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=[*X, y], inplace=True)
        st.write("Missing and infinity values removal successful. Current number of rows:", len(data))
    vif_data = calculate_vif(sm.add_constant(data[X]))
    
    reg_X = sm.add_constant(data[X])
    regression = sm.OLS(data[y], reg_X).fit()

    st.subheader("Regression Summary:")
    st.write(regression.summary2(alpha=0.1))

    st.subheader("VIF Data:")
    st.write(vif_data)

    y_pred = regression.predict(reg_X)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=data[y])
    sns.lineplot(x=y_pred, y=y_pred, color='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Linear Regression')
    st.pyplot(plt.gcf())
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=regression.resid)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    st.pyplot(plt.gcf())
    plt.clf()

    residuals = regression.resid
    ad_test = anderson(residuals, dist='norm')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(residuals, fill=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.title('Residuals Distribution')
    st.pyplot(plt.gcf())
    plt.clf()

    # plot_regression_lines(X, y, data) removed due to processing issue!!

    st.write("Anderson-Darling Test:")
    st.write(f"Test Statistic: {ad_test.statistic}")
    for critical_value, significance_level in zip(ad_test.critical_values, ad_test.significance_level):
        st.write(f"Critical Value at {significance_level}% significance level: {critical_value}")
    if ad_test.statistic > ad_test.critical_values[2]:  # 5% significance level
        st.write("Residuals are not normally distributed.")
    else:
        st.write("Residuals are normally distributed.")
    return regression

st.title('Regression Analysis Tool')
st.text('A simple linear regression analysis tool.')
st.subheader('How to use this app?')
st.text('1. Upload your file by clicking the "Upload file" button.')
st.text('2. Choose the independent and dependent variables from the left sidebar.')
st.text('3. Get the results by clicking the "Start regression analysis" button.')


is_dummy = st.radio("Use dummy data?", ["Yes", "No"]) 
if is_dummy == "No" : 
    uploaded = st.file_uploader("Please upload your Excel file", type=['xlsx'])
else : 
    uploaded = sns.load_dataset("penguins")

st.session_state['regression_button'] = None

def buttonclick() : 
    st.session_state['regression_button'] = True
    print(st.session_state['regression_button'])

if uploaded is not None:
    if not is_dummy :
        dataframe = load_data(uploaded)
    else : 
        dataframe = uploaded
    st.write("Great, here is the preview of your data.")
    st.write(dataframe.head(5))
    st.write(f"Number of rows: {len(dataframe)}")
    dataframe = filter_dataframe(dataframe)
    dataframe = transform_dataframe(dataframe)
    dataframe_model = dataframe.select_dtypes(include='number')
    st.sidebar.header("Regression Settings")
    independent_vars = st.sidebar.multiselect("Select independent variable(s) (X)", dataframe_model.columns)
    dependent_var = st.sidebar.selectbox("Select dependent variable (Y)", dataframe_model.columns)
    start_button = st.button("Start regression analysis", on_click=buttonclick())
    if  (start_button and independent_vars and dependent_var):
        st.text('If you like this app, kindly click "share" or "star" on my GitHub.')
        result = regression_analysis(independent_vars, dependent_var, dataframe)
        summary_str =  result.summary2().tables[1]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            summary_str.to_excel(writer, sheet_name = 'Regression Summary')
            dataframe.to_excel(writer, sheet_name='Dataset')
        st.download_button(
            label="Download Dataset Summary as Excel",
            data=output.getvalue(),
            file_name="regression_result_with_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

