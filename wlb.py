import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import streamlit as st


def visualize_percentage(df, aggregate_column, percentage_column, title):
  # 1️⃣ Get total respondents per industry
  total_counts = (
      df
      .groupby(aggregate_column)
      .size()
      .reset_index(name='Total_Respondents')
  )

  # 2️⃣ Get counts per issue per industry
  issue_cols = percentage_column

  issue_counts = (
      df
      .groupby(aggregate_column)[issue_cols]
      .sum()
      .reset_index()
  )

  # 3️⃣ Merge totals and compute percentages
  merged = pd.merge(issue_counts, total_counts, on=aggregate_column)

  # Compute percentages
  for col in issue_cols:
      issue = col.replace('is_', '').capitalize()
      merged[issue] = merged[col] / merged['Total_Respondents'] * 100

  # 4️⃣ Reshape to long format
  melted = merged.melt(
      id_vars=[aggregate_column],
      value_vars=[col.replace('is_', '').capitalize() for col in issue_cols],
      var_name='Issue',
      value_name='Percentage'
  )

  # 5️⃣ Plot grouped bar chart
  fig = px.bar(
      melted,
      x=aggregate_column,
      y='Percentage',
      color='Issue',
      barmode='group',
      text=melted['Percentage'].round(1).astype(str) + '%',
      title=title,
      color_discrete_sequence=px.colors.qualitative.Dark2
  )

  fig.update_traces(textposition='outside')

  fig.update_layout(
      xaxis_title=aggregate_column.replace("_", " "),
      yaxis_title='Percentage of Respondents',
      xaxis_tickangle=45,
      plot_bgcolor='white'
  )

  return fig



def create_one_hot_encoding(df, column_name):
    data = pd.Series(df[column_name])
    unique_issues = set()
    for entry in data.dropna():
        parts = [x.strip() for x in entry.split(';')]
        unique_issues.update(parts)
    print("Unique issues:", sorted(unique_issues))
    created_cols = []   # Store column names here
    for item in unique_issues:
        if item != 'Normal':
            col_name = 'is_' + '_'.join(word.lower() for word in item.split(' '))
            df[col_name] = df[column_name].str.contains(item, na=False).astype(int)
            created_cols.append(col_name)

    df[column_name + '_score'] = df[created_cols].sum(axis=1)

def plot_categorical_distribution(
    df,
    column,
    exclude_value=None,
    title=None,
    color_palette=px.colors.qualitative.Set3
):
    """
    Create a horizontal bar chart showing the distribution of a categorical column.

    Parameters:
    - df: DataFrame
    - column: column name (string)
    - exclude_value: optional value to exclude from the counts
    - title: optional title string
    - color_palette: list of colors to use
    """
    # Filter out the exclude_value if specified
    if exclude_value:
        data = df[df[column] != exclude_value][column]
    else:
        data = df[column]

    # Create count data
    counts = (
        data.value_counts()
        .reset_index()
        # .rename(columns={"index": column, column: "Count"})
    )

    # Use default title if not specified
    if title is None:
        title = f"Distribution of {column}"

    # Create figure
    fig = px.bar(
        counts,
        x="count",
        y=column,
        orientation="h",
        text="count",
        color=column,
        color_discrete_sequence=color_palette,
        title=title,
    )

    # Style tweaks
    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending"
        ),
        xaxis_title="Number of Respondents",
        yaxis_title=column.replace("_", " "),
        bargap=0.3,
        plot_bgcolor="white",
        showlegend=False
    )

    return fig


import pandas as pd
import plotly.express as px

def visualize_percentage(df, aggregate_column, percentage_column, title):
  # 1️⃣ Get total respondents per industry
  total_counts = (
      df
      .groupby(aggregate_column)
      .size()
      .reset_index(name='Total_Respondents')
  )

  # 2️⃣ Get counts per issue per industry
  issue_cols = percentage_column

  issue_counts = (
      df
      .groupby(aggregate_column)[issue_cols]
      .sum()
      .reset_index()
  )

  # 3️⃣ Merge totals and compute percentages
  merged = pd.merge(issue_counts, total_counts, on=aggregate_column)

  # Compute percentages
  for col in issue_cols:
      issue = col.replace('is_', '').capitalize()
      merged[issue] = merged[col] / merged['Total_Respondents'] * 100

  # 4️⃣ Reshape to long format
  melted = merged.melt(
      id_vars=[aggregate_column],
      value_vars=[col.replace('is_', '').capitalize() for col in issue_cols],
      var_name='Issue',
      value_name='Percentage'
  )

  # 5️⃣ Plot grouped bar chart
  fig = px.bar(
      melted,
      x=aggregate_column,
      y='Percentage',
      color='Issue',
      barmode='group',
      text=melted['Percentage'].round(1).astype(str) + '%',
      title=title,
      color_discrete_sequence=px.colors.qualitative.Dark2
  )

  fig.update_traces(textposition='outside')

  fig.update_layout(
      xaxis_title=aggregate_column.replace("_", " "),
      yaxis_title='Percentage of Respondents',
      xaxis_tickangle=45,
      plot_bgcolor='white'
  )

  return fig



df = pd.read_csv("data/post_pandemic_remote_work_health_impact_2025.csv")
df.head(5)
create_one_hot_encoding(df, 'Physical_Health_Issues')
create_one_hot_encoding(df, 'Mental_Health_Status')
df['n_Burnout_Level'] = df['Burnout_Level'].apply(lambda x:1 if x=='Low' else 2 if x=='Medium' else 3)
salary_mapping = {
    '$40K-60K': 1,
    '$60K-80K': 2,
    '$80K-100K': 3,
    '$100K-120K': 4,
    '$120K+': 5
}
df['n_Salary_Range'] = df['Salary_Range'].apply(lambda x: salary_mapping.get(x, None))

# print(df['Salary_Range'].unique())
st.markdown("""In this section, I will try to extract as many information as possible from the data. The Post-Pandemic Remote Work Health Impact 2025 dataset presents a comprehensive, global snapshot of how remote, hybrid, and onsite work arrangements are influencing the mental and physical health of employees in the post-pandemic era. Collected in June 2025, this dataset aggregates responses from a diverse workforce spanning continents, industries, age groups, and job roles. It is designed to support research, data analysis, and policy-making around the evolving landscape of work and well-being.


for more details about the data kindly visit this link : https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025/data
            """, unsafe_allow_html=True) 


with st.expander("Helper Functions", expanded=False):
    st.code("""def create_one_hot_encoding(df, column_name):
    data = pd.Series(df[column_name])
    unique_issues = set()
    for entry in data.dropna():
        parts = [x.strip() for x in entry.split(';')]
        unique_issues.update(parts)
    print("Unique issues:", sorted(unique_issues))
    created_cols = []   # Store column names here
    for item in unique_issues:
        if item != 'Normal':
            col_name = 'is_' + '_'.join(word.lower() for word in item.split(' '))
            df[col_name] = df[column_name].str.contains(item, na=False).astype(int)
            created_cols.append(col_name)

    df[column_name + '_score'] = df[created_cols].sum(axis=1)""", language='python')



with st.expander('Statistics of the dataset', expanded=False) : 
    st.write(df.describe().transpose()) 

st.header("Univariate Analysis")
with st.expander("Physical Health Issues Distribution", expanded=True):
    st.plotly_chart(plot_categorical_distribution(df, 'Mental_Health_Status', 'Normal', 'Mental Health Status Distribution'))
    st.write("""Among the 3,157 respondents, PTSD emerged as the most frequently reported mental health issue, accounting for 423 cases (approximately 13% of the sample). This was followed by high rates of anxiety and burnout, which also affected a substantial proportion of respondents.""")



with st.expander("Participant region distribution", expanded=True):
    st.plotly_chart(plot_categorical_distribution(df, 'Region'))
    st.write("""The dataset includes a diverse range of regions, with North America and Europe being the most represented. This diversity allows for a more comprehensive understanding of how remote work impacts mental health across different cultural contexts.""") 

with st.expander("Salary Distribution", expanded=True):
    st.plotly_chart(plot_categorical_distribution(df, 'Salary_Range'))
    st.write("""The Respondent seems to be having a salary ranging from 40K USD to more than 120k USD per year. with the majority of respondents eaerning between 60K and 100K USD""") 

with st.expander("Industry Distribution", expanded=True):
    st.plotly_chart(plot_categorical_distribution(df, 'Industry'))
    st.write(""".""") 

st.header("Bivariate Analysis")
with st.expander("Mental Health Issues by Industry", expanded=True):
    st.plotly_chart(visualize_percentage(df, 'Industry', ['is_anxiety', 'is_burnout', 'is_depression', 'is_ptsd' ], 'Percentage of Mental Health Issues by Job Role'))
    st.write("""The analysis reveals that there are some notable mental health issues that are more prevalent in certain industries. For example, the tech has a higher percentage of ptsd. and the similar phenomenon is observed in the marketing and manufacturing industry with burnout and finance with depression. We can conclude that PTSD, Burnout and Depression are the most common mental health issues across all industries, with Anxiety being slightly less prevalent.""")

with st.expander("Mental Health Issues by Region", expanded=True):
    st.plotly_chart(visualize_percentage(df, 'Region', ['is_anxiety', 'is_burnout', 'is_depression', 'is_ptsd'], 'Percentage of Mental Health Issues by Region'))
    st.write("""the chart shows that the prevalence of these mental health issues varies by region, with some regions experiencing higher percentages of certain issues than others. For example, Asia shows the highest percentages for Burnout and PTSD, while North America has the highest percentage for Depression.""") 

with st.expander("Burnout Level by Region and Industry") : 
    agg_df = (
    df
    .groupby(['Region', 'Industry'])
    .agg(Mean_Burnout=('n_Burnout_Level', 'mean'))
    .reset_index()
    )
    fig = px.treemap(
        agg_df,
        path=[px.Constant("all"),'Region', 'Industry',],
        values='Mean_Burnout',
        color='Mean_Burnout', 
        color_continuous_scale='RdBu_r',
    )

 
    st.plotly_chart(fig)
    st.write("""Customer Service frequently appears in darker red in several continents (e.g., Oceania, Europe), indicating higher mean burnout in those regions.
    Education is consistently lighter or blue-toned in many regions (South America, North America, Europe), suggesting lower mean burnout where it is visible.
    Technology varies: darker red in Asia (high burnout), moderate in South America, and lighter elsewhere.
    Retail shows notable variation, from dark blue (low burnout in Asia and North America) to pale shades elsewhere.
    The color spectrum across continents shows that burnout levels vary both by geography and by industry.""")



with st.expander("Work hour arrangement by region") : 

    agg_df = (
        df
        .groupby(['Region', 'Work_Arrangement'])
        .agg(median_working_hour_per_week=('Hours_Per_Week', 'median'))
        .reset_index()
        )
    fig = px.treemap(
        agg_df,
        path=[px.Constant("all"),'Region', 'Work_Arrangement',],
        values='median_working_hour_per_week',
        color='median_working_hour_per_week', 
        color_continuous_scale='RdBu_r',
    )

    st.plotly_chart(fig)
    st.write("""aaa""")


with st.expander("Burnout Level by work arrangement") : 

    agg_df = (
        df
        .groupby(['Work_Arrangement'])
        .agg(burnout_level_median=('n_Burnout_Level', 'mean'))
        .reset_index()
        )
    fig = px.treemap(
        agg_df,
        path=[px.Constant("all"), 'Work_Arrangement',],
        values='burnout_level_median',
        color='burnout_level_median', 
        color_continuous_scale='RdBu_r',
    ) 

    st.dataframe(agg_df, use_container_width=True)
    st.write("I was surprised to see that the median burnout level for remote workers is higher than that of hybrid workers, which is higher than that of onsite workers. This suggests that remote work may be more stressful than hybrid or onsite work, at least in terms of burnout levels.")

    st.plotly_chart(fig)

    c_df = (
        df
        .groupby(['Work_Arrangement'])
        .agg(self_isolation_score_mean=('Social_Isolation_Score', 'mean'))
        .reset_index()
        )
    fig = px.treemap(
        c_df,
        path=[px.Constant("all"), 'Work_Arrangement',],
        values='self_isolation_score_mean',
        color='self_isolation_score_mean', 
        color_continuous_scale='RdBu_r',
        hover_data={'Work_Arrangement': True, 'self_isolation_score_mean': True}
        
    ) 

    st.write("So the mean self isolation score for remote workers is the highest. this might be due to the fact that remote workers are more likely to work from home, which can lead to feelings of isolation and loneliness. Hybrid workers have a lower mean self isolation score, which suggests that they may have more opportunities for social interaction than remote workers. Onsite workers have the lowest mean self isolation score, which suggests that they may have the most opportunities for social interaction.")


    st.plotly_chart(fig) 

    

