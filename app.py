import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="IndiaCrimetrics 🚔")

# --------------------------------------
# 🧼 Data Cleaning Function
# --------------------------------------
def preprocess_crime_data(df):
    df.columns = df.columns.str.strip()
    df = df.fillna(0)
    df.drop_duplicates(inplace=True)

    if 'YEAR' in df.columns:
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').fillna(0).astype(int)

    if 'TOTAL  CRIMES' not in df.columns:
        crime_columns = [col for col in df.columns if col not in ['STATE/UT', 'YEAR']]
        df['TOTAL  CRIMES'] = df[crime_columns].select_dtypes(include=np.number).sum(axis=1)

    return df

# --------------------------------------
# 🔄 Convert dataframe to Excel download
# --------------------------------------
def to_excel_download(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    return output.getvalue()

# --------------------------------------
# 📤 File Upload
# --------------------------------------
st.sidebar.title("📁 Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Choose a crime data Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    df = preprocess_crime_data(df_raw)

    st.title("🔍 IndiaCrimetrics")
    st.dataframe(df, use_container_width=True)  # 🔄 Show full DataFrame here

    # --------------------------------------
    # 🔎 Filters
    # --------------------------------------
    st.sidebar.title("🔎 Filters")
    years = sorted(df['YEAR'].unique())
    states = sorted(df['STATE/UT'].unique())
    crime_columns = [col for col in df.columns if col not in ['STATE/UT', 'YEAR', 'TOTAL  CRIMES', 'Cluster', 'PCA1', 'PCA2']]

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_state = st.sidebar.selectbox("Select State", states)
    search_crime = st.sidebar.text_input("🔍 Search Crime Type").lower()
    
    filtered_crimes = [crime for crime in crime_columns if search_crime in crime.lower()]
    selected_crimes = st.sidebar.multiselect("Select Crime Type(s)", filtered_crimes, default=filtered_crimes)

    filtered_df = df[(df['YEAR'] == selected_year) & (df['STATE/UT'] == selected_state)]

    # --------------------------------------
    # 📊 Crime Chart
    # --------------------------------------
    st.header(f"📈 Crime Summary for {selected_state} in {selected_year}")
    if not filtered_df.empty:
        crime_data = filtered_df[selected_crimes].select_dtypes(include=np.number).sum()
        fig_bar = px.bar(
            x=crime_data.index,
            y=crime_data.values,
            labels={'x': 'Crime Type', 'y': 'Total Incidents'},
            title='Crime Count by Selected Type',
            color=crime_data.index
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No data available for selected year and state.")

    # --------------------------------------
    # 🔗 Clustering
    # --------------------------------------
    st.header("🔗 Crime Clustering Analysis")
    numeric_cluster_data = df[crime_columns + ['TOTAL  CRIMES']].select_dtypes(include=np.number)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_cluster_data)

    # Elbow Curve
    st.subheader("📈 Elbow Curve for Optimal Clusters")
    distortions = []
    k_range = range(1, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_data)
        distortions.append(km.inertia_)

    fig_elbow = px.line(x=list(k_range), y=distortions, markers=True,
                        labels={'x': 'Number of Clusters', 'y': 'Inertia'},
                        title='Elbow Curve')
    st.plotly_chart(fig_elbow, use_container_width=True)

    # Clustering & PCA
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

    fig_cluster = px.scatter(
        df, x='PCA1', y='PCA2', color=df['Cluster'].astype(str),
        hover_data=['STATE/UT', 'YEAR'], title='Crime Pattern Clustering'
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # Cluster Distribution
    st.subheader("📊 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig_cluster_dist = px.pie(values=cluster_counts.values, names=cluster_counts.index.astype(str),
                              title="Distribution of Records Across Clusters")
    st.plotly_chart(fig_cluster_dist, use_container_width=True)

    # Crime Level Summary
    st.subheader("📘 Crime Level Summary")
    cluster_summary = df.groupby('Cluster')['TOTAL  CRIMES'].agg(['mean', 'sum', 'count']).reset_index()
    st.dataframe(cluster_summary, use_container_width=True)

    # Crime Trends Over the Years
    st.subheader("📈 Crime Trends Over the Years")
    yearly_trend = df.groupby('YEAR')['TOTAL  CRIMES'].sum().reset_index()
    fig_trend = px.line(yearly_trend, x='YEAR', y='TOTAL  CRIMES', markers=True,
                        title="Crime Trend Over Years")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Total IPC Crimes per Cluster
    st.subheader("🚨 Total IPC Crimes per Cluster")
    fig_ipc = px.bar(df.groupby('Cluster')['TOTAL  CRIMES'].sum().reset_index(),
                     x='Cluster', y='TOTAL  CRIMES', color='Cluster',
                     title="Total IPC Crimes in Each Cluster")
    st.plotly_chart(fig_ipc, use_container_width=True)

    # Most Common Crimes by State
    st.subheader("🔥 Most Common Crimes by State")
    numeric_crimes = df[crime_columns].select_dtypes(include=np.number).columns.tolist()
    state_crime_totals = df.groupby('STATE/UT')[numeric_crimes].sum()
    most_common_crimes = state_crime_totals.idxmax(axis=1).reset_index()
    most_common_crimes.columns = ['State/UT', 'Most Common Crime']
    st.dataframe(most_common_crimes, use_container_width=True)


    # Crime Level Breakdown (per Cluster)
    st.subheader("🔎 Crime Level Breakdown by Cluster")
    cluster_crime_avg = df.groupby('Cluster')[numeric_crimes].mean().T
    fig_heat = px.imshow(cluster_crime_avg,
                         labels=dict(x="Cluster", y="Crime Type", color="Avg Count"),
                         title="Average Crime Type per Cluster", aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Average Crime Rate by State
    st.subheader("🌐 Average Crime Rate by State")
    state_avg = df.groupby('STATE/UT')['TOTAL  CRIMES'].mean().reset_index()
    fig_avg_state = px.bar(state_avg, x='STATE/UT', y='TOTAL  CRIMES',
                           title="Average Crime Rate by State",
                           labels={'TOTAL  CRIMES': 'Average Crime'})
    st.plotly_chart(fig_avg_state, use_container_width=True)

    # --------------------------------------
    # 🔢 Top 10 States with Highest Crime in Selected Type
    # --------------------------------------
    st.header("🔢 Top 10 States with Highest Crime in Selected Type")

    top_crime_df = df[df['YEAR'] == selected_year]
    if not top_crime_df.empty and selected_crimes:
        top_states = (
            top_crime_df.groupby('STATE/UT')[selected_crimes]
            .sum()
            .sum(axis=1)
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_states.columns = ['STATE/UT', 'Total Crimes']
        fig_top_states = px.bar(top_states, x='STATE/UT', y='Total Crimes', color='STATE/UT',
                                title=f"Top 10 States with Highest '{', '.join(selected_crimes)}' in {selected_year}")
        st.plotly_chart(fig_top_states, use_container_width=True)
    else:
        st.warning("Not enough data to display Top 10 States chart.")
        
    # --------------------------------------
    # 📥 Cleaned Data Download
    # --------------------------------------
    st.header("📥 Download Cleaned Data")
    excel_data = to_excel_download(df)
    st.download_button(
        label="📤 Download Excel File",
        data=excel_data,
        file_name="Cleaned_Crime_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.title("📊 IndiaCrimetrics")
    st.info("Please upload a valid Excel file to begin.")
