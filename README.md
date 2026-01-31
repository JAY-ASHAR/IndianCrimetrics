# IndianCrimetrics
📊 IndiaCrimetrics – Crime Analytics Dashboard

IndiaCrimetrics is a crime data analysis dashboard developed as a final-year BCA project.
The application analyzes Indian crime data using data preprocessing, clustering, and visual analytics to help understand crime trends across states and years.

Live Dashboard:
https://indiancrimetrics-dashboard.streamlit.app/

⸻

Project Objective

Crime data in India is available but scattered and difficult to interpret quickly.
This project aims to provide a single interactive platform to:
	•	Explore crime data by state, year, and crime type
	•	Identify crime patterns using clustering
	•	Visualize trends clearly for better understanding and comparison

⸻

Problem Statement
	•	Crime trends are hard to understand without interactive visuals
	•	Data is unorganized for quick analysis
	•	No unified dashboard to compare crime across states
	•	Difficult to allocate resources without regional crime insights

⸻

Proposed Solution

IndiaCrimetrics provides:
	•	An interactive Streamlit dashboard
	•	State-wise and year-wise crime analysis
	•	Crime severity grouping (high / medium / low) using clustering
	•	Visual summaries to support data-driven decisions

⸻

Key Features
	•	Upload crime data using Excel files
	•	Automatic data cleaning and preprocessing
	•	Filters for state, year, and crime type
	•	K-Means clustering with PCA visualization
	•	Interactive charts:
	•	Bar charts
	•	Line charts
	•	Pie charts
	•	Heatmaps
	•	Crime trend analysis over years
	•	Download cleaned and processed data as Excel

⸻

Dataset
	•	Input format: Excel (.xlsx)
	•	Contains crime records across Indian states and years
	•	Data is cleaned and structured automatically after upload

⸻

System Architecture (High Level)
	1.	User uploads Excel file
	2.	Data preprocessing (null handling, structuring)
	3.	K-Means clustering and PCA analysis
	4.	Interactive visualizations using Plotly
	5.	Crime trend analysis and severity grouping
	6.	Downloadable processed dataset

⸻

Technology Stack
	•	Frontend: Streamlit
	•	Data Processing: Pandas, NumPy
	•	Machine Learning: Scikit-learn (K-Means, PCA)
	•	Visualization: Plotly, Matplotlib
	•	File Handling: Excel
	•	Deployment: Streamlit Cloud
	•	IDE: VS Code

⸻

Testing Summary

All core features were tested, including:
	•	File upload
	•	Data preprocessing
	•	Clustering and PCA visualization
	•	Charts (bar, pie, line, heatmap)
	•	Excel download

All test cases passed successfully.

⸻

Future Enhancements
	•	Multi-language support (Indian regional languages)
	•	Advanced search functionality
	•	Real-time crime data integration
	•	Geo-mapping using maps

⸻

Conclusion

IndiaCrimetrics converts raw crime data into structured insights using data analysis and machine learning.
By combining clustering, PCA, and visual analytics, the dashboard helps identify crime patterns and trends across Indian states, supporting better understanding and analysis of crime data.
