import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# Load in the iris Dataset
df_iris = sns.load_dataset("iris")

# Show First 5 Rows of the Dataset
df_iris.head()

# Create & Display the 3D Scatterplot
fig = px.scatter_3d(df_iris, 
              x="petal_length", 
              y="petal_width", 
              z="sepal_length", 
              size="sepal_width", 
              color="species",
              labels={"petal_length":"Petal Length (cm)", "petal_width":"Petal Width (cm)", "sepal_length":"Sepal Length (cm)"},
              color_discrete_map={"setosa": "blue", "versicolor":"deepskyblue", "virginica": "aqua"})

# Display graph via web application using streamlit
st.plotly_chart(fig, use_container_width=True)