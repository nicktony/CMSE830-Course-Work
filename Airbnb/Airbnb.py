#######################
##                   ##
##  Nickolaus White  ##
##  Midterm App      ##
##  CMSE 830         ##
##                   ##
#######################


######################
## Import Libraries ##
######################
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import matplotlib.pyplot as plt
import IPython
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import streamlit as st
from mpl_interactions import ioff, panhandler, zoom_factory
import altair as alt
import mplcursors
from matplotlib.patheffects import withSimplePatchShadow
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components

import warnings
warnings.filterwarnings('ignore')





#---------------------------------------------------------Start of streamlit program---------------------------------------------------------#





#################
## Import Data ##
#################
global df
df = pd.read_csv(r'C:\Users\white\Desktop\Airbnb\Airbnb_Open_Data.csv', low_memory=False)  


#######################
## Clean & Prep Data ##
#######################
# Check to see how many missing values there are
#msno.matrix(df);
# Drop columns with more than 50% missing values
df = df.drop(df.columns[df.isnull().mean() > 0.5], axis=1)
# Drop any rows that has at least 1 NaN value
df = df.dropna()
# Fix row in  neighbourhood group that has 'brookln' instead of 'brooklyn'
df['neighbourhood group'] = df['neighbourhood group'].replace('brookln', 'Brooklyn')
df['neighbourhood group'].value_counts()
# Drop the id and host id columns (won't need these)
df = df.drop(['id', 'host id'], axis=1)
# Convert 'last review' column to datetime type
df['last review'] = pd.to_datetime(df['last review'])
# Convert 'Construction year', 'minimum nights', 'availability 365' column to int type
list_of_columns = ['Construction year', 'review rate number', 'minimum nights', 'calculated host listings count', 'availability 365', 'number of reviews']
for column in list_of_columns:
    df[column] = df[column].astype(int)
# Fix the pricing format issues
def fix_price(price):
    return price.replace('$', '').replace(',', '')
df['price'] = df['price'].apply(fix_price)
df['price'] = df['price'].astype(float)
df['service fee'] = df['service fee'].apply(fix_price)
df['service fee'] = df['service fee'].astype(float)
# Convert `host_identity_verified` to categorical column then convert to binomial (0 & 1)
df['host_identity_verified'] = df['host_identity_verified'].astype('category')
df['host_identity_verified'] = df['host_identity_verified'].cat.codes
# Convert `instant_bookable` to 0 and 1 categorical column
df['instant_bookable'] = df['instant_bookable'].astype('category')
df['instant_bookable'] = df['instant_bookable'].cat.codes
# Drop columns where value counts is equal to 1
for col in df.columns:
    if df[col].value_counts().shape[0] == 1:
        df = df.drop(col, axis=1)
# Remove row where minimum nights = 5645, outliers
#df = df[df['minimum nights'] < 400]
# Remove row where availability 365 = 3677, outliers
#df = df[df['availability 365'] < 1000]
 
 
#######################
## Streamlit Sidebar ##
#######################
# Collapse on start
st.set_page_config(initial_sidebar_state="collapsed")
# Add title
st.sidebar.title("Borough NY Airbnb")
# Drop down menus
st.sidebar.write('Set Dropdown Filters')
axis_choice = st.sidebar.selectbox("Global Map Filter",('','price','service fee','review rate number'))
room_choice = st.sidebar.selectbox("Room Types",('','Private room','Entire home/apt','Shared room','Hotel room'))
policy_choice = st.sidebar.selectbox("Cancellation Policy",('','strict','moderate','flexible','nan'))
# Sliders
st.sidebar.write('Set Slide Filters')
values=[]
values = st.sidebar.slider('Price',float(df['price'].min()),float(df['price'].max()),(float(df['price'].min()),float(df['price'].max())))
priceLower = values[0] # Assign upper and lower values for filtering
priceUpper = values[1]
values = st.sidebar.slider('Service Fee',float(df['service fee'].min()),float(df['service fee'].max()),(float(df['service fee'].min()),float(df['service fee'].max())))
serviceLower = values[0] # Assign upper and lower values for filtering
serviceUpper = values[1]
values = st.sidebar.slider('Minimum Nights',0,int(df['minimum nights'].max()),(0,int(df['minimum nights'].max())))
nightsLower = values[0] # Assign upper and lower values for filtering
nightsUpper = values[1]
values = st.sidebar.slider('Review Rate Number',0,int(df['review rate number'].max()),(0,int(df['review rate number'].max())))
reviewLower = values[0] # Assign upper and lower values for filtering
reviewUpper = values[1]
values = st.sidebar.slider('Ability to Book x Number of Days Out',0.00,float(df['availability 365'].max()),(0.00,float(df['availability 365'].max())))
availLower = values[0] # Assign upper and lower values for filtering
availUpper = values[1]
# Checkboxes
hostVerified = st.sidebar.checkbox('Host Identity Verified')
instantBookable = st.sidebar.checkbox('Instant Bookable')
if hostVerified:
    df = df[df['host_identity_verified'] == 1]
if instantBookable:
    df = df[df['instant_bookable'] == 1]


####################################
## Filter DF Based on User Input  ##
####################################
df = df[df['price'] >= priceLower]
df = df[df['price'] <= priceUpper]
df = df[df['service fee'] >= serviceLower]
df = df[df['service fee'] <= serviceUpper]
df = df[df['minimum nights'] >= nightsLower]
df = df[df['minimum nights'] <= nightsUpper]
df = df[df['review rate number'] >= reviewLower]
df = df[df['review rate number'] <= reviewUpper]
df = df[df['availability 365'] >= availLower]
df = df[df['availability 365'] <= availUpper]
if room_choice != "": 
    df = df[df['room type'] == room_choice]
if policy_choice != "": 
    df = df[df['cancellation_policy'] == policy_choice]





#---------------------------------------------------------Functions---------------------------------------------------------#





def welcomePage():
    st.markdown("# Welcome to Nickolaus White's MSU CMSE830 Midterm Project 😄")
    st.markdown("Check out my code on [Github](https://github.com/nicktony).")

def heatMapPage():
    global df
    global axis_choice
    ################################
    ## Display Map of Borough NY  ##
    ################################
    if df.empty:
        st.markdown("# DataFrame is empty! Modify the filters to show data.")
    else: 
        # Import .shp file of Borough NY
        street_map = gpd.read_file(r'C:\Users\white\Desktop\Airbnb\Map\geo_export_336838c1-1789-4dd5-84ec-a3625918536c.shp')
        # Designate coordinate system
        crs = {'init':'epsg:4326'}
        # Zip x and y coordinates into single feature
        geometry = [Point(xy) for xy in zip(df['long'], df['lat'])]
        # Create GeoPandas dataframe
        geo_df = gpd.GeoDataFrame(df,crs=crs,geometry=geometry)
        # Create figure and axes, assign to subplot
        fig, ax = plt.subplots(figsize=(20,20))
        # Add .shp mapfile to axes
        street_map.plot(ax=ax,alpha=0.4,color='grey')
        if axis_choice != "":
            points = geo_df.plot(column=axis_choice,ax=ax,alpha=0.5,legend=True,markersize=100,colormap="GnBu")
        else: 
            points = geo_df.plot(ax=ax,alpha=0.5,legend=True,markersize=10)
        # Add title to graph
        plt.title('Borough NY Airbnb',fontsize=20,fontweight='bold')
        # Add annotation
        mplcursors.cursor(points,hover=True)
        # Show map
        st.pyplot(fig)
       
def dfOverviewPage():
    global df  
    global axis_choice    
    #################
    ## Display DF  ##
    #################
    # Plot interactive version of map
    alt.data_transformers.enable('default', max_rows=None)
    fig = alt.Chart(df).mark_point().encode(
        x='lat',
        y='long',
        color=alt.Color('price', scale=alt.Scale(scheme='greenblue')),
        tooltip=['NAME','host name','price']
    ).interactive()
    st.altair_chart(fig, use_container_width=True)
    # Format columns
    temp = df
    temp = temp.drop(columns=['lat','long','number of reviews','last review','reviews per month'])
    temp["host_identity_verified"] = temp["host_identity_verified"].astype(bool)
    temp["instant_bookable"] = temp["instant_bookable"].astype(bool)
    # Plot table
    st.write(temp)

def test():
    global df
    global axis_choice  





#---------------------------------------------------------Other---------------------------------------------------------#





#####################
## Page Functions  ##
#####################
page_names_to_funcs = {
    "Welcome Page": welcomePage,
    "Airbnb Heat Map": heatMapPage,
    "Dataframe Overview": dfOverviewPage,
}
selected_page = st.sidebar.selectbox("Select a Page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


#########
## CSS ##
#########
padding = 2
st.markdown(f""" <style>
    .css-hxt7ib{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
    
    
    
    
    