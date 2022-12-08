#######################
##                   ##
##  Nickolaus White  ##
##  Final App        ##
##  CMSE 830         ##
##                   ##
#######################


######################
## Import Libraries ##
######################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')





#---------------------------------------------------------Start of streamlit program---------------------------------------------------------#





#######################
## Streamlit Sidebar ##
#######################
# Collapse on start
st.set_page_config(initial_sidebar_state="collapsed")
# Add title
st.sidebar.title("Hapiness Datasets")


#################
## Import Data ##
#################
global df15, df16, df17, df18, df19, df20, df21, df22
df15 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2015.csv')
df16 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2016.csv')
df17 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2017.csv')
df18 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2018.csv')
df19 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2019.csv')
df20 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2020.csv')
df21 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2021.csv')
df22 = pd.read_csv(r'C:\Users\white\Desktop\CMSE830\Final\Data\2022.csv')


#######################
## Clean & Prep Data ##
#######################
# 2015 -----------------------------------------
# Drop Happiness Rank
df15 = df15.drop(columns=['Happiness Rank'])

# Replace Region values with numerical values
df15.Region = pd.Categorical(df15.Region)
df15['temp'] = df15.Region.cat.codes
df15['Region'] = df15.Region.astype('category').cat.codes
df15.head()

# Replace Country values with numerical values
df15.Country = pd.Categorical(df15.Country)
df15['temp'] = df15.Country.cat.codes
df15['Country'] = df15.Country.astype('category').cat.codes

df15 = df15.drop(columns=['temp'])
#df15.head()

# 2016 -----------------------------------------
# Drop Happiness Rank & Intervals
df16 = df16.drop(columns=['Happiness Rank'])
df16 = df16.drop(columns=['Lower Confidence Interval'])
df16 = df16.drop(columns=['Upper Confidence Interval'])

# Replace Region values with numerical values
df16.Region = pd.Categorical(df16.Region)
df16['temp'] = df16.Region.cat.codes
df16['Region'] = df16.Region.astype('category').cat.codes

# Replace Country values with numerical values
df16.Country = pd.Categorical(df16.Country)
df16['temp'] = df16.Country.cat.codes
df16['Country'] = df16.Country.astype('category').cat.codes

df16 = df16.drop(columns=['temp'])
#df16.head()

# 2017 -----------------------------------------
# Drop Happiness Rank & Intervals
df17 = df17.drop(columns=['Happiness.Rank'])
df17 = df17.drop(columns=['Whisker.high'])
df17 = df17.drop(columns=['Whisker.low'])

# Rename main columns
df17 = df17.rename(columns={"Happiness.Score": "Happiness Score"})

# Replace Country values with numerical values
df17.Country = pd.Categorical(df17.Country)
df17['temp'] = df17.Country.cat.codes
df17['Country'] = df17.Country.astype('category').cat.codes

df17 = df17.drop(columns=['temp'])
#df17.head()

# 2018 -----------------------------------------
# Drop Overall Rank
df18 = df18.drop(columns=['Overall rank'])

# Rename main columns
df18 = df18.rename(columns={"Score": "Happiness Score", "Country or region": "Country"})

# Replace Country values with numerical values
df18['Country'] = pd.Categorical(df18['Country'])
df18['temp'] = df18['Country'].cat.codes
df18['Country'] = df18['Country'].astype('category').cat.codes

df18 = df18.drop(columns=['temp'])
#df18.head()

# 2019 -----------------------------------------
# Drop Overall Rank
df19 = df19.drop(columns=['Overall rank'])

# Rename main columns
df19 = df19.rename(columns={"Score": "Happiness Score", "Country or region": "Country"})

# Replace Country values with numerical values
df19['Country'] = pd.Categorical(df19['Country'])
df19['temp'] = df19['Country'].cat.codes
df19['Country'] = df19['Country'].astype('category').cat.codes

df19 = df19.drop(columns=['temp'])
#df19.head()

# 2020 -----------------------------------------
# Drop Intervals
df20 = df20.drop(columns=['upperwhisker'])
df20 = df20.drop(columns=['lowerwhisker'])

# Rename main columns
df20 = df20.rename(columns={"Ladder score": "Happiness Score", "Country name": "Country"})

# Replace Region values with numerical values
df20['Regional indicator'] = pd.Categorical(df20['Regional indicator'])
df20['temp'] = df20['Regional indicator'].cat.codes
df20['Regional indicator'] = df20['Regional indicator'].astype('category').cat.codes

# Replace Country values with numerical values
df20['Country'] = pd.Categorical(df20['Country'])
df20['temp'] = df20['Country'].cat.codes
df20['Country'] = df20['Country'].astype('category').cat.codes

df20 = df20.drop(columns=['temp'])
#df20.head()

# 2021 -----------------------------------------
# Drop Intervals
df21 = df21.drop(columns=['upperwhisker'])
df21 = df21.drop(columns=['lowerwhisker'])

# Rename main columns
df21 = df21.rename(columns={"Ladder score": "Happiness Score", "Country name": 'Country'})

# Replace Region values with numerical values
df21['Regional indicator'] = pd.Categorical(df21['Regional indicator'])
df21['temp'] = df21['Regional indicator'].cat.codes
df21['Regional indicator'] = df21['Regional indicator'].astype('category').cat.codes

# Replace Country values with numerical values
df21['Country'] = pd.Categorical(df21['Country'])
df21['temp'] = df21['Country'].cat.codes
df21['Country'] = df21['Country'].astype('category').cat.codes

df21 = df21.drop(columns=['temp'])
#df21.head()

# 202 -----------------------------------------
# Rename main columns
df22 = df22.rename(columns={"Happiness score": "Happiness Score"})

# Change commas to periods in numerical data
df22['Happiness Score']=df22['Happiness Score'].str.replace(',','.')
df22['Whisker-high']=df22['Whisker-high'].str.replace(',','.')
df22['Whisker-low']=df22['Whisker-low'].str.replace(',','.')
df22['Dystopia (1.83) + residual']=df22['Dystopia (1.83) + residual'].str.replace(',','.')
df22['Explained by: GDP per capita']=df22['Explained by: GDP per capita'].str.replace(',','.')
df22['Explained by: Social support']=df22['Explained by: Social support'].str.replace(',','.')
df22['Explained by: Healthy life expectancy']=df22['Explained by: Healthy life expectancy'].str.replace(',','.')
df22['Explained by: Freedom to make life choices']=df22['Explained by: Freedom to make life choices'].str.replace(',','.')
df22['Explained by: Generosity']=df22['Explained by: Generosity'].str.replace(',','.')
df22['Explained by: Perceptions of corruption']=df22['Explained by: Perceptions of corruption'].str.replace(',','.')

# Replace Country values with numerical values
df22.Country = pd.Categorical(df22.Country)
df22['temp'] = df22.Country.cat.codes
df22['Country'] = df22.Country.astype('category').cat.codes

# Drop tables/ last variable in Country column
df22 = df22.drop(columns=['temp'])
df22 = df22.drop(columns=['RANK'])

# Drop last row full of NaN values
df22.drop(df22.tail(1).index, inplace=True)

# Display changes
#df22.head()





#---------------------------------------------------------Functions---------------------------------------------------------#





##################
## Welcome Page ##
##################
def welcomePage():
    st.markdown("# Welcome to Nickolaus White's MSU CMSE830 Final Project 😄")
    st.markdown("Check out my code on [Github](https://github.com/nicktony/CMSE830).")

####################
## Mean Happiness ##
#################### 
def meanPage():
    global df15, df16, df17, df18, df19, df20, df21, df22
    
    st.write("Mean Happiness Score/yr with Vertically Plotted Happiness Scores")
    
    # Initiate figure
    fig, ax = plt.subplots()
    
    # Plot Line of Means
    x = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    y = [df15["Happiness Score"].mean(),
         df16["Happiness Score"].mean(),
         df17["Happiness Score"].mean(),
         df18["Happiness Score"].mean(),
         df19["Happiness Score"].mean(), 
         df20["Happiness Score"].mean(), 
         df21["Happiness Score"].mean()]
    ax.plot(x, y, 'r')
    #plt.xlabel('Year', fontweight ='bold', fontsize = 15)
    #plt.ylabel('Mean Happiness', fontweight ='bold', fontsize = 15)
    
    # Plot scatter of Happiness Scores on y-axis
    temp1 = list(range(0,len(df15["Happiness Score"]),1))
    for i in temp1:
        temp1[i] = 2015  
    temp2 = list(range(0,len(df16["Happiness Score"]),1))
    for i in temp2:
        temp2[i] = 2016 
    temp3 = list(range(0,len(df17["Happiness Score"]),1))
    for i in temp3:
        temp3[i] = 2017 
    temp4 = list(range(0,len(df18["Happiness Score"]),1))
    for i in temp4:
        temp4[i] = 2018 
    temp5 = list(range(0,len(df19["Happiness Score"]),1))
    for i in temp5:
        temp5[i] = 2019     
    temp6 = list(range(0,len(df20["Happiness Score"]),1))
    for i in temp6:
        temp6[i] = 2020  
    temp7 = list(range(0,len(df21["Happiness Score"]),1))
    for i in temp7:
        temp7[i] = 2021  
    x = np.array([temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7])
    y = df15["Happiness Score"].append(df16["Happiness Score"])
    y = y.append(df17["Happiness Score"])
    y = y.append(df18["Happiness Score"])
    y = y.append(df19["Happiness Score"])
    y = y.append(df20["Happiness Score"])
    y = y.append(df21["Happiness Score"])
    y = np.array(y)
    y = np.reshape(y, (1, 1084))
    ax.scatter(x, y, s = 0.25, color='black')       
        
    # Display graph
    st.pyplot(fig)
    
    # Display table with Happines Score values for each year
    st.write("Years & Happiness Scores")
    st.write(np.concatenate((x, y)))


####################
## Classification ##
####################  
def classifierPage(): 
    
    # Sliders
    st.sidebar.write('Set Slider Filters')
    featureNum = st.sidebar.slider('Number of Columns for Feature Selection',1,4,1)
    # Checkboxes
    st.sidebar.write('Set Checkbox Filters')
    scaledData = st.sidebar.checkbox('Scale Data')
    
    # Create object to hold all datasets
    datasets = [
        df15,
        df16,
        df17,
        df18,
        df19,
        df20,
        df21,
        #df22
    ]
    
    # Create object to hold classifier names
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
    ]
    
    # Create object to hold classifiers
    classifiers = [
        KNeighborsClassifier(),
        svm.SVC(kernel="linear", C=0.025),
        svm.SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
    ]
    
    # Create object to store accuracy's
    accuracy = [[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], 
                [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], 
                [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], 
                [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], 
                [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
                [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
                [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]]
    
    # Title
    st.write("Classification Accuracy Bar Graph")
    st.write("Classifiers From Left to Right: ")
    st.write("- Nearest Neighbors (Red) - Linear SVM (Orange) - RBF SVM (Yellow)")
    st.write("- Gaussian Process (Blue) - Decision Tree (Cyan) - Random Forest (Green)")
    st.write("- Neural Net (Pink) - Adaboost (Violet) - Naive Bayes (Purple")
    
    # Loop through datasets
    for num1, df in enumerate(datasets):
        
        # Feature engineering due to bad scores
        # Grab highest correlation values
        corr = df.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")
        
        # Assign X based on number of features
        if (featureNum == 1):
            temp = temp.head(1)
            temp = temp.to_frame()
            columns = list(("",))
            i = 0
            for index in temp.index:
                columns[i] = index
                i = i + 1    
            # Assign X variable
            X = df[[columns[0]]].to_numpy()        
        if (featureNum == 2):
            temp = temp.head(2)
            temp = temp.to_frame()
            columns = list(("",""))  
            i = 0
            for index in temp.index:
                columns[i] = index
                i = i + 1      
            # Assign X variable
            X = df[[columns[0],columns[1]]].to_numpy()        
        if (featureNum == 3):
            temp = temp.head(3)
            temp = temp.to_frame()
            columns = list(("","","")) 
            i = 0
            for index in temp.index:
                columns[i] = index
                i = i + 1 
            # Assign X variable
            X = df[[columns[0],columns[1],columns[2]]].to_numpy()        
        if (featureNum == 4):
            temp = temp.head(4)
            temp = temp.to_frame()
            columns = list(("","","",""))  
            i = 0
            for index in temp.index:
                columns[i] = index
                i = i + 1     
            # Assign X variable
            X = df[[columns[0],columns[1],columns[2],columns[3]]].to_numpy()
        
        # Assign y variable
        y = df["Happiness Score"].to_numpy()
        y = np.around(y)
    
        # Loop through classifiers
        num2 = 0
        for name, clf in zip(names, classifiers):
            
            # Percentage of test sample size
            test_fraction = 0.3
    
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)
            
            # Scaler is instantiated
            my_scaler = StandardScaler()
    
            # Scaler learns from the data how to perform the scaling
            my_scaler.fit(X_train)
    
            if scaledData:     
                # Instantiate the training set & scale it
                X_train_scaled = my_scaler.transform(X_train)
                # Instantiate the testing set & scale it
                X_test_scaled = my_scaler.transform(X_test)
                # Assign classifier
                my_classifier = classifiers[num2]
                # Apply classifier
                my_classifier.fit(X_train_scaled, y_train)
                # Creating the test
                my_predictions = my_classifier.predict(X_test_scaled)
            else:
                # Assign classifier
                my_classifier = classifiers[num2]
                # Apply classifier
                my_classifier.fit(X_train, y_train)
                # Creating the test
                my_predictions = my_classifier.predict(X_test)            
    
            # Calculate and print out the accuracy of the classifier when compared with y values (true values)
            accuracy[num1][num2] = accuracy_score(y_test, my_predictions)
            num2 = num2 + 1
            #print(name)
            #print(accuracy_score(y_test, my_predictions))
        #print("\n")
    #print(accuracy)    
    
    # Set width of bar
    barWidth = 0.09    
     
    # Set height of bar
    bar1 = [accuracy[0][0], accuracy[1][0], accuracy[2][0], accuracy[3][0], accuracy[4][0], accuracy[5][0], accuracy[6][0]]
    bar2 = [accuracy[0][1], accuracy[1][1], accuracy[2][1], accuracy[3][1], accuracy[4][1], accuracy[5][1], accuracy[6][1]]
    bar3 = [accuracy[0][2], accuracy[1][2], accuracy[2][2], accuracy[3][2], accuracy[4][2], accuracy[5][2], accuracy[6][2]]
    bar4 = [accuracy[0][3], accuracy[1][3], accuracy[2][3], accuracy[3][3], accuracy[4][3], accuracy[5][3], accuracy[6][3]]
    bar5 = [accuracy[0][4], accuracy[1][4], accuracy[2][4], accuracy[3][4], accuracy[4][4], accuracy[5][4], accuracy[6][4]]
    bar6 = [accuracy[0][5], accuracy[1][5], accuracy[2][5], accuracy[3][5], accuracy[4][5], accuracy[5][5], accuracy[6][5]]
    bar7 = [accuracy[0][6], accuracy[1][6], accuracy[2][6], accuracy[3][6], accuracy[4][6], accuracy[5][6], accuracy[6][6]]
    bar8 = [accuracy[0][7], accuracy[1][7], accuracy[2][7], accuracy[3][7], accuracy[4][7], accuracy[5][7], accuracy[6][7]]
    bar9 = [accuracy[0][8], accuracy[1][8], accuracy[2][8], accuracy[3][8], accuracy[4][8], accuracy[5][8], accuracy[6][8]]
     
    # Set position of bar on X axis
    br1 = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021])
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x + barWidth for x in br6]
    br8 = [x + barWidth for x in br7]
    br9 = [x + barWidth for x in br8]
     
    # Create plots
    fig, ax = plt.subplots()
    ax.bar(br1, bar1, color ='red', width = barWidth,
            edgecolor ='white', label ='Nearest Neighbors')
    ax.bar(br2, bar2, color ='orange', width = barWidth,
            edgecolor ='white', label ='Linear SVM')
    ax.bar(br3, bar3, color ='yellow', width = barWidth,
            edgecolor ='white', label ='RBF SVM')
    ax.bar(br4, bar4, color ='blue', width = barWidth,
            edgecolor ='white', label ='Gaussian Process')
    ax.bar(br5, bar5, color ='cyan', width = barWidth,
            edgecolor ='white', label ='Decision Tree')
    ax.bar(br6, bar6, color ='green', width = barWidth,
            edgecolor ='white', label ='Random Forest')
    ax.bar(br7, bar7, color ='pink', width = barWidth,
            edgecolor ='white', label ='Neural Net')
    ax.bar(br8, bar8, color ='violet', width = barWidth,
            edgecolor ='white', label ='AdaBoost')
    ax.bar(br9, bar9, color ='purple', width = barWidth,
            edgecolor ='white', label ='Naive Bayes')
     
    # Add labels
    #ax.xlabel('Database Year', fontweight ='bold', fontsize = 15)
    #ax.ylabel('Classifier Accuracy', fontweight ='bold', fontsize = 15)
    #ax.xticks([r + barWidth + 0.2625 for r in range(0,5)],
            #['2015', '2017', '2019', '2020', '2021'])
    
    # Display graph
    st.pyplot(fig)


##############
## Overview ##
##############
def dfOverviewPage(): 
    global df15, df16, df17, df18, df19, df20, df21, df22, accuracy
    
    st.write('Overview')
    choice = st.selectbox("Select Happiness Dataset Year",('','2015','2016','2017','2018','2019','2020','2021'))
    
    # Display each dataframe
    if (choice == "2015"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df15.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df15[[columns[0]]], df15['Happiness Score'], s=8)
        ax.scatter(df15[[columns[1]]], df15['Happiness Score'], s=4)
        ax.scatter(df15[[columns[2]]], df15['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df15)   
    if (choice == "2016"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df16.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])        
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df16[[columns[0]]], df16['Happiness Score'], s=8)
        ax.scatter(df16[[columns[1]]], df16['Happiness Score'], s=4)
        ax.scatter(df16[[columns[2]]], df16['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df16)  
    if (choice == "2017"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df17.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])        
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df17[[columns[0]]], df17['Happiness Score'], s=8)
        ax.scatter(df17[[columns[1]]], df17['Happiness Score'], s=4)
        ax.scatter(df17[[columns[2]]], df17['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df17)   
    if (choice == "2018"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df18.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])    
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df18[[columns[0]]], df18['Happiness Score'], s=8)
        ax.scatter(df18[[columns[1]]], df18['Happiness Score'], s=4)
        ax.scatter(df18[[columns[2]]], df18['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df18)    
    if (choice == "2019"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df19.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])    
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df19[[columns[0]]], df19['Happiness Score'], s=8)
        ax.scatter(df19[[columns[1]]], df19['Happiness Score'], s=4)
        ax.scatter(df19[[columns[2]]], df19['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df19) 
    if (choice == "2020"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df20.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])    
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df20[[columns[0]]], df20['Happiness Score'], s=8)
        ax.scatter(df20[[columns[1]]], df20['Happiness Score'], s=4)
        ax.scatter(df20[[columns[2]]], df20['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df20) 
    if (choice == "2021"):
        # Create graphs for top 3 correlations with Happiness Score
        st.write("Top 3 Correlated Columns w/ Happiness Score")
        corr = df21.corr()
        temp = corr["Happiness Score"]
        temp = temp.sort_values(ascending = False)
        temp = temp.drop("Happiness Score")    
        temp = temp.head(3)
        temp = temp.to_frame()
        columns = list(("","","")) 
        i = 0
        for index in temp.index:
            columns[i] = index
            i = i + 1 
        st.write("- " + columns[0])
        st.write("- " + columns[1])
        st.write("- " + columns[2])    
        # Plot graphs
        fig, ax = plt.subplots()
        ax.scatter(df21[[columns[0]]], df21['Happiness Score'], s=8)
        ax.scatter(df21[[columns[1]]], df21['Happiness Score'], s=4)
        ax.scatter(df21[[columns[2]]], df21['Happiness Score'], s=2)
        st.pyplot(fig)        
        st.write(df21)             
    #st.write("2022")
    #st.write(df22)    
 
 
    


#---------------------------------------------------------Other---------------------------------------------------------#





#####################
## Page Functions  ##
#####################
page_names_to_funcs = {
    "Welcome Page": welcomePage,
    "Mean Happiness": meanPage,
    "Classifier Predictions": classifierPage,
    "Dataframe Overview": dfOverviewPage
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
    
    
    
    
    