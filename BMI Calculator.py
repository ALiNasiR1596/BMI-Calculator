#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# ## Importing Libraries and Data

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

#Importing data
data = pd.read_csv('Body_Measurements.csv')
data.head()


# ## Reviewing Data

# We need to see if the data set is complete with valid entries. This means checking if there are:
#  - any null value entries
#  - an equal number of entries for all features (columns)



data.isnull().any()


# This means we have no null values in the data. Next we want to check:
#  - what features we are working with
#  - how much data there is




#Seeing what columns we have, what the data types there are, 
#and what number of entries per column we have.
data.info()


# This tells us that there are 4 feature columns - `Gender`, `Height`, `Weight`, and `Index` (BMI Index). 
# 
# Knowing that BMI is a calculated value, we can say that the variables we will be working with are `Gender`,`Height`, and `Weight`.
# 
# We can verify this with the data description given to us:
# 
# >Gender : Male / Female
# 
# >Height : Number (cm)
# 
# > Weight : Number (Kg)
# 
# >Index :
# 0 - Extremely Weak, 1 - Weak, 2 - Normal, 3 - Overweight, 4 - Obesity, 5 - Extreme Obesity

# ## Data Visualization
# 
# Now we can start looking for trends in the data.

# We can first check the distribution of data




# Set default plot grid
sns.set_style('whitegrid')





# Index Historgram: Frequency of values falling under each Index [0,1,2,3,4,5]
plt.rcParams['figure.figsize'] = (6, 6)
sns.countplot(data['Index'], palette='YlGnBu')
ax = plt.gca()
ax.set_title("Histogram of Index")





# Height Historgram: Frequency of values falling under certain height intervals
plt.rcParams['figure.figsize'] = (30, 10)
sns.countplot(data['Height'], palette='YlGnBu')
ax = plt.gca()
ax.set_title("Histogram of Height")





# Weight Historgram: Frequency of values falling under certain weight intervals
plt.rcParams['figure.figsize'] = (30, 10)
sns.countplot(data['Weight'], palette='YlGnBu')
ax = plt.gca()
ax.set_title("Histogram of Weight")





# Plot relation between weight and height
sns.jointplot(x='Weight', y='Height', data=data, kind='kde')





# Trend in Gender based on relationship between Height and Weight
sns.lmplot(x='Height', y='Weight', hue='Gender', data=data,
           fit_reg=True, height=7, aspect=1.25, palette = "Accent")
ax = plt.gca()
ax.set_title("Height Vs Weight Data Grouped by Gender")


# The distribution of height vs weight does not follow any trends when categorized by gender.
# 
# So, we can hypothesize that gender does not affect the index/BMI value significantly.




# Trend in Index based on relationship between Height and Weight 
sns.lmplot(x='Height', y='Weight', hue='Index', data=data,
           fit_reg=True, height=7, aspect=1.25, palette='Accent')
ax = plt.gca()
ax.set_title("Height Vs Weight Data Grouped by Index")


# We can make out distinct bands in the data based on the index value.
# 
# So, there is a general positive correlation between height and weight when categorized by index value.
# 
# Now, let us see if there are any discrepencies in the relation when looking at each gender separately.




# Segregate data based on whether the gender is Male or Female
male_data = data[data['Gender']=='Male']
female_data = data[data['Gender']=='Female']





# Trend in Index based on relationship between Height and Weight 
male_data = data[data['Gender']=='Male']
female_data = data[data['Gender']=='Female']
sns.lmplot(x='Height', y='Weight', hue='Index', data=male_data,
           fit_reg=True, height=7, aspect=1.25,palette='Accent')
ax = plt.gca()
ax.set_title("Male Height Vs Weight Data Grouped by Index")

sns.lmplot(x='Height', y='Weight', hue='Index', data=female_data,
           fit_reg=True, height=7, aspect=1.25,palette='Accent')
ax = plt.gca()
ax.set_title("Female Height Vs Weight Data Grouped by Index")


# We can also see if there are any correlation in the data by producing correlation matrices.




# Gives us basic correlation index for numerical variables
data.corr()





# Provides visual context for correlations via color scale
plt.rcParams['figure.figsize'] = (8, 7)
sns.heatmap(data.corr(), annot=True)


# Once again, let's see if this changes for people of different genders.




plt.rcParams['figure.figsize'] = (8, 7)
sns.heatmap(male_data.corr(), annot=True)





plt.rcParams['figure.figsize'] = (8, 7)
sns.heatmap(female_data.corr(), annot=True)


# ## Processing Data

# Before moving on to creating the predictive model, we need to find a way to include our non-numeric variable which is the Gender.
# Even though we found that it does not 
# 
# Knowing gender is a categorical value (Male/Female based on the data description) we need to encode the data for Gender to make it useable.
# 
# There are 2 ways we can do this:
#  - Ordinal Encoding:
#      - Assign arbitrary numbers such as 1 to `Female` and 0 to `Male` (similar to a Boolean/Truth value) to differentiate them
#      - End up with one new column with number values from 0 to n representing n unique values
#  - One-Hot Encoding:
#      - Create dummy variables, where we produce new columns for `Female` and `Male` and have binary values (0 or 1) for each of them
#      - End up with n new columns representing n unique values, but we only use n-1 of these columns dropping the last one as it ends up being redundant in nature




# Ordinal Encoding
data["Gender"] = data["Gender"].astype('category')
data["Gender_Enc"] = data["Gender"].cat.codes
data.head()





# One Hot Encoding
dummies = pd.get_dummies(data['Gender'])
data = data.join(dummies)
data.head()


# Here we can see that the results from Ordinal Encoding and One-Hot Encoding are very similar.
# 
# The new column `Male` from One-Hot Encoding is the same as the column `Gender_Enc` from Ordinal Encoding.  <br />
# This happens to be the case since we are working with the categorical variable `Gender` which only has two exclusive values in the data.
# 
# So, for this case it does not matter which encoded values for `Gender` we use.




# Dropping last two columns with dummy values from one-hot encoding as they are redundant
data = data.drop(columns=['Male', 'Female'], axis=1)
data.head()


# ## Building Model

# For making our predictive model we will need to proceed with certain steps:
#  - Assign our data instances and target value (X and y columns)
#  - Split data into training and test sets
#  - Train our model
#  - Test and evaluate the model

# 

# ### Prepare Data

# Since we want to predict what Index a person would be assigned based on their height, weight, and gender we will be making `Index` our target value or y. <br>
# So, this makes our features `Height`, `Weight`, and `Gender_Enc` or X. <br>




# Select columns to add to X and y sets
features = list(data.columns.values)
features.remove('Gender')
features.remove('Index')
X = data[features]
y = data['Index']


# Next we split the X and y data between the training set and testing set.




# Import additional required libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import *
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.model_selection import cross_val_score





# Import required class from sklearn library
from sklearn.model_selection import train_test_split

# Split X and y into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)


# ### Train Model

# Here we will introduce our model and train it to fit the data we give it. <br>
# We will be using the k-nearest neighbours algorithm first.




# Import required class from sklearn library
from sklearn.neighbors import KNeighborsClassifier

# Fit Linear Regression classifier
reg = LinearRegression()
reg.fit(X,y)


# ### Test and Evaluate Model

# To test our model, we will:
#  - Run and compare the models predictions to the real values using `X_test` and `y_test`
#  - Produce a confusion matrix and classification report
#  - Get mean accuracy scores and error rate for the model




# Run a prediction
y_pred = reg.predict(X_test)





# Import remaining required classes from sklearn
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.model_selection import cross_val_score





#Get confusion matrix
print(confusion_matrix(y_test,y_pred))





# Get classification report
print(classification_report(y_test,y_pred))





# Get accuracy score
score = np.mean(y_pred == y_test)
print(score)





# Get error rate
error = np.mean(y_pred != y_test)
print(error)


# Let's compare the results of the predictor to the actual values using a plot.




sns.regplot(x=y_test, y=y_pred)





fig = sns.jointplot(x=y_test, y=y_pred, kind='hex')
x0, x1 = fig.ax_joint.get_xlim()
y0, y1 = fig.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
fig.ax_joint.plot(lims, lims, ':k')    





df = pd.DataFrame({ 'ytest':y_test,'ypred':y_pred})
sns.residplot('ytest','ypred',data=df) 

