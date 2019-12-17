#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ipywidgets as widgets
from IPython.display import display, HTML

javascript_functions = {False: "hide()", True: "show()"}
button_descriptions  = {False: "Show code", True: "Hide code"}


def toggle_code(state):

    """
    Toggles the JavaScript show()/hide() function on the div.input element.
    """

    output_string = "<script>$(\"div.input\").{}</script>"
    output_args   = (javascript_functions[state],)
    output        = output_string.format(*output_args)

    display(HTML(output))


def button_action(value):

    """
    Calls the toggle_code function and updates the button description.
    """

    state = value.new

    toggle_code(state)

    value.owner.description = button_descriptions[state]


state = False
toggle_code(state)

button = widgets.ToggleButton(state, description = button_descriptions[state])
button.observe(button_action, "value")

display(button)


# In[2]:


get_ipython().system(u'pip install plotly==4.4.1')
get_ipython().system(u'pip install --upgrade plotly-express')
get_ipython().system(u'pip install --upgrade pip')
get_ipython().system(u'pip install sqlalchemy')
get_ipython().system(u'pip install lxml')
get_ipython().system(u'pip install html5lib')
get_ipython().system(u'pip install BeautifulSoup4')
get_ipython().system(u'pip install hvplot')
get_ipython().system(u'pip install streamz')


# In[3]:


import numpy as np
import scipy as sp
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import sklearn
import hvplot.streamz
import hvplot.pandas  
import hvplot.dask
from streamz.dataframe import Random
import graphviz
import matplotlib
import IPython
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
get_ipython().magic(u'matplotlib inline')
from numpy.random import randn
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
from IPython.display import IFrame
from plotly import __version__
from plotly.offline import init_notebook_mode, plot
from plotly.graph_objs import Scatter


init_notebook_mode()

print("plotly version:", __version__)


# In[6]:


melbourne = pd.read_csv('melb_data.csv')


# In[8]:


#Display top of DataFrame

melbourne.head()


# In[9]:


#View data types and non-missing values

melbourne.info()


# In[10]:


# Stats for each column

melbourne.describe()


# In[44]:


# How to create a function for missing values by each column

def missing_values(melbourne):
    
    missing_val = melbourne.isnull().sum()
    
    missing_val_percent = 100 * missing_val / len(melbourne)
    
    missing_val_table = pd.concat([missing_val, missing_val_percent], axis=1)
    
    missing_val_table_ren_columns = missing_val_table.rename(
    columns = {0 : 'Missing Values', 1 : 'Percent of Total Values'})
    
    missing_val_table_ren_columns = missing_val_table_ren_columns[
        missing_val_table_ren_columns.iloc[:,1] !=0].sort_values(
    'Percent of Total Values', ascending=False).round(1)
    
    print ('The selected dataFrame has ' + str(melbourne.shape[1]) + ' columns.\n'
          'There are ' +  str(missing_val_table_ren_columns.shape[0])  + ' columns that have missing values.')
    
    return missing_val_table_ren_columns
    


# In[45]:


missing_values(melbourne)


# In[48]:


# Get the columns with > 47% missing
missing_melb = missing_values(melbourne);
missing_col = list(missing_melb[missing_melb['Percent of Total Values'] > 40].index)
print('We will remove %d columns.' % len(missing_col))


# In[49]:


melbourne = melbourne.drop(columns = list(missing_col))


# In[52]:


# New Title 
melbourne = melbourne.rename(columns = {'Melbourne Year Built': 'YearBuilt'})

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(melbourne['YearBuilt'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Year Built'); plt.ylabel('Land Size'); 
plt.title('Melbourne Houses Built');


# In[53]:


melbourne['YearBuilt'].describe()


# In[54]:


# Calculate first and third quartile
first_quartile = melbourne['YearBuilt'].describe()['25%']
third_quartile = melbourne['YearBuilt'].describe()['75%']

# Interquartile range
iqr = third_quartile - first_quartile

# Remove outliers
melbourne = melbourne[(melbourne['YearBuilt'] > (first_quartile - 3 * iqr)) &
            (melbourne['YearBuilt'] < (third_quartile + 3 * iqr))]


# In[55]:


# Histogram Plot of Year Built

plt.hist(melbourne['YearBuilt'].dropna(), bins = 20, edgecolor = 'black');
plt.xlabel('Year Built'); 
plt.ylabel('Count'); plt.title('Year Built Distribution');


# In[56]:


# Find all correlations and sort 
correlations_melb = melbourne.corr()['YearBuilt'].sort_values()

# Print the most negative correlations
print(correlations_melb.head(15), '\n')

# Print the most positive correlations
print(correlations_melb.tail(15))


# In[61]:


# Select the numeric columns
numeric_subset = melbourne.select_dtypes('number')

# Create columns with square root and log of numeric columns
for col in numeric_subset.columns:
    # Skip the Year Built column
    if col == 'YearBuilt':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Select the categorical columns
categorical_subset = melbourne[['Price', 'Propertycount']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Drop buildings without a Year Built
features = features.dropna(subset = ['YearBuilt'])

# Find correlations with the score 
correlations = features.corr()['YearBuilt'].dropna().sort_values()


# In[62]:


# Negative Corr.
correlations.head(15)


# In[63]:


# Positive Corr. 
correlations.tail(15)


# In[64]:


no_year_built = features[features['YearBuilt'].isna()]
year_built = features[features['YearBuilt'].notnull()]

print(no_year_built.shape)
print(year_built.shape)


# In[68]:


# Separate out the features and targets
features = year_built.drop(columns='YearBuilt')
targets = pd.DataFrame(year_built['YearBuilt'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)


# In[69]:


# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


# In[70]:


baseline_guess = np.median(y)

print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(y_test, baseline_guess))


# In[ ]:




