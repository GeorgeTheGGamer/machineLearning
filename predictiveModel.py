# Firstly, import the needed libraries such as pandas, numPy, seaborn and scikit learn.

import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#next thing to do is read the data set

#use pandas to display the first 5 rows rows of the data set
df = pd.read_csv("any file")
df.head(5)

#next up you need to explore the data set 

# The info() function shows us the data type of each column, number of columns, memory usage, and the number of records in the dataset:
df.info()

# The shape function displays the number of records and columns:
df.shape()

# The describe() function summarizes the dataset’s statistical properties, such as count, mean, min, and max:
df.describe()

# The corr() function displays the correlation between different variables in our dataset:
df.corr()

#In order to train this Python model, we need the values of our target output to be 0 & 1. So, we'll replace values in the Floods column (YES, NO) with (1, 0) respectively:

df['FLOODS'].replace(['YES', 'NO'], [1,0], inplace=True)
df.head(5)


# Feature Selection

# Choosing features of the data set that relate most to the target ouput, i.e best need info for the predicted result.
# Use the SelectKBest library

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


