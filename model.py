import pandas as pd
#Powerful tool for data analysis

teams = pd.read_csv("teams.csv")
#Using the pd call, read the csv in for processing 

# print(teams)
#Shows the teams.csv data and tells you the number of rows and columns

teams = teams[["team","country","year","athletes","age","prev_medals","medals"]]
#Restates the teams data as required information, taking out the columns with the least important information

# print(teams)
#this now prints the data required

# teams.corr()["medals"]
#Uses to check if it is possible to make the predictions
#Corrolation br=etween the medals column and the other columns.
#The higher to 1 the better the values that can be used for the prediction

import seaborn as sns
#Pyphon graphing library

sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True,ci=None)
#ci means confidence interval
#using seasborn to plot points on a graph as x axis is athletes and y is medals, using the data already recieved from the csv as teams 
#Using another coding platform it will show the outputed graph 
#using the graph shown there is a linear relationship between the number pf atheltes vs medals won

sns.lmplot(x="age",y="medals",data=teams,fit_reg=True,ci=None)
#using the graph outputted, there is no relation between the age and medals 
#As proven by the correlation command which showed that the age to medals correltion is closer to 0 than to 1 meaning that there is very little correlation.
#Can be used later down the line since it seems in the data that there is more medals won in the 20 to 30 range than any other age between 0 and 60+

teams.plot.hist(y="medals")
#Makes a historgram of the medals vs frequency

#First step is done, found the data and have explored it

#Next step is to clean the data 

missing = teams[teams.isnull().any(axis=1)]
# print(missing)
#This code finds if any rows have missing values
#reason for missing values is that some teams such as albania have not participated in the previous olypics and so does not have previous medals to show for the country

teams =teams.dropna()
#This removes the empty rows in the data
# print(teams)
#The new printed teams now outputs the data that has no empty values within 

#Time to split up the data
#Take the early years as the test data, such as the last two years 
#Take the previous years and put in the training data
#Train the model and use past data to predict the future
#So for the euros predictor, the games to play is the test set and the previous fixtures against is the training data to predict the outcome of the new bouts against 

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()
#Makes copy of the original teams data and then splits by the year for the new years to predict and the previous years to train the model
#Splitting up the training and the test set
#reason is to train on the training set and use a different set to evaluate how weel the model is doing

# print(train.shape)
#Prints the number of rows and columns of the training data and next line the test data
# print(test.shape)

#Using the mean absolute error and error metric and use after we train it

from sklearn.linear_model import LinearRegression
# Allows us to train and make predictions to make a linear model
#Machine learning library

reg = LinearRegression()
#this is the linear regression model as a built in scikit learn function

predictors = ["athletes","prev_medals"]
# preditors, columns we are using to predict the target

target = "medals"
#This is the target you are tring to predict 

fit = reg.fit(train[predictors],train["medals"])
print(fit)
#This shows that this data does indeed fit into the linear regression model

predictions = reg.predict(test[predictors])
#Makes the prediction using the predictors and not the actual values
#make predictions without knowing the answers

# print(predictions)
#Numpy array
#values are not rounded, not making sense since there is a needed whole number output
#also there are negative number, and they can't have negative medals

test["predictions"] = predictions
# apply to a colums in the test data frame 
#allows to look at predictions more easily

test.loc[test["predictions"] < 0, "predictions"] = 0
#index the test data frame, find rows where predictions are less than 0 and then get replaced with 0

test["predictions"] = test["predictions"].round()
#This rounds all the predictions to a whole number to make more sense 


#Now it is time to explore the mean absolute error

from sklearn.metrics import mean_absolute_error
# Imports mean absolute error from scikit learn library

error = mean_absolute_error(test["medals"],test["predictions"])
#parse in the actual values and the predicted values into the called function
#Gets back a value given the mean absolute error
#Time to find out if this is a good value or a bad one

# print(error)

teams.describe()["medals"]
#look into the medals column with more depth using the pandas describe method
#Shows the minimum value in the columns, mean and standard deviation

#Good to have an error below the standard deviation
#An error above the standard deviation is bad, reasons could be using useless precictors,messed up the model, etc...

# print(test[test["team"]] == "USA")
#This checks the specific teams 

#Mean abolute error can be very differnt for countrys which have many medals vs countries which have very few medals.

errors = (test["medals"] - test["predictions"].abs())
#this finds the absolute error, suntracts the absolute against the medals
#actual - predicted

error_by_team = errors.groupby(test["team"]).mean()
#groupby is a pandas method which creates a seperate group for each team 
#and then finds the mean error for each team 
#printing this will show how many medals of we were for each country


#Now look at how many medals each country earned on average

medals_by_team = test["medals"].groupby(test["team"]).mean()
#Tells us how many medals each counry earned on averge in the olympics

error_ratio = error_by_team/medals_by_team
print(error_ratio)

#Many NaN values, since many teams the averge number of medals they earn is 0 and so is getting divided by 0

error_ratio[~pd.isnull(error_ratio)]
#102 countries left 
#Some countries have infinite medals
#Other way round, dividing by 0 is infity and so we need to take out the ones with 0 again

import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
print(error_ratio)
#And now there is 97 values left
#With proper error ratio

#make another histogram
error_ratio.plot.hist()
#Shows that the error ratio is 2 and above at some points and at points is below 1 and so pretty far off the predicted value.


#Some countrys the ratios is pretty good

error_ratio.sort_values()
# Shows how countries which send many athletes the model performs well but at times countries which have very few atheltes does not 



#Ways to improve 

#Add in more predictors
#Try differnt models 

























