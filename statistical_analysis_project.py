url="https://raw.guthubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import numpy as np


#load the dataset
url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df= pd.read_csv(url)

#inspect the dataset
print(df.info())
print(df.describe())

del df["sex"]
del df["smoker"]
del df["day"]
del df["time"]

#visualize the distribution of total bill amounts
sns.histplot(df['total_bill'], bins=30, kde=True)
plt.title('Distribution of Total Bill Amounts')
plt.show()

#correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

from scipy.stats import ttest_ind

#seperate data by gender
male_tips = df[df['sex'] == 'Male']['tip']
female_tips = df[df['sex'] == 'Female']['tip']

#perform t-test
t_stat, p_value = ttest_ind(male_tips, female_tips)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
print("The p-value is greater than 0.05, indicating that there is no significant difference in tips between male and female customers.")

#interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in tips between male and female customers.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in tips between male and female customers.")


#define variables
x=df['total_bill'].values.reshape(-1,1)
y=df['tip'].values

#fit linear regression model
model = LinearRegression()
model.fit(x,y)

#output the coefficients
print("slope: ", model.coef_[0])
print("intercept: ", model.intercept_)
print("R-squared: ", model.score(x,y))

#plot regression line
sns.scatterplot(x='total_bill', y='tip', data=df)
plt.plot(x, model.predict(x), color='red', linewidth=2)
plt.title('Linear Regression: Total Bill vs Tip')
plt.legend()
plt.show()

