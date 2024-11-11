# Import necessary libraries
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
# Load the Iris dataset
df = px.data.iris()
# Select the 'sepal_length' attribute
attribute = 'sepal_length'
# 1. Print the first five records of the selected attribute
print("1. Printing the First Five Records:")
print(df.head())
# 2. Plot the Histogram and Kernel Density Estimation for the attribute 'sepal_length'
print("2. Print the Histogram and Kernel Density Estimation for the attribute 'sepal_length'")
sns.histplot(df[attribute], kde=True)
plt.title('Histogram and KDE of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# 3. Print the probability plot of 'sepal_length'
print("3. Print the probability plot of 'sepal_length'")
stats.probplot(df[attribute], dist="norm", plot=plt)
plt.title('Probability Plot of Sepal Length')
plt.show()

# 4. Print the skewness of the attribute 'sepal_length'
skewness = df[attribute].skew()
print(f"4. Print the skewness of the attribute 'sepal_length'\nSkewness for sepal length = {skewness}")

# 5. Print the kurtosis of the attribute 'sepal_length'
kurtosis = df[attribute].kurtosis()
print(f"5. Print the kurtosis of the attribute 'sepal_length'\nKurtosis for sepal length = {kurtosis}")
