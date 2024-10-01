import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("student_scores.csv")

# print(data.head())
#points reperesnt the dataframe
def loss_function(m,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        total_error += (y - (m*x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_old,b_old,points, lr):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range (n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores

        m_gradient += -(2/n) * x * (y - (m_old * x + b_old))
        b_gradient += -(2/n) * (y - (m_old * x + b_old))

    m_new = m_old - m_gradient * lr
    b_new = b_old - b_gradient * lr
    return m_new,b_new

m =0 
b = 0
lr = 0.0001
epochs = 500

for i in range (epochs):
    print(f"Epoch:{i}")
    m,b = gradient_descent(m,b,data, lr)

print(m,b)

#print the mse from the loss function
mse = loss_function(m,b,data)
print(f"mean square error: {mse}")

# Scatter plot of data points
plt.scatter(data.Hours, data.Scores, label='Data Points', color='blue')  

# Plot the regression line
plt.plot(list(range(1, 10)), [m * x + b for x in range(1, 10)], color='red', label='Regression Line')  

# Adding labels and title
plt.xlabel('Hours Studied')  
plt.ylabel('Scores Obtained')  
plt.title('Hours Studied vs. Scores Obtained')  
plt.legend()  

# Save the plot
plt.savefig('hours_vs_scores_plot.png') 
plt.show()

