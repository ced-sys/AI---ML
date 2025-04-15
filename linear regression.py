from sklearn.linear_model import LinearRegression
import numpy as np

#Example data
X=np.array([[50], [100], [150], [200], [250], [300]]) #Input (e.g., hours studied)
y=np.array([100, 200, 300, 400, 500, 600]) #Output (e.g., test scores)


#Train the model
model=LinearRegression()
model.fit(X, y)

print("Slope:", model.coef_[0])
print("Y-intercept:" ,model.intercept_)
#Make Predictions




