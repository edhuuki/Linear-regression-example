from turtle import color
import numpy as np
from matplotlib import pyplot as plt

def MS_Error(M,f):				# Mean Square of error

	ms = sum([(y-f(x))**2 for x,y in M])
	ms*= 1/(2*len(M))

	return ms

def D_MS(M,theta):# derivative of mean square error
	m,b = theta

	dms_dm = sum([(y-(m*x+b))*x for x,y in M])
	dms_dm *= 1/(len(M))

	dms_db = sum([y-(m*x+b) for x,y in M])
	dms_db *= 1/(len(M))

	return np.array([dms_dm,dms_db])

theta_list = [[],[]] # for plotting how theta changes over the regression

def linear_regression(M,theta,a,epochs): # estimates [a,b] for the form y = mx+b

	for i in range(epochs):
		theta += a*D_MS(M,theta)		# learning rate is a vector for m and b
		theta_list[0].append(theta[0])		# Track the value of theta through the regression
		theta_list[1].append(theta[1])

	print(theta)
	return theta


M = np.linspace([-3,0],[0,1],100)			# y = 1*x+2
M[:,1]+= .25*(np.random.random([100])-.5) 	# adding noise to the dataset


theta = np.array([0.0,0.0])					# Initial m and b values
m,b = linear_regression(M,theta,.01,5000)

plt.scatter(M[:,0],M[:,1])

fx = lambda x:m*x+b

x = np.linspace(-3,0,10)
plt.plot(x,list(map(fx,x)),color='r')

plt.show()
# plt.plot(theta_list[0],theta_list[1])
# plt.show()