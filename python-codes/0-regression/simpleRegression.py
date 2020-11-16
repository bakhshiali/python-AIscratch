#simple linear regression
##Used Modules
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random
##Used Pyplot style
plt.style.use('seaborn-darkgrid')
##Definition of regression function
def FitLine(X,Y):
    slope=(((mean(X) * mean(Y))-mean(X*Y))/
       ((mean(X)**2)-mean(X**2)))
    intercept=mean(Y)-slope*mean(X)
    rLine=[(slope*x)+intercept for x in X]
    LYmean=[mean(Y) for y in Y]
    return slope,intercept,rLine,(1-(sum((Y-rLine)**2)/sum((Y-LYmean)**2)))
##Define Input Data
X=np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float64)
Y=np.array([3,2,4,5,6,7,10,9,14,12], dtype=np.float64)
##Call Functions
slope,intercept,rLine,rs=FitLine(X,Y)
print('Slope (m) = ',slope ,' , Y Intercept= ',intercept,
      ' , R^2 = ',rs)
##Predict Values
NewX=11
NewY=(slope*NewX)+intercept
##Plot Data
plt.scatter(X,Y)
plt.scatter(NewX,NewY,color=[tuple(e/255 for e in (200,0,200))])
plt.plot(X,rLine)
plt.title('Fit Line to Data.')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
