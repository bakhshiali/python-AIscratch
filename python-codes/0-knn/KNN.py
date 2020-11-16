##KNN stands for k nearest neighbors
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-darkgrid')
#k : number of nearest neighbors
def KNN(prediction,taggedData,k=3):
    '''
    Gets prediction as untagged newData, taggedData as input dict data,
    k as numer of nearest neighbors.
    returns newData tag as prediction using KNN algorithm.
    '''
    if len(taggedData)>=k:
        print('number of nearest neighbors can\'t be less than data numbers!')
    distances=[]
    centerPoints={}
    for tag in taggedData:
        for point in taggedData[tag]:
            distances.append([np.linalg.norm(np.array(point)-np.array(prediction)),tag])
        centerPoints[tag]=np.average(taggedData[tag][:],axis=0)
    near=[i[1]for i in sorted(distances)[:k]]
    BelogGroup=Counter(near).most_common(1)[0][0]
    return BelogGroup,centerPoints
##define input data with tagging (dictionary) and new untagged data
data={'down':[[1,1],[1,3],[3,1],[2,2],[2,4],[3,5],[3,3],[5,2],[4,3]],
      'up':[[6,5],[7,7],[8,6],[9,7],[9,8],[7,8],[8,9],[6,6],[7,5]]}#two classes (dict)
newData=[5,4.3]#[5,7]
##Call KNN function and print result, cPs stands for centerPoints
result,cPs=KNN(newData,data,k=6)
plt.title('k=6 , newData=[5,4.3]')
print('New Data belongs to \'',result,'\' group.')
##Plot all using coloring corresponding to tags
colors={'down':'y','up':'b'}
[[plt.scatter(j[0],j[1],s=100,color=colors[i]) for j in data[i]] for i in data]
##cPs stands for center points
[plt.scatter(cPs[i][0],cPs[i][1],s=100,marker='$C$',color=colors[i]) for i in cPs]
plt.scatter(newData[0],newData[1],color=colors[result])
##boundary line based on center of masses
middlePoint=[(cPs['down'][0]+cPs['up'][0])/2,(cPs['down'][1]+cPs['up'][1])/2]
boundrySlope=-1/((cPs['down'][1]-cPs['up'][1])/(cPs['down'][0]-cPs['up'][0]))
Yintercept=middlePoint[1]-boundrySlope*middlePoint[0] #b=Y-mX
print('Boundary line :  Y=',boundrySlope,'X +',Yintercept)
x=np.linspace(2,8)
y=boundrySlope*x+Yintercept #Y=mX+b
plt.plot(x,y,linestyle='-.',color='c')
plt.scatter(middlePoint[0],middlePoint[1],s=100,
            marker='$M$',color=[tuple(e/255 for e in (200,0,200))])
plt.show()

