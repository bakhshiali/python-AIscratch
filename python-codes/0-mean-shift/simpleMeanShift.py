#simple Mean Shift (circle clustering)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='w'
from matplotlib.lines import Line2D

class MS:
    '''
    iteration : number of searching (for new/precise cluster) process
    r : maximum radius of data distance to belong to a cluster
    cs : centers of clusters (found with r)
    '''
    def __init__(self, r=4,iteration=300):#radius is key parameter (must guess!) to know classes
        self.iteration=iteration
        self.r = r #radius
        self.colors = 200*["g","r","k","b","c"]
        self.cs={}#centers
        self.fig = plt.figure('Mean Shift PLOT',figsize=(9, 6))
        self.ax = self.fig.add_subplot(111)#1*1 grid , no.1
    def fit(self, data):
        self.cs={i:data[i] for i in range(len(data))}
        for _ in range(self.iteration):
            ncs = []#new centers
            for c in self.cs:
                inCircle = []
                [inCircle.append(d) for d in data if np.linalg.norm(d-self.cs[c]) < self.r]       
                ncs.append(np.average(inCircle,axis=0))
            ucs=np.unique(ncs,axis=0)#remove duplicates
            pcs = self.cs #store previous centers
            self.cs = {j:ucs[j] for j in range(len(ucs))}
            for i in self.cs:
                if (self.cs[i] != pcs[i]).any():
                    break
                elif i ==len(self.cs)-1 :
                    return self.cs
    def visualize(self,data):
        plt.scatter(X[:,0], X[:,1], s=100)   
        [plt.scatter(self.cs[c][0],self.cs[c][1],color=self.colors[c],marker='$C$',s=100) for c in self.cs]
        self.ax.set_title('Mean Shift clustering, untagged data',fontsize=14)
        self.ax.set_xlabel('X1',fontsize=12)
        self.ax.set_ylabel('X2',fontsize=12)
        customLines = [Line2D([0], [0], color='w', marker='*',
                               markersize=15,markerfacecolor='k'),
                        Line2D([0], [0], color='w', marker='o',
                               markersize=10,markerfacecolor='k'),
                       Line2D([0], [0], color='w', marker='$C$',
                               markersize=15,markerfacecolor='k')]
        self.ax.legend(customLines,['new data','data','Center'],
                       loc='upper right', shadow=True)
    def predictWithRadius(self,data):
        tag = [i for i in self.cs if np.linalg.norm(data-self.cs[i]) < self.r]
        if len(tag)==1:
            plt.scatter(data[0], data[1], marker='*',s=100,color=self.colors[tag[0]])
        elif tag==[]:
            print(data," is not close to any center with r : ",self.r,"")
        else:
            common=[]
            [common.append(self.cs[t]) for t in tag]
            print(data," is in range of centers :",common," with r : ",self.r,"")
    def predict(self,data):
        distances = [np.linalg.norm(data-self.cs[c]) for c in self.cs]
        tag = (distances.index(min(distances)))
        plt.scatter(data[0], data[1], marker='*',s=100,color=self.colors[tag])
        return tag
#define input data
X = np.array([[1,1],[1.5,1.8],[1,0.6],[2,1],
              [4,12],[6,12],[5,14],[6,11],
              [10,1],[11,2],[10.5,3],
              [6,6],[6,4]])
#calling functions
ms = MS(r=4,iteration=10)
ms.fit(X)
print(ms.cs)
ms.visualize(X)
newData = np.array([[1,3],[8,10],[5,5],[10,5]])
for unknown in newData:
    ms.predict(unknown)
plt.show()
