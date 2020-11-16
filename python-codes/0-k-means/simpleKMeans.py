#simple k-means
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='w'
from matplotlib.lines import Line2D
class KMeans:
    '''
    k = number of groups / clusters / ... (group)
    tolerance = acceptable level of variation in precision (tol)
    Iteration : repetition of process
    '''
    ##you could use another tolerance stop limits as :
    #error : (actual-forecast/forecast)*100
    #accuracy : 1-error
    #Note : centroid, center of mass and geometric center could be different.
    def __init__(self, group=2, maxTolerance=0.001, iteration=300):
        self.k = group
        self.tol = maxTolerance
        self.iteration = iteration
        self.fig = plt.figure('K-Means PLOT',figsize=(9, 6))
        self.ax = self.fig.add_subplot(111)#1*1 grid , no.1
        self.colors = 200*["r","g","b","k","c"]
    def fit(self,data):
        self.centroids = {}
        #start with first k data as centroids
        self.centroids={i:data[i] for i in range(self.k)}
        for _ in range(self.iteration):
            self.classes={i:[] for i in range(self.k)}
            for j in data:#j : featureset
                distances = [np.linalg.norm(j-self.centroids[i]) for i in self.centroids]
                self.classes[np.argmin(distances)].append(j)#min as cluster
            pc = self.centroids #pc : prev_centroids
            self.centroids={i:np.average(self.classes[i],axis=0) for i in self.classes}
            print(self.centroids) 
            op=[False for c in self.centroids if np.sum(self.centroids[c]-pc[c]) > self.tol]
            if op==[] : break #not op : optimum
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[i]) for i in self.centroids]
        self.ax.scatter(data[0], data[1], marker="*",
                    color=self.colors[np.argmin(distances)], s=150, linewidths=2)
        return np.argmin(distances)
    def visualize(self):
        for centroid in clf.centroids:
            self.ax.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                            marker="$C$", color="k", s=100, linewidths=2)
        for j in clf.classes:
            [plt.scatter(i[0],i[1],marker="x",color=self.colors[j],s=150,linewidth=2) for i in clf.classes[j]]
        self.ax.set_title('K-Means clustering, untagged data',fontsize=14)
        self.ax.set_xlabel('X1',fontsize=12)
        self.ax.set_ylabel('X2',fontsize=12)
        customLines = [Line2D([0], [0], color='w', marker='*',
                               markersize=15,markerfacecolor='k'),
                        Line2D([0], [0], color='w', marker='$x$',
                               markersize=15,markerfacecolor='k'),
                       Line2D([0], [0], color='w', marker='$C$',
                               markersize=15,markerfacecolor='k')]
        self.ax.legend(customLines,['new data','data','Center'],
                       loc='upper center', shadow=True)
#define input data
X = np.array([[1,4],[1.5,1.8],[7,7],[8,8],
              [1,0.6],[9,9],[0,2],[8,6],
              [0,1],[3,8],[2,10],[0,10],[1,8],
              [2,8]])
#call Kmeans functions
clf = KMeans(group=3,maxTolerance=0.001, iteration=300)
clf.fit(X)
clf.visualize()
newData = np.array([[1,3],[8,4],[0,3],[4,4],[3,6],[6,6],[4.5,7],[4.6,7]])
for unknown in newData:
    clf.predict(unknown)
plt.show()
