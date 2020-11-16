#simple fast linear svm
#svm stands for support vector machine
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
plt.rcParams['axes.facecolor']='w'
from matplotlib.lines import Line2D
class SVM:
    '''
    #The loss function that helps maximize the margin is hinge loss.
    #The cost is 0 if the predicted value and the actual value are of the same sign.
    #If they are not, we then calculate the loss value. 
    #take partial derivatives:
    #min(λ||w||^2) --> w=w-α.(2λw)
    #loss -->(-xi.yi)+
    #w :(not necessarily normalized)normal vector to the hyperplane(normal vector)
    #x : set of points (samples and features)
    #bias : hyper plane intersect (each distances)
    #offset : SVMs distances from decision boundary (margins)
    #margin: reinforcement range of values([-1,1])
    #α : Learning rate
    #λ : regularization parameter (to balance the margin maximization and loss)
    #Iteration : repetition of process
    '''
    def __init__(self, learningRate=0.001, lambdaParameter=0.01, Iteration=1000):
        
        self.α = learningRate
        self.λ = lambdaParameter
        self.Iteration = Iteration
        self.w = None
        self.bias = None
        self.fig = plt.figure('SVM PLOT',figsize=(9, 6))
        self.ax = self.fig.add_subplot(111)#1*1 grid , no.1
        self.colors=[]
        for i in y:
            if i==1:
                self.colors.append('c')
            else:
                self.colors.append('r')
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.Iteration):
            for x_id, x_i in enumerate(X):#Gradient Updates
                if y[x_id] * (np.dot(x_i, self.w) - self.bias) >= 1:#No misclassification
                    #w=w-α.(2λw)
                    self.w -= self.α * (2 * self.λ * self.w)
                else:#misclassification
                    #w=w-α.(yi.xi-2λw)
                    self.w -= self.α * (2 * self.λ * self.w - np.dot(x_i, y[x_id]))
                    self.bias -= self.α * y[x_id]
    def predict(self, X):
        group = np.sign(np.dot(X, self.w) - self.bias)
        if group != 0 :
            if group == 1:
                self.ax.scatter(X[0],X[1],s=200,marker='*',color='c')
            else:
                self.ax.scatter(X[0],X[1],s=200,marker='*',color='r')
        return group
    def visualization(self):
        def HP(x, w, bias, offset):
            #hyper plane 
            return (-w[0] * x + bias + offset) / w[1]
        GlobalMin = [np.amin(X[:,0]),np.amin(X[:,1])]
        GlobalMax = [np.amax(X[:,0]),np.amax(X[:,1])]
        plt.scatter(X[:,0], X[:,1], marker='o',s=100,c=self.colors)
        self.ax.plot([GlobalMin[0], GlobalMax[0]],
                     [HP(GlobalMin[0], clf.w, clf.bias, 0),
                      HP(GlobalMax[0], clf.w, clf.bias, 0)],
                     'g--',linewidth=2)
        self.ax.plot([GlobalMin[0], GlobalMax[0]],
                     [HP(GlobalMin[0], clf.w, clf.bias, -1),
                      HP(GlobalMax[0], clf.w, clf.bias, -1)],
                     'k',linewidth=2)
        self.ax.plot([GlobalMin[0], GlobalMax[0]],
                     [HP(GlobalMin[0], clf.w, clf.bias, 1),
                      HP(GlobalMax[0], clf.w, clf.bias, 1)],
                     'c',linewidth=2)
        self.ax.set_ylim([GlobalMin[1]-3,GlobalMax[1]+3])
        customLines = [Line2D([0], [0], color='c', lw=2),
                        Line2D([0], [0], color='k', lw=2),
                        Line2D([0], [0], linestyle='--',color='g', lw=2),
                        Line2D([0], [0], color='w', marker='*',
                               markersize=15,markerfacecolor='k'),
                        Line2D([0], [0], color='w', marker='o',
                               markersize=10,markerfacecolor='k')]
        self.ax.legend(customLines,['+SVM ',
                                    '-SVM', 'boundary','new data','data'],
                       loc='upper left', shadow=True)
        self.ax.set_xlabel('X1',fontsize=12)
        self.ax.set_ylabel('X2',fontsize=12)

        svmP=str(np.round((-clf.w[0]/clf.w[1]),3))+' X '+str(np.round((clf.bias+1)/clf.w[1],3))
        print('Positive svm : ',svmP)
        hp=str(np.round(-clf.w[0]/clf.w[1],3))+' X '+str(np.round(clf.bias/clf.w[1],3))
        print('decision boundary : ',hp)
        svmN=str(np.round(-clf.w[0]/clf.w[1],3))+' X '+str(np.round((clf.bias-1)/clf.w[1],3))
        print('Negative svm : ',svmN)
        s='+svm : '+svmP+'\n\n'+'decision boundary : '+hp+'\n\n'+'-svm : '+svmN
        plt.text(GlobalMin[0],
                 GlobalMin[1],s, fontsize=12)
        self.ax.set_title('Two classes, linearly separable data',fontsize=12)
        plt.show()

#define input data
X=np.array([[0,0],[2,6],[3,8],[1.5,3],[5,0],[6,2],[7,-3]])#sample,feature
y=np.array([1,1,1,1,-1,-1,-1])#tags (wich group belongs)
#print(X,y)
#call svm class : 1)fit 2)predict 3)visualize
clf = SVM(learningRate=0.0001, lambdaParameter=0.001, Iteration=10000)
clf.fit(X, y)
newData=[[0,2],[1,3],[3,4],[2,-1],
         [2,0.5],[5,1],[5,-2],[6,-5],
         [5,8]]
for data in newData:
    print(clf.predict(data))
clf.visualization()

