
from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math   import log

class Reganim(object):
    def __init__(self,file,tr_ts=0.7,lr=0.05,iterations=53,intervals=50):
        self.file=file
        self.tr_ts=tr_ts
        self.lr=lr
        self.iterations=iterations
        self.intervals=intervals
        self.fig = plt.figure()
       
       # super().__init__()
        
        self.X,self.Y = self.loadData(self.file)
        #add bias to X
        self.X= np.append(np.ones((np.size(self.X, 0), 1)),self.X, axis=1)
        added_params = [[x[1]**2, x[1]*x[2], x[2]**2] for x in np.array(self.X)]
        self.X = np.append(self.X, np.matrix(added_params), axis=1)
        #standardize X
        self.X =self.standardize(self.X)
        #create vector of parameters
        self.Theta=np.zeros((np.size(self.X, 1), 1))
        self.Theta_vals = []
        self.Error_vals = []
        for i in range(0, self.iterations):
          #print('iter=',self.iterations)
          self.Theta_vals.append(np.asarray(self.Theta).flatten())
          self.Error_vals.append(self.error(self.X,self.Y,self.Theta))
          self.Theta = self.gradientStep(self.X, self.Y, self.Theta, self.lr)
        #plot data:

        
        #.fig=Chart()
        self.def_ax = self.fig.add_subplot(211)
        #print('xlim',np.amin(self.X[:,1:2]), np.amax(self.X[:,1:2]))
      #  print('ylim',np.amin(self.X[:,2:3]), np.amax(self.X[:,2:3]))

        self.def_ax.set_xlim(np.amin(self.X[:,1:2]), np.amax(self.X[:,1:2]))
        self.def_ax.set_ylim(np.amin(self.X[:,2:3]), np.amax(self.X[:,2:3]))
        self.err_ax =self.fig.add_subplot(212)

        self.err_ax.set_ylim(0, self.error(self.X,self.Y,self.Theta))
        self.err_ax.set_xlim(0,self.iterations)
        self.positive_X1 = []
        self.positive_X2 = []
        self.negative_X1 = []
        self.negative_X2 = []
        for i in range(0, np.size(self.Y, 0)):
            if(self.Y[i, 0] == 1):
                self.positive_X1.append(self.X[i, 1])
                self.positive_X2.append(self.X[i, 2])
            else:
                self.negative_X1.append(self.X[i, 1])
                self.negative_X2.append(self.X[i, 2])

        
        self.err_ax.set_ylim(np.amin(self.Error_vals), np.amax(self.Error_vals))
        #self.anim=FuncAnimation(plt.gcf(),self.animation, frames=self.iterations, interval=self.intervals, repeat_delay=50)
        

    def sigmoid(self,x):
        #print('sigmoid outpur=',1.0 / (1.0 + np.exp(-x)))
        return 1.0 / (1.0 + np.exp(-x))

    def loadData(self,filepath):
        source=""
        try:
            f = open(filepath, "r")
            source = f.read()
            f.close()
        except IOError:
            print("Error while reading file (" + filepath + ")")
            return ""


        raw_data = source.split("\n")
        raw_data = [x.split(",") for x in raw_data if x !=""]
        raw_data = np.matrix(raw_data).astype(float)
        
        return (raw_data[:,:np.size(raw_data,1)-1], raw_data[:,np.size(raw_data, 1)-1:])

    def standardize(self,dataset, skipfirst=True):
        means = np.amin(dataset, 0)
        deviation = np.std(dataset, 0)
        if skipfirst:
            dataset[:,1:] -= means[:,1:]
            dataset[:,1:] /= deviation[:,1:]
            return dataset
        else:
            dataset -= means
            dataset /= deviation
            return dataset


# cost function =ylog(1-a)-(1-y)log(a)
    def error(self,X,Y, Theta):
        "Calculates error values"
        v_sigm = np.vectorize(self.sigmoid)
        h_x = X @ Theta
        sigmo = v_sigm(h_x)
        partial_vect = (Y-1).T @ np.log(1-sigmo) - Y.T @ np.log(sigmo) 
        err=1/(2*np.size(Y, axis=0))*np.sum(partial_vect)
        if err is not nan:
            return err
        else :return 3
         

    def gradientStep(self,X, Y, Theta, lr):
        "Returns new theta Values"
        #v_sigm = np.vectorize(self.sigmoid)
        h_x = X @ Theta
        modif = -1*lr/np.size(Y, 0)*(h_x-Y) 
        sums = np.sum(modif.T @ X, axis = 0)
        return Theta + sums.T

    
    #CALCULATING FINISHES HERE

    
    def animation(self,frame):
        
        
        def_limX = self.def_ax.get_xlim()
        def_limY = self.def_ax.get_ylim()
        err_limX =self.err_ax.get_xlim()
        err_limY =self.err_ax.get_ylim()
        self.def_ax.clear()
        self.err_ax.clear()
        self.def_ax.set_xlim(def_limX)
        self.def_ax.set_ylim(def_limY)
        self.err_ax.set_xlim(err_limX)
        self.err_ax.set_ylim(err_limY)
      
        self.def_ax.scatter(self.positive_X1, self.positive_X2, marker="^")
        self.def_ax.scatter(self.negative_X1, self.negative_X2, marker="o")
      
        self.Theta =self.Theta_vals[frame]
        res_x = np.linspace(*self.def_ax.get_xlim(), num=5)
        delta_x = [(self.Theta[4]*x+self.Theta[2])**2-4*self.Theta[5]*(self.Theta[3]*x**2+self.Theta[1]*x+self.Theta[0]-0.5) for x in res_x]
        delta_x = [np.sqrt(x) if x >= 0 else 0 for x in delta_x]
        minb = [-(self.Theta[4]*x+self.Theta[2]) for x in res_x]
        res_1 = []
        res_2 = []
        for i in range(0, len(res_x)):
            if self.Theta[5] == 0:
                res_1.append(0)
                res_2.append(0)
            else:
                res_1.append((minb[i]+delta_x[i])/(2*self.Theta[5]))
                res_2.append((minb[i]-+delta_x[i])/(2*self.Theta[5]))
      
      #  for xi in range(0,frame-1):
        self.def_ax.plot(res_x, res_1)
        self.def_ax.plot(res_x, res_2)
        err_x = np.linspace(0, frame, frame)
        err_y = self.Error_vals[0:frame]
        self.err_ax.plot(err_x, err_y)
   