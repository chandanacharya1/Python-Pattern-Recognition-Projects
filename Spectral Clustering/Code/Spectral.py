import csv
import numpy as np
from pylab import plot,show
from scipy.cluster.vq import kmeans,vq
import numpy.linalg as la
import matplotlib.pyplot as plt


def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)
    # plot the data
    axs.plot(X[0, :], X[1, :], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0, :].min()
    xmax = X[0, :].max()
    axs.set_xlim(xmin - 1, xmax + 1)
    axs.set_ylim(-1, X[1, :].max() + 1)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


def cluster(Fiedler,Actualdata):
    N=len(Fiedler)

    for i in range(N):
        if(Fiedler[i]>0):
            c=Actualdata[i]
            plt.plot(c[:1],c[1:],'ro')
        else:
            c = Actualdata[i]
            plt.plot(c[:1],c[1:], 'yo')

    plt.show()
    #plt.savefig("spectral.pdf")

# import csv file
with open('data-clustering-2.csv', newline='') as myfile:
    reader = csv.reader(myfile)

    x = list(reader)
    X = np.array(x).astype("float")
    data = X.T

# Plotting the input data
plotData2D(X,'plot.pdf')

#####################################Spectral Clustering############################################################
beta=5.5
N=len(data)
# Similarity matrix
S= [[np.exp((-beta)*(np.linalg.norm(data[i]-data[j])**2)) for j in range(N)]for i in range(N)]
#Diagonal Matrix
D = np.diag([sum([S[j][i] for i in range(j+1)]) for j in range(N)])
# Laplacian Matrix
L=D-S
# finding Eigen values and Eigen Vectors
l, U = la.eigh(L)
# eigh returns sorted values
# Eigen vector u2 that corresponds to eigen value lamda 2
FiedlerVector = U[:,1]
# Cluster and plot
cluster(FiedlerVector,data)
###################################################################################################################




