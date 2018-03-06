import timeit
import math
import operator
import numpy as np


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def compute_nearest(n):
    # prepare data
    trainingSet= np.loadtxt('data2-train.dat', dtype=np.longfloat, comments='#', delimiter=None)
    testSet= np.loadtxt('data2-test.dat', dtype=np.longfloat, comments='#', delimiter=None)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    print(testSet.shape)
    print(trainingSet.shape)
    # generate predictions
    predictions = []
    for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], n)
            result = getResponse(neighbors)
            predictions.append(result)

        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy='+ repr(accuracy) + '%')

start = timeit.default_timer() #start timer
compute_nearest(1) #for n=1
stop = timeit.default_timer() #end timer
print("Overall Runtime:")
print(stop - start) #print exectuion time
compute_nearest(3) #for n=2
compute_nearest(5) #for n=3