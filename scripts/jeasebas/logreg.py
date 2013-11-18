#Copyright (C) 2007 Roland Memisevic
#
#This program is distributed WITHOUT ANY WARRANTY; without even the implied 
#warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
#LICENSE file for more details.

""" 
The class Logreg in this module defines a logistic regression classifier. The rest are helper functions. 
"""

import numpy
import numpy.random



def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.
    
       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is also the default), logsumexp is 
       computed along the last dimension. 
    """
    if len(x.shape) < 2:  #only one possible dimension to sum over?
        xmax = x.max()
        return xmax + numpy.log(numpy.sum(numpy.exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + numpy.log(numpy.sum(numpy.exp(x-xmax[...,None]),lastdim))


def onehot(x,numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding. 
      
        If numclasses (the number of classes) is not provided, it is assumed 
        to be equal to the largest class index occuring in the labels-array + 1.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels. 
    """
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
    z = numpy.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result


def unhot(labels):
    """ Convert one-hot encoding for class-labels to integer encoding 
        (starting with 0!): This can be used to 'undo' a onehot-encoding.

        The input-array can be of any shape. The one-hot encoding is assumed 
        to be along the last dimension.
    """
    return labels.argmax(len(labels.shape)-1)


class Logreg(object):
    """ Logistic regression. 

        Labels are always expected to be in 'onehot'-encoding. 
        Use train() to train the model or the method train_cg() which requires the module minimize.py 
    """

    def __init__(self,numclasses, numdims):
        self.numclasses = numclasses 
        self.numdimensions = numdims 
        self.params = 0.01*numpy.random.randn(self.numclasses*self.numdimensions+self.numclasses)
        self.weights = self.params[:self.numclasses*self.numdimensions].reshape((self.numclasses,self.numdimensions))
        self.biases = self.params[self.numclasses*self.numdimensions:]
    
    def cost(self, features, labels, weightcost):
        scores = numpy.dot(self.weights,features) + self.biases[:,None]
        negloglik = ( -(labels*scores).sum() + 
                       logsumexp(scores,0).sum() ) / numpy.double(features.shape[1]) + \
                       weightcost * numpy.sum(numpy.sum(self.weights**2))
        return negloglik 

    def grad(self, features, labels, weightcost):
        gradw = numpy.zeros((self.numclasses,self.numdimensions), dtype=float)
        gradb = numpy.zeros(self.numclasses, dtype=float)
        scores = numpy.dot(self.weights,features) + self.biases[:,None]
        probs = numpy.exp(scores - logsumexp(scores, 0))
        for c in range(self.numclasses):
            gradw[c,:] = -numpy.sum((labels[c,:]-probs[c,:])*features,1) 
            gradb[c] = -numpy.sum(labels[c,:]-probs[c,:])
        gradw /= numpy.double(features.shape[1])
        gradb /= numpy.double(features.shape[1])
        gradw = gradw + 2*weightcost * self.weights
        return numpy.hstack((gradw.flatten(),gradb))

    def probabilities(self, features):
        scores = numpy.dot(self.weights,features) + self.biases[:,None]
        return numpy.exp(scores - logsumexp(scores,0))

    def classify(self, features): 
        """Use input weights to classify instances (provided columnwise 
           in matrix features.)"""
        if len(features.shape)<2:
            features = features[:,None]
        numcases = features.shape[1]
        scores = numpy.dot(self.weights,features) + self.biases[:,None]
        labels = numpy.argmax(scores, 0)
        return onehot(labels, self.numclasses).T

    def zeroone(self, features, labels):
        """ Computes the average classification error (aka. zero-one-loss) 
            for the given instances and their labels. 
        """
        return 1.0 - (self.classify(features)*labels).sum() / numpy.double(features.shape[1])

    def train(self, features, labels, weightcost, numsteps, verbose=True):
        """Train the model using gradient descent.
  
           Inputs:
           -Instances (column-wise),
           -'One-hot'-encoded labels, 
           -Scalar weightcost, specifying the amount of regularization"""
      
        numcases = features.shape[1]
        stepsize = 0.01
        gradw = numpy.zeros((self.numclasses,self.numdimensions), dtype=float)
        gradb = numpy.zeros((self.numclasses), dtype=float)
        likelihood = -self.cost(features, labels, weightcost)
        likelihood_new = -numpy.inf
        iteration = 0
        while stepsize > 10**-6 and iteration<=numsteps:
            iteration += 1
            if verbose:
                print 'stepsize:' + str(stepsize)
                print 'likelihood:' + str(likelihood)
            params_old = self.params.copy()
            self.params[:] -= stepsize * self.grad(features, labels, weightcost)
            likelihood_new = -self.cost(features, labels, weightcost)
            if likelihood_new > likelihood:
                stepsize = stepsize * 1.1
                likelihood = likelihood_new
            else:
                stepsize = stepsize*0.5
                self.params[:] = params_old 

    def f(self,x,features,labels,weightcost):
        """Wrapper function for minimize"""
        xold = self.params.copy()
        self.updateparams(x.copy())
        result = self.cost(features,labels,weightcost)
        self.updateparams(xold.copy())
        return result

    def g(self,x,features,labels,weightcost):
        """Wrapper function for minimize"""
        xold = self.params.copy()
        self.updateparams(x.copy())
        result = self.grad(features,labels,weightcost).flatten()
        self.updateparams(xold.copy())
        return result

    def updateparams(self,newparams):
        """Wrapper function for minimize"""
        self.params[:] = newparams.copy()

    def train_cg(self, features, labels, weightcost, maxnumlinesearch=numpy.inf, verbose=False):
        """Train the model using conjugate gradients.
  
           Like train() but faster. Uses minimize.py for the optimization. 
        """

        from minimize import minimize
        p, g, numlinesearches = minimize(self.params.copy(), 
                                         self.f, 
                                         self.g, 
                                         (features, labels, weightcost), maxnumlinesearch, verbose=verbose)
        self.updateparams(p)
        return numlinesearches



if __name__ == "__main__":
    #make some random toy-data:
    traininputs = numpy.hstack((numpy.random.randn(2,100)-1.0,numpy.random.randn(2,100)+1.0))
    trainlabels = onehot(numpy.hstack((numpy.ones((100))*0,numpy.ones((100))*1)).astype("int")).T
    testinputs = numpy.hstack((numpy.random.randn(2,100)-1.0,numpy.random.randn(2,100)+1.0))
    testlabels = onehot(numpy.hstack((numpy.ones((100))*0,numpy.ones((100))*1)).astype("int")).T
    testinputs = numpy.hstack((numpy.random.randn(2,100)-1.0,numpy.random.randn(2,100)+1.0))
    #build and train a model:
    model = Logreg(2,2)
    #model.train(traininputs,trainlabels,0.001)
    model.train_cg(traininputs,trainlabels,0.001)
    #or use a monte trainer object:
    #from monte.gym import trainer 
    #trainer = trainer.Conjugategradients(model,20)
    #trainer.step(traininputs,trainlabels,0.001)
    #try model on test data:
    predictedlabels = model.classify(testinputs) 
    print 'true labels: '
    print unhot(testlabels.T)
    print 'predicted labels: '
    print unhot(predictedlabels.T)
    print 'error rate: '
    print numpy.sum(unhot(testlabels.T)!=unhot(predictedlabels.T))/float(testinputs.shape[1])


