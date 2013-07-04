from numpy.core.memmap import memmap
import numpy as np

class LazyMemmap(memmap):
    batch_size = 1000
    avg = None
    stdev = None
   
    def mean(self, axis=0):
        print 'calculating mean'
        if self.avg is not None and axis in self.avg: 
            return self.avg[axis]

        if self.avg is None:
            self.avg = {}

        numSamples = self.shape[0]
        numBatches = numSamples/self.batch_size
        
        sampleSum = None
        for i in xrange(numBatches+1):
            #print 'batch number', i+1
            index1 = self.batch_size * i
            index2 = index1 + self.batch_size
            #print index1, index2
            if index2 > numSamples:
                index2 = numSamples

            if axis != 0:
                #print np.mean(np.asarray(self[index1:index2]), axis=axis).shape
                if sampleSum is None:
                    sampleSum = np.mean(np.asarray(self[index1:index2]), axis=axis)
                else:
               #     print sampleSum.shape, np.mean(np.asarray(self[index1:index2]), axis=axis).shape
                    sampleSum = np.hstack((sampleSum, np.mean(np.asarray(self[index1:index2]), axis=axis)))
                    
            else:
                if sampleSum is None:
                    sampleSum = np.sum(self[index1:index2], axis=axis)
                else:
                    sampleSum += np.sum(self[index1:index2], axis=axis)

            #if index2 == numSamples:
            #    break
            
            
        if axis != 0:
            self.avg[axis] = np.cast['uint8'](sampleSum.T)
            
        else:
            self.avg[axis] = np.cast['uint8'](sampleSum/numSamples)

        print self.avg[axis].shape
        return self.avg[axis]
        
    def std(self, axis=0):
        print 'in std'
        if self.stdev is not None and axis in self.stdev:
            return self.stdev[axis]

        if self.stdev is None:
            self.stdev = {}

        meanSamples = self.mean( axis)
        numSamples = self.shape[0]
        sampleSum = 0
        numBatches = numSamples/self.batch_size
        for i in xrange(numBatches+1):
            #print 'batch number', i+1
            index1 = self.batch_size * i
            index2 = index1 + self.batch_size
            if index2 > numSamples:
                index2 = numSamples
            
            sampleSum += np.sum(np.square(self[index1:index2] - meanSamples), axis=axis)
        sampleSum /= numSamples
        output = np.cast['float32'](np.sqrt(sampleSum))

        self.stdev[axis] = output

        print output.shape
        return output
        
