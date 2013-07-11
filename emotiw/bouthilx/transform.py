import numpy as np
from theano import config
from scipy import ndimage, misc
from pylearn2.train_extensions import TrainExtension
from pylearn2.monitor import Monitor
from pylearn2.gui import patch_viewer
from bezier import bezier_curve
import os

from pylearn2.datasets.dataset import Dataset

class TransformerDataset(Dataset):
    """
        A dataset that applies a transformation on the fly
        as examples are requested.

        Modified by plcarrier so that when get_test_set is called, 
        the raw dataset is returned instead of the TransformerDataset so
        that no transformation is performed on the test data
    """
    def __init__(self, raw, transformer, cpu_only = False,
            space_preserving=False):
        """
            raw: a pylearn2 Dataset that provides raw data
            transformer: a pylearn2 Block to transform the data
        """
        self.__dict__.update(locals())
        self.view_converter = self.raw.view_converter #########################
        self.y = self.raw.y
        del self.self

    def get_batch_design(self, batch_size, include_labels=False):
        raw = self.raw.get_batch_design(batch_size, include_labels)
        return self.transformer.perform(raw)

    def get_topo_batch_axis(self):
        return self.view_converter.axes.index('b')

    def get_batch_topo(self, batch_size, include_labels = False):

        if include_labels:
            batch_design, labels = self.get_batch_design(batch_size, True)
        else:
            batch_design = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        if include_labels:
            return rval, labels

        return rval

#    def get_batch_topo(self, batch_size, include_labels = False):
#        """
#        If the transformer has changed the space, we don't have a good
#        idea of how to do topology in the new space.
#        If the transformer just changes the values in the original space,
#        we can have the raw dataset provide the topology.
#        """
#        if include_labels:
#            
#        data = self.get_batch_design(batch_size)
#        if self.space_preserving:
#            return self.raw.get_topological_view(data)
#        return X.reshape(X.shape[0],X.shape[1],1,1)

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        raw_iterator = self.raw.iterator(mode, batch_size, num_batches, topo, targets, rng, data_specs, return_tuple)

        final_iterator = TransformerIterator(raw_iterator, self)

        return final_iterator

    def has_targets(self):
        return self.raw.y is not None

    def adjust_for_viewer(self, X):
        if self.space_preserving:
            return self.raw.adjust_for_viewer(X)
        return X

    def get_weights_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def get_topological_view(self, *args, **kwargs):
        if self.space_preserving:
            return self.raw.get_weights_view(*args, **kwargs)
        raise NotImplementedError()

    def adjust_to_be_viewed_with(self, *args, **kwargs):
        return self.raw.adjust_to_be_viewed_with(*args, **kwargs)


class TransformerIterator(object):

    def __init__(self, raw_iterator, transformer_dataset):
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        self.uneven = raw_iterator.uneven

    def __iter__(self):
        return self

    def next(self):

        raw_batch = self.raw_iterator.next()

        if self.raw_iterator._targets:
            rval = (self.transformer_dataset.transformer.perform(raw_batch[0]), raw_batch[1])
        else:
            rval = self.transformer_dataset.transformer.perform(raw_batch)

        return rval

    @property
    def num_examples(self):
        return self.raw_iterator.num_examples

# Decorator to reshape batch data before and after transformations.
# Transformations need a shape of ['b',0,1,'c']
# The class to which belongs the wrapped function must have an attribute 
# input_space which is a Conv2DSpace object.
def reshape(default=['b',0,1,'c']):
    def decorate(perform):
        def call(self,data):
            if isinstance(data,tuple):
                (X, Y) = data
            else:
                X = data
            
#            X = X.astype('uint8')
            # convert X shape
            shape = X.shape
            if self.input_space:
                # needs reshaping
                if len(self.input_space.axes) != len(shape):
                    X = X.reshape([X.shape[0]]+list(self.input_space.shape)+[self.input_space.num_channels])

                # dimension transposition
                else:
                    # How can we detect axes of X?
                    X = self.input_space.convert_numpy(X,self.input_space.axes,default)

            # apply perform function

            result = perform(self,X)

            # revert X shape

            if self.input_space:
                # needs reshaping
                if len(self.input_space.axes) != len(shape):
                    X = X.reshape(shape)

                # dimension transposition
                else:
                    # How can we detect axes of X?
                    X = self.input_space.convert_numpy(X,default,self.input_space.axes)

            if isinstance(data,tuple):
                return (X, Y)
            else:
                return X

        return call

    return decorate
 

class TransformationPipeline:
    """
        Apply transformations sequentially
    """
    # shape should we switched to Conv2DSpace or something similar to make space conversion more general
    def __init__(self,transformations,seed=None,shape=None,input_space=None):
        self.transformations = transformations
        self.input_space = input_space

        if seed:
            np.random.RandomState(seed)

    @reshape()
    def perform(self, X):

        import matplotlib.pyplot as plt

        for transformation in self.transformations:
#            x = X[0]
#            print x[:,:,0].min()
#            print x[:,:,1].min()
#            print x[:,:,2].min()
#            print x[:,:,0].max()
#            print x[:,:,1].max()
#            print x[:,:,2].max()
#
#            plt.imshow(x)
#            plt.show()
#
#            print transformation

            X = transformation.perform(X)

#        print x[:,:,0].min()
#        print x[:,:,1].min()
#        print x[:,:,2].min()
#        print x[:,:,0].max()
#        print x[:,:,1].max()
#        print x[:,:,2].max()
#
#        plt.imshow(x)
#        plt.show()

        return X

class TransformationPool:
    """
        Apply one of the transformations given the probability distribution
        default : equal probability for every transformations
    """
    def __init__(self,transformations,p_distribution=None,seed=None,input_space=None):
        self.transformations = transformations
        self.input_space = input_space

        if p_distribution:
            if np.sum(p_distribution)<0.9999 or np.sum(p_distribution)>1.0001:
                raise ValueError("p_distribution does not sum to 1")
            self.p_distribution = p_distribution
        else:
            self.p_distribution = np.ones(len(transformations))/(.0+len(transformations))

        if seed:
            np.random.RandomState(seed)

    @reshape()
    def perform(self, X):

        for i in xrange(X.shape[0]):
            # randomly pick one transformation according to probability distribution
            t = np.array(self.p_distribution).cumsum().searchsorted(np.random.sample(1))

            X[i] = self.transformations[t[0]].perform(X[i])

        return X

def gen_mm_random(f,args,min,max,max_iteration=50):
    if min==None or min=="None":
        min = -1e99

    if max==None or max=="None":
        max = 1e99

    i = 0
    r = f(**args)
    while (r < min or max < r) and i < max_iteration:
        r = f(**args)
        i += 1

    if i >= max_iteration:
        raise RuntimeError("Were not able to generate a random value between "+str(min)+" and "+str(max)+" in less than "+str(i)+" iterations")

    return r

class RandomTransformation:
    def __init__(self,p):
        self.p = p

    def perform(self,X):
        
        for i in xrange(X.shape[0]):
            p = np.random.random(1)[0]
            if p < self.p:
                X[i] = self.transform(X[i])

        return X

    def transform(self,X):
        return X

class Translation(RandomTransformation):
    def __init__(self,p=1,random_fct='normal',fct_settings=None,min=None,max=None):
        if not fct_settings:
            fct_settings = dict(loc=0.0,scale=2.0)
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        self.I = np.identity(3)

    def transform(self,x):

#        import matplotlib.pyplot as plt
#
#        print x[:,:,0].min()
#        print x[:,:,1].min()
#        print x[:,:,2].min()
#        print x[:,:,0].max()
#        print x[:,:,1].max()
#        print x[:,:,2].max()
#
##        plt.imshow(x.astype('uint8'))
#        plt.imshow(x)
#        plt.show()

        rx = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)
        ry = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)

#        x = ndimage.affine_transform(x.astype('uint8'),self.I,offset=(rx,ry,0))
        x = ndimage.affine_transform(x,self.I,offset=(rx,ry,0))

#        plt.imshow(x.astype('uint8'))
#        plt.imshow(x)
#        plt.show()
        return x

class Scaling(RandomTransformation):
    def __init__(self,p=1,random_fct='normal',fct_settings=None,min=0.1,max=None):
        if not fct_settings:
            fct_settings = dict(loc=1.0,scale=0.1)
 
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        pass

    def transform(self, x):

        zoom = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)
        shape = x.shape
        im = np.zeros(x.shape)
        # crop channels individually
        for c in xrange(x.shape[2]):
            im_c = ndimage.zoom(x[:,:,c],zoom)

            try:
                im[:,:,c] = self._crop(im_c,shape[:2])
            except ValueError as e:
                # won't crop so better return it already
                return x

        # balance channels

        return im

    def _crop(self,x,shape):
        """
            in 2D (x,y)
        """
        y = np.zeros(shape)


        if x.shape[0] < shape[0]:
            y[int((y.shape[0]-x.shape[0])/2.0+0.5):int(-(y.shape[0]-x.shape[0])/2.0),
              int((y.shape[1]-x.shape[1])/2.0+0.5):int(-(y.shape[1]-x.shape[1])/2.0)] = x
        else:
            y = x[int((x.shape[0]-y.shape[0])/2.0+0.5):int(-(x.shape[0]-y.shape[0])/2.0),
                  int((x.shape[1]-y.shape[1])/2.0+0.5):int(-(x.shape[1]-y.shape[1])/2.0)]

        return y

class Rotation(RandomTransformation):
    def __init__(self,p=1,random_fct='normal',fct_settings=None,min=None,max=None):
        if not fct_settings:
            fct_settings = dict(loc=0.0,scale=10.0)
 
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

        pass

    def transform(self,x):

        degree = gen_mm_random(np.random.__dict__[self.random_fct],self.fct_settings,self.min,self.max)

        return ndimage.rotate(x,degree,reshape=False)

class SaveTransform(TrainExtension):
    """
        Save a png of the transform every i epoch
    """
    def __init__(self, freq, save_path, rows=20, cols=20, rescale='global'):
        """
            freq: how often a png is saved
            save_path: name of a folder to save images, created if does not exists
        """
 
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path + "/examples_epoch%04d.png"
        self.count = 0

    def on_monitor(self, model, dataset, algorithm):

        self.count += 1
        if self.count % self.freq != 0:
            return None

        import matplotlib.pyplot as plt

        if self.rescale == 'none':
            global_rescale = False
            patch_rescale = False
        elif self.rescale == 'global':
            global_rescale = True
            patch_rescale = False
        elif self.rescale == 'individual':
            global_rescale = False
            patch_rescale = True
        else:
            assert False

        # implement saving from show_examples
        examples = dataset.get_batch_design(self.rows*self.cols)
        X = examples
        input_space = dataset.transformer.input_space
        if input_space:
            # needs reshaping
            if len(input_space.axes) != len(X.shape):
                X = X.reshape([X.shape[0]]+list(input_space.shape)+[input_space.num_channels])

                # dimension transposition
            else:
                # How can we detect axes of X?
                X = input_space.convert_numpy(X,input_space.axes,default)

        examples = X#.astype('float32')

        if global_rescale:
            examples /= np.abs(examples).max()

        if examples.shape[3] == 1:
            is_color = False
        elif examples.shape[3] == 3:
            is_color = True

        pv = patch_viewer.PatchViewer( (self.rows, self.cols), examples.shape[1:3], is_color=is_color)
        for i in xrange(self.rows*self.cols):
            pv.add_patch(examples[i,:,:,:], activation = 0.0, rescale = patch_rescale)

        pv.save(self.save_path % self.count)

        print "dataset image saved"

class VanishingNoise(TrainExtension):
    """
        Makes the noise smaller over epochs
    """
    def __init__(self, start, var, noiseClass, channel_name=None, half_life= None, bezier=False,
                 bezier1=True,bezier_count=100):
        """
            start: the epoch on which to start shrinking the learning rate
            var:   the variable name to vanish
            noiseClass: the noise class that needs to vanish
            channel_name: 
            half_life: how many epochs after start it will take for the learning rate
                       to lose half its value for the first time
                        (to lose the next half of its value will take twice
                        as long)
        """
 
        self.__dict__.update(locals())
        del self.self
        self._initialized = False
        self._count = 0
        assert start >= 0
        if half_life is None:
            self.half_life = start + 1
        else:
            assert half_life > 0

        if not self.channel_name:
            self.channel_name = self.var+"_vanishing_noise"

        assert isinstance(noiseClass,type(Denoising)), "noiseClass is not a class definition"

    def _get_noiseClass(self, transformer):
        if (isinstance(transformer,TransformationPipeline) or 
            isinstance(transformer,TransformationPool)):
            c = None
            for transformation in transformer.transformations:
                try:
                    c = self._get_noiseClass(transformation)
                except RuntimeError:
                    c = None

                if isinstance(c,self.noiseClass):
                    return c

        elif isinstance(transformer,self.noiseClass):
            return transformer

        raise RuntimeError("The noise class %s is not present in the dataset" % str(self.noiseClass))

    def on_monitor(self, model, dataset, algorithm):
        noiseClass = self._get_noiseClass(dataset.transformer)
        
        if not self._initialized:
            monitor = Monitor.get_monitor(model)
            monitor.add_channel(name=self.channel_name,
                                ipt=model.get_input_space().make_theano_batch(),
                                val=noiseClass.__dict__[self.var],
                                dataset=dataset)

            self._init_val = noiseClass.__dict__[self.var].get_value()
            self._initialized = True

            if self.bezier:
                if self.bezier1:
                    points = np.array([(0,0), (1.,0.),(1.,0.),(1,1)])
                else:
                    points = np.array([(0,0), (.45,.13),(.62,.64),(1,1)])
                xvals, yvals = bezier_curve(points,self.bezier_count*100)
                xvals = xvals*self.bezier_count
                yvals = yvals*self._init_val

                xis = []
                r_xvals = np.array(list(reversed(xvals)))

                for i in range(self.bezier_count):
                    xis.append(r_xvals.searchsorted(i)) 
                xis.append(yvals.shape[0]-1)
                                                    
                xvals = np.array(list(reversed(xvals)))
                yvals = np.array(list(reversed(yvals)))
                                                                
                xs = xvals[xis]
                self.bezier_ys = list(reversed(yvals[xis]))

        self._count += 1

        noiseClass.__dict__[self.var].set_value(np.cast[config.floatX](self.current_val()))

    def current_val(self):
        if self.bezier:
            return self.bezier_ys[self._count] if self._count < len(self.bezier_ys) else self.bezier_ys[-1]
        else:
            if self._count < self.start:
                scale = 1
            else:
                scale = float(self.half_life) / float(self._count - self.start +self.half_life)

            return self._init_val * scale

class GaussianNoise(RandomTransformation):
    def __init__(self,p,sigma=1.0):
        """
            Sigma: from 1.0 to 5.0
        """
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        return ndimage.gaussian_filter(x,self.sigma)

class Sharpening(RandomTransformation):
    def __init__(self,p,sigma=[3,1],alpha=30):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        blurred_l = ndimage.gaussian_filter(x, int(self.sigma[0]))

        filter_blurred_l = ndimage.gaussian_filter(blurred_l, int(self.sigma[1]))
        return blurred_l + int(self.alpha) * (blurred_l - filter_blurred_l)

class Noise(RandomTransformation):
    def __init__(self,p,alpha=1.0):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())

    def transform(self,x):
        return x + self.alpha.get_value()*x.std()*np.random.random(x.shape)

class Denoising(RandomTransformation):
    def __init__(self,p,sigma=[2,3],alpha=0.4):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):
        noisy = x + self.alpha * x.std() * np.random.random(x.shape)

        gauss_denoised = ndimage.gaussian_filter(noisy, self.sigma[0])

        return ndimage.median_filter(noisy, self.sigma[1])

class Flipping(RandomTransformation):
    def __init__(self,p=0.25):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,x):

        return np.fliplr(x)
      
class Occlusion:

    # Random function given as a parameter controls the size of the boxes, not their positions
    # Boxes potisions are sampled with numpy.random.random_sample
 
    def __init__(self,p=0.05,nb=10,random_fct='uniform',fct_settings=None,min=2,max=None):
        if not fct_settings:
            fct_settings = dict(low=2.0,high=20.0)
            
        self.__dict__.update(locals())
        pass

    def perform(self,X):
        
        for i in xrange(X.shape[0]):
            X[i] = self.transform(X[i])

        return X

    def transform(self,image):
        im = image.copy()
        nb = np.sum(np.random.random(self.nb)<self.p)
        for i in xrange(nb):

            dx = int(gen_mm_random(np.random.__dict__[self.random_fct],
                               self.fct_settings,
                               self.min,image.shape[0]-2))
            dy = int(gen_mm_random(np.random.__dict__[self.random_fct],
                               self.fct_settings,
                               self.min,image.shape[1]-2))

            x = int(gen_mm_random(np.random.uniform,dict(low=0,high=image.shape[0]-dx),
                                  0,image.shape[0]-dx))

            y = int(gen_mm_random(np.random.uniform,dict(low=0,high=image.shape[0]-dy),
                                  0,image.shape[1]-dy))

            im[x:(x+dx),y:(y+dy)] = np.zeros((dx,dy,image.shape[2])).astype(int)

        return im

class HalfFace(RandomTransformation):
    def __init__(self,p=0.05):
        RandomTransformation.__init__(self,p)
        self.__dict__.update(locals())
        pass

    def transform(self,image):
        x_center = image.shape[0]/2
        y_center = image.shape[1]/2
        if np.random.rand(1) >= 0.5:
            image[:,:y_center] = np.zeros(image[:,:y_center].shape)#(np.random.rand(*image[:x_center,:y_center].shape)*255).astype(int)
        else:
            image[:,y_center:] = np.zeros(image[:,:y_center].shape)#(np.random.rand(*image[:,y_center:].shape)*255).astype(int)
        return image

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    d = np.load('/data/afew/facetubes/sample.npy')
    print d.shape

    plt.imshow(d)
    plt.show()

    # translation
    t = Translation(p=1,random_fct='normal',fct_settings=dict(loc=0.0,scale=10.0),min=None,max=None)
    plt.imshow(t.transform(d))
    plt.show()

    # scale
    s = Scaling(1,random_fct='normal',fct_settings=dict(loc=1.0,scale=0.2),min=0.1,max=None)
    i = s.transform(d).astype('uint8')
    plt.imshow(i)
    plt.show()

    # rotation
    r = Rotation(p=1,random_fct='normal',fct_settings=dict(loc=0.0,scale=20.0),min=None,max=None)
    i = r.transform(d)
    plt.imshow(i)
    plt.show()

    # occlusion
    o = Occlusion(p=0.5,nb=10,random_fct='uniform',fct_settings=None,min=2,max=None)
    i = o.transform(d)
    plt.imshow(i)
    plt.show()

    # flip
    f = Flipping(p=1.0)
    i = f.transform(d)
    plt.imshow(i)
    plt.show()

    # all 
    i = f.transform(r.transform(o.transform(s.transform(t.transform(d)).astype('uint8'))))
    plt.imshow(i)
    plt.show()
