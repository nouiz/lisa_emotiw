import numpy as np

def apply(f,data,batch_size):
    results = []
    for idx in xrange(0,data.shape[0],batch_size):
        if idx+batch_size > data.shape[0]:
            batch = np.cast[data.dtype](np.zeros([batch_size]+list(data.shape[1:])))
            batch[:data.shape[0]-idx] = data[idx:data.shape[0]]
        else:
            batch = data[idx:idx+batch_size]
        results.append(f(batch))

    results = np.concatenate(results,0)
    # cut-off extra 0s
    results = results[:data.shape[0]]
        
    return results
