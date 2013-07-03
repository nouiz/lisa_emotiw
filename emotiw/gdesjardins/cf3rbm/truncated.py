import numpy
import theano
import theano.tensor as T
from theano.printing import Print
floatX = theano.config.floatX

SQRT2 = numpy.cast[floatX](numpy.sqrt(2))
def truncated_normal(size, avg, std, lbound, ubound, theano_rng, dtype):

    def phi(x):
        erfarg = (x - avg) / (std * SQRT2)
        rval = 0.5 * (1. + T.erf(erfarg))
        return rval.astype(dtype)
    
    def phi_inv(phi_x):
        erfinv_input = T.clip(2. * phi_x - 1., -1.+1e-6, 1.-1e-6)
        rval = avg + std * SQRT2 * T.erfinv(erfinv_input)
        return rval.astype(dtype)

    # center lower and upper bounds based on mean
    u = theano_rng.uniform(size=size, dtype=dtype)

    cdf_range = phi(ubound) - phi(lbound)
    sample = phi_inv(phi(lbound) + u * cdf_range)

    # if avg >> ubound, return ubound
    # if avg << lbound, return lbound
    # else return phi(lbound) + u * [phi(ubound) - phi(lbound)]
    rval = T.switch(
                T.or_(sample < lbound, sample > ubound),
                T.switch(avg >= ubound, ubound, lbound),
                sample)

    return rval
