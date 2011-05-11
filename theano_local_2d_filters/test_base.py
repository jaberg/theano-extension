import numpy
import base
import theano

tensor4 = theano.tensor.tensor4
def tensor6( dtype = theano.config.floatX):
    ttype = theano.tensor.TensorType(
            dtype=dtype,
            broadcastable=[False]*6)
    return ttype()


def test_basic():

    images = tensor4()
    filters = tensor6()

    irows = icols = 8
    icolors = 3
    n_images = 2
    n_filters = 5
    frows = fcols = 5
    mrows = mcols = 6

    op_mfi = base.CpuFilterActs('MODULE_FILTER_IMAGE', 0, 4)
    op_fmi = base.CpuFilterActs('FILTER_MODULE_IMAGE', 0, 4)

    f = theano.function([images, filters], [op_mfi(images, filters), op_fmi(images, filters)])

    idata = numpy.random.rand(icolors, irows, icols, n_images).astype(images.dtype)
    fdata = numpy.random.rand(mrows, mcols, icolors, frows, fcols, n_filters).astype(filters.dtype)

    zmfi, zfmi = f(idata, fdata)

    assert numpy.all(zmfi == zfmi.transpose(1, 2, 0, 3))


def test_noncontig_args():
    raise NotImplementedError()

