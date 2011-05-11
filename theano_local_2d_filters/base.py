"""
"""
import theano
from theano import tensor

"""
Credits for C code: Alex Krizhevsky
 *  Created on: 2010-11-06
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
"""

class Op(theano.gof.Op):
    def _properties(self):
        # return hashable tuple of self.* attributes
        raise NotImplementedError('override-me')
    def __eq__(self, other):
        return (type(self) == type(other)
                and self._properties() == other._properties())
    def __hash__(self):
        return hash((type(self), self._properties()))

    def apply(self, inputs, outputs):
        return theano.gof.Apply(self, inputs, outputs)

class CpuImgActs(Op):

    def make_node(self, x, y):
        pass

    def c_code(self, node, inputs, outputs):
        hid

    def c_support_code(self, node, nodename):
        return """
/*
 * hidActs:     (numModules, numFilters, numImages) if hidActsOrder == MODULE_FILTER_IMAGE
 *              (numFilters, numModules, numImages) otherwise
 * filters:     (numModules, numColors, filterPixels, numFilters)
 *
 * targets:     (numColors, imgPixels, numImages)
 */
void %(nodename)s_cpuImgActs(float* hidActs, float* filters, float* targets,
               int numModulesX,  int numImages,  int numFilters,
               int filterSize,  int imgSize,  int moduleStart,
               int moduleStride,  int numColors, FILTER_OUTPUT_ORDER hidActsOrder) {
    int filterPixles = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numModules = numModulesX * numModulesX;
    for (int py = 0; py < imgSize; py++) {
        for (int px = 0; px < imgSize; px++) {
            for (int my = 0; my < numModulesX; my++) {
                int moduleTop = moduleStart + my * moduleStride;
                int moduleBottom = moduleTop + filterSize;
                for (int mx = 0; mx < numModulesX; mx++) {
                    int m = my * numModulesX + mx;
                    int moduleLeft = moduleStart + mx * moduleStride;
                    int moduleRight = moduleLeft + filterSize;
                    int pixInModuleX = px - moduleLeft;
                    int pixInModuleY = py - moduleTop;
                    int pixInModule = pixInModuleY * filterSize + pixInModuleX;
                    if (py >= moduleTop && py < moduleBottom && px >= moduleLeft && px < moduleRight) {
                        for (int f = 0; f < numFilters; f++) {
                            for (int i = 0; i < numImages; i++) {
                                for (int c = 0; c < numColors; c++) {
                                    float w = filters[(m*numColors+c) * numFilters * filterPixles + pixInModule * numFilters + f];
                                    float h = hidActsOrder == MODULE_FILTER_IMAGE 
                                            ? hidActs[m * numImages * numFilters + f * numImages + i]
                                            : hidActs[m * numImages + f * numModules * numImages + i];
                                    targets[c * imgPixels * numImages + i] += w * h;
                                }
                            }

                        }
                    }
                }
            }
            targets += numImages;
        }
    }
}
        """ %{nodename:nodename}


class CpuFilterActs(Op):
    """Apply local filters to a stack of images.

    The formats are
        images: colors x irows x icols x nimages (c-contiguous)
        filters: mrows x mcols x colors x frows x fcols x nfilters (c-contiguous)

    There are two possible output orders:
    If self.output_order == 'MODULE_FILTER_IMAGE', then
        rval: mrows x mcols x nfilters x nimages (c-contiguous)

    Elif self.output_order == 'FILTER_MODULE_IMAGE', then
        rval: nfilters x mrows x mcols x nimages (c-contiguous)

    Images are expected to be contiguous in the 2dimages dimension.
    That is to say adjacent floats in memory should usually correspond to the
    same colour channel of the same image location, but of consecutive images.
    """
    def __init__(self, output_order, paddingStart, moduleStride):
        assert output_order in ('MODULE_FILTER_IMAGE', 'FILTER_MODULE_IMAGE')
        self.output_order = output_order
        self.paddingStart = paddingStart
        self.moduleStride = moduleStride
    def _properties(self):
        return (self.output_order,
                self.paddingStart,
                self.moduleStride,
                )
    def make_node(self, images, filters):
        images_ = tensor.as_tensor_variable(images)
        filters_ = tensor.as_tensor_variable(filters)
        if images_.dtype != 'float32': raise TypeError(images)
        if filters_.dtype != 'float32': raise TypeError(filters)
        if images_.ndim != 4: raise TypeError(images)
        if filters_.ndim != 6: raise TypeError(filters)
        return self.apply([images_,filters_], [images_.type()])

    def c_code(self, node, name, inputs, outputs, sub):
        images, filters = inputs
        output_order = self.output_order
        z, = outputs
        fail = sub['fail']
        paddingStart = self.paddingStart
        moduleStride = self.moduleStride

        src = """
        int icolors, irows, icols, n_images;
        int mrows, mcols, fcolors, frows, fcols, n_filters;
        int paddingStart = %(paddingStart)s;
        int moduleStride = %(moduleStride)s;
        npy_intp zdims[4];
        if (%(images)s->nd != 4) { PyErr_Format(PyExc_TypeError, "transforms wrong rank"); %(fail)s; }
        if (%(filters)s->nd != 6) { PyErr_Format(PyExc_TypeError, "imgs wrong rank"); %(fail)s; }

        //image dimensions
        icolors = %(images)s->dimensions[0];
        irows = %(images)s->dimensions[1];
        icols = %(images)s->dimensions[2];
        n_images = %(images)s->dimensions[3];

        //filter dimensions
        mrows = %(filters)s->dimensions[0];
        mcols = %(filters)s->dimensions[1];
        fcolors = %(filters)s->dimensions[2];
        frows = %(filters)s->dimensions[3];
        fcols = %(filters)s->dimensions[4];
        n_filters = %(filters)s->dimensions[5];

        if (irows != icols){ PyErr_Format(PyExc_NotImplementedError, "imgs must be square"); %(fail)s; }
        if (mrows != mcols){ PyErr_Format(PyExc_NotImplementedError, "module grid must be square"); %(fail)s; }
        if (frows != fcols){ PyErr_Format(PyExc_NotImplementedError, "filters must be square"); %(fail)s; }

        if (%(output_order)s == MODULE_FILTER_IMAGE){
            zdims[0] = mrows;
            zdims[1] = mcols;
            zdims[2] = n_filters;
            zdims[3] = n_images;
        }
        else
        {
            zdims[0] = n_filters;
            zdims[1] = mrows;
            zdims[2] = mcols;
            zdims[3] = n_images;
        }

        if ((NULL == %(z)s)
            || ((%(z)s->dimensions)[0] != zdims[0])
            || ((%(z)s->dimensions)[1] != zdims[1])
            || ((%(z)s->dimensions)[2] != zdims[2])
            || ((%(z)s->dimensions)[3] != zdims[3]))
        {
            Py_XDECREF(%(z)s);
            if (!(%(z)s = (PyArrayObject*)PyArray_SimpleNew(4, zdims, type_num_%(images)s)))
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc output");
                %(fail)s;
            }
        }
        { // NESTED SCOPE
            PyArrayObject * c_images = PyArray_GETCONTIGUOUS(%(images)s);
            PyArrayObject * c_filters = PyArray_GETCONTIGUOUS(%(filters)s);
            assert (NPY_CARRAY & %(z)s->flags);

            assert (c_images);
            assert (c_filters);
            %(name)s_cpuFilterActs(
                (float*)c_images->data,
                (float*)c_filters->data,
                (float*)%(z)s->data,
                n_images, n_filters,
                icols, fcols, paddingStart,
                moduleStride, mcols, icolors, %(output_order)s);
            Py_DECREF(c_images);
            Py_DECREF(c_filters);

        } // END NESTED SCOPE
        """

        return src % locals()

    def c_support_code_apply(self, node, nodename):
        return """
#ifndef MFI_H_THING
#define MFI_H_THING
        enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};
#endif
        /*
         * images:      (numColors, imgPixels, numImages) with stride given
         * filters:     (numModules, numColors, filterPixels, numFilters)
         *
         * targets:     (numModules, numFilters, numImages) if foo == MODULE_FILTER_IMAGE
         *              (numFilters, numModules, numImages) otherwise
         */
        void %(nodename)s_cpuFilterActs(float* images, float* filters, float* targets,
                               int numImages, int numFilters,
                               int imgSize, int filterSize, int paddingStart,
                               int moduleStride, int numModulesX, int numColors, FILTER_OUTPUT_ORDER targetsOrder) {
            int filterPixels = filterSize * filterSize;
            int numModules = numModulesX * numModulesX;
            int imgPixels = imgSize * imgSize;
            for (int my = 0; my < numModulesX; my++) {
                int mStartY = paddingStart + my * moduleStride;
                for (int mx = 0; mx < numModulesX; mx++) {
                    int mStartX = paddingStart + mx * moduleStride;
                    int m = (my * numModulesX + mx);
                    for (int f = 0; f < numFilters; f++) {
                        for (int i = 0; i < numImages; i++) {
                            float prod = 0;
                            for (int c = 0; c < numColors; c++) {
                                for (int y = 0; y < filterSize; y++) {
                                    for (int x = 0; x < filterSize; x++) {
                                        float imgVal = mStartY + y >= 0 && mStartY + y < imgSize && mStartX + x >= 0 && mStartX + x < imgSize
                                                    ? images[c * imgPixels * numImages + i + ((mStartY+y) * imgSize + mStartX+x) * numImages]
                                                    : 0;
                                        prod += filters[c * filterPixels * numFilters + m * numFilters * filterPixels * numColors + f + (y * filterSize + x) * numFilters] * imgVal;
                                    }
                                }
                            }
                            if (targetsOrder == MODULE_FILTER_IMAGE) {
                                targets[m * numFilters * numImages + f * numImages + i] = prod;
                            } else {
                                targets[f * numModules * numImages + m * numImages + i] = prod;
                            }
        //                    targets++;
                        }
                    }
                }
            }
        }
        """%locals()
