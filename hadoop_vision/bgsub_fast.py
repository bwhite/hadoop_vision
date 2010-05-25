import numpy as np
import ctypes

# Required Types
_uint8_ptr = ctypes.POINTER(ctypes.c_uint8)
_uint32_ptr = ctypes.POINTER(ctypes.c_uint32)
_float_ptr = ctypes.POINTER(ctypes.c_float)
_int = ctypes.c_int32

# Load library
_bg = np.ctypeslib.load_library('libbgsub_fast', '.')

#void bgsub_accum(unsigned int *image, int size, float *s, float *ss) {
_bg.bgsub_accum.restype = ctypes.c_int
_bg.bgsub_accum.argtypes = [ctypes.c_char_p, _int, _float_ptr, _float_ptr]
def accum(image, s, ss):
    _bg.bgsub_accum(image,
                    len(image),
                    s.ctypes.data_as(_float_ptr),
                    ss.ctypes.data_as(_float_ptr))


#void bgsub_mean_var(int size, float *s, float *ss, int c, float *m, float *v)
_bg.bgsub_mean_var.restype = ctypes.c_int
_bg.bgsub_mean_var.argtypes = [_int, _float_ptr, _float_ptr, _int, _float_ptr, _float_ptr]
def mean_var(s, ss, c, m, v):
    _bg.bgsub_mean_var(len(s),
                    s.ctypes.data_as(_float_ptr),
                    ss.ctypes.data_as(_float_ptr),
                    c,
                    m.ctypes.data_as(_float_ptr),
                    v.ctypes.data_as(_float_ptr))



#void bgsub_classify(unsigned char *image, int size, float *m, float *bgsub)
_bg.bgsub_classify.restype = ctypes.c_int
_bg.bgsub_classify.argtypes = [ctypes.c_char_p, _int, _float_ptr, _float_ptr]
def classify(image, m, v, fg):
    _bg.bgsub_classify(image,
                       len(image),
                       m.ctypes.data_as(_float_ptr),
                       fg.ctypes.data_as(_float_ptr))
    return fg > v


if __name__ == '__main__':
    image = 'abc'
    # accum
    s = np.zeros(len(image), dtype=np.float32)
    ss = np.zeros(len(image), dtype=np.float32)
    accum(image, s, ss)
    print(s)
    print(ss)
