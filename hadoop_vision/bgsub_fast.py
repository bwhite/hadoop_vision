import numpy as np
import ctypes

# Required Types
_uint8_ptr = ctypes.POINTER(ctypes.c_uint8)
_uint32_ptr = ctypes.POINTER(ctypes.c_uint32)
_double_ptr = ctypes.POINTER(ctypes.c_double)
_int = ctypes.c_int32

# Load library
_bg = np.ctypeslib.load_library('libbgsub_fast', '.')

#void bgsub_accum(unsigned char *image, int size, unsigned int *s, unsigned int *ss) {
_bg.bgsub_accum.restype = ctypes.c_int
_bg.bgsub_accum.argtypes = [_uint8_ptr, _int, _uint32_ptr, _uint32_ptr]
def accum(image, s, ss):
    _bg.bgsub_accum(image.ctypes.data_as(_uint8_ptr),
                    len(image),
                    s.ctypes.data_as(_uint32_ptr),
                    ss.ctypes.data_as(_uint32_ptr))


#void bgsub_mean_var(int size, unsigned int *s, unsigned int *ss, int c, double *m, double *v)
_bg.bgsub_mean_var.restype = ctypes.c_int
_bg.bgsub_mean_var.argtypes = [_int, _uint32_ptr, _uint32_ptr, _int, _double_ptr, _double_ptr]
def mean_var(s, ss, c, m, v):
    _bg.bgsub_mean_var(len(s),
                    s.ctypes.data_as(_uint32_ptr),
                    ss.ctypes.data_as(_uint32_ptr),
                    c,
                    m.ctypes.data_as(_double_ptr),
                    v.ctypes.data_as(_double_ptr))



#void bgsub_classify(unsigned char *image, int size, double *m, double *v, unsigned char *bgsub)
_bg.bgsub_classify.restype = ctypes.c_int
_bg.bgsub_classify.argtypes = [_uint8_ptr, _int, _double_ptr, _double_ptr, _uint8_ptr]
def classify(image, m, v, fg):
    _bg.bgsub_classify(image.ctypes.data_as(_uint8_ptr),
                       len(image),
                       m.ctypes.data_as(_double_ptr),
                       v.ctypes.data_as(_double_ptr),
                       fg.ctypes.data_as(_uint8_ptr))


if __name__ == '__main__':
    image = np.array([1,2,3], dtype=np.uint8)
    # accum
    s = np.zeros(image.shape, dtype=np.uint32)
    ss = np.zeros(image.shape, dtype=np.uint32)
    accum(image, s, ss)
    print(s)
    print(ss)
