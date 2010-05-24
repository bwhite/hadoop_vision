void bgsub_accum(unsigned char *image, int size, unsigned int *s, unsigned int *ss) {
  int i;
  unsigned char val;
  for (i = 0; i < size; ++i, ++s, ++ss, ++image) {
    val = *image;
    *s += val;
    *ss += val * val;
  }  
}

void bgsub_mean_var(int size, unsigned int *s, unsigned int *ss, int c, double *m, double *v) {
  double inv_c_sqr = 6.25 / (c * c);
  double inv_c = 1. / c;
  int i;
  for (i = 0; i < size; ++i, ++s, ++ss, ++m, ++v) {
    *m = *s * inv_c;
    *v = (*ss * c - *s * *s) * inv_c_sqr;
  }
}

void bgsub_classify(unsigned char *image, int size, double *m, double *v, unsigned char *bgsub) {
  int i;
  double val;
  for (i = 0; i < size; ++i, ++m, ++v, ++image, ++bgsub) {
    val = *image - *m;
    *bgsub = val * val > (*v);
  }  
}
