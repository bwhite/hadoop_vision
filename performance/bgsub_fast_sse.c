void bgsub_accum(unsigned int *image, int size, unsigned int *s, unsigned int *ss) {
  int i;
  for (i = 0; i < size; i++) {
    s[i] += image[i];
    ss[i] += image[i] * image[i];
  }  
}

void bgsub_mean_var(int size, float *s, float *ss, int c, float *m, float *v) {
  float inv_c_sqr = 6.25 / (c * c);
  float inv_c = 1. / c;
  int i;
  for (i = 0; i < size; ++i) {
    m[i] = s[i] * inv_c;
    v[i] = (ss[i] * c - s[i] * s[i]) * inv_c_sqr;
  }
}

void bgsub_classify(float *image, int size, float *m, float *v, unsigned char *bgsub) {
  int i;
  float val;
  for (i = 0; i < size; ++i) {
    bgsub[i] = (image[i] - m[i]) * (image[i] - m[i]) > v[i];
  }  
}
