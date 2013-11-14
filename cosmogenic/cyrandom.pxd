from numpy cimport int64_t, uint8_t, uint64_t

from cosmogenic._random cimport RandomObject


cdef class Random:
    cdef RandomObject state

    cdef void seed(self, uint64_t n) nogil
    cdef double random(self) nogil
    cdef int getrandbits(self, int k, uint8_t [::1] output) nogil
    cdef int _getrandbits(self, int k, uint8_t *output, size_t length) nogil
    cdef uint64_t _randbelow(self, uint64_t n) nogil
    cdef int randrange(self, int64_t start, int64_t stop, int64_t step,
                       int64_t *output) nogil
    cdef int randint(self, int64_t a, int64_t b, int64_t *output) nogil
    cdef double uniform(self, double a, double b) nogil