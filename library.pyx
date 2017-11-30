# distutils: language = c++

import cython
cimport numpy as np
import numpy

ctypedef np.float32_t DTYPE_t
ctypedef np.float64_t DTYPE_64_t
ctypedef np.int32_t DTYPE_int_t
ctypedef np.int64_t DTYPE_int64_t
ctypedef np.uint32_t DTYPE_uint_t
ctypedef np.int8_t DTYPE_int8_t
cdef double Inf = numpy.inf

cdef extern from "math.h":
    double exp(double x) nogil
    double log(double x) nogil
    double log2(double x) nogil
    double log10(double x) nogil
    double sqrt(double x) nogil
    double pow(double x, double x) nogil
    double abs(double x) nogil
    double round(double x) nogil
    double floor(double x) nogil
    double ceil(double x) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def vertical_dynamic_binning(
        np.ndarray[DTYPE_t, ndim=3] data,
        np.ndarray[DTYPE_t, ndim=3] binned,
        np.ndarray[DTYPE_int_t, ndim=1] valid,
        int minreads):
    cdef int i, j, k, y0, y1, oob
    cdef int num_xbins = data.shape[0]
    cdef int num_ybins = data.shape[1]
    with nogil:
        for i in range(num_xbins):
            if valid[i] == 0:
                continue
            for j in range(num_ybins):
                if j == 0:
                    y0 = 0
                    y1 = 1
                elif j - y0 + 1 == y1:
                    y0 += 2
                else:
                    y0 = j
                    y1 = j + 1
                if valid[j] == 0:
                    continue
                for k in range(y0, y1):
                    binned[i, j, 0] += data[i, k, 0]
                    binned[i, j, 1] += data[i, k, 1]
                while binned[i, j, 0] < minreads:
                    y0 -= 1
                    y1 += 1
                    oob = 0
                    if y0 < 0:
                        y0 = 0
                        oob += 1
                    else:
                        binned[i, j, 0] += data[i, y0, 0]
                        binned[i, j, 1] += data[i, y0, 1]
                    if y1 <= num_ybins:
                        binned[i, j, 0] += data[i, y1, 0]
                        binned[i, j, 1] += data[i, y1, 1]
                    else:
                        y1 = num_ybins
                        oob += 1
                    if oob == 2:
                        break
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_binning_expected(
        np.ndarray[DTYPE_int_t, ndim=1] mapping not None,
        np.ndarray[DTYPE_t, ndim=1] binning_corrections not None,
        np.ndarray[DTYPE_int_t, ndim=1] binning_num_bins not None,
        np.ndarray[DTYPE_int_t, ndim=3] fend_indices not None,
        np.ndarray[DTYPE_int_t, ndim=1] mids not None,
        np.ndarray[DTYPE_t, ndim=2] parameters,
        np.ndarray[DTYPE_t, ndim=1] signal not None,
        double chrom_mean,
        int startfend,
        int startfend1,
        int stopfend1,
        int startfend2,
        int stopfend2,
        int startbin1,
        int stopbin1,
        int startbin2,
        int stopbin2,
        int diag):
    cdef long long int fend1, fend2, afend1, afend2, j, k, index, map1, map2
    cdef long long int start1, start2, stop1, stop2
    cdef double distance, value
    cdef long long int num_fends = mapping.shape[0]
    cdef int diag2 = diag * 2
    cdef long long int num_bins = int(0.5 + pow(0.25 + 2 * signal.shape[0], 0.5)) - diag
    cdef int num_parameters = fend_indices.shape[1]
    with nogil:
        for fend1 in range(startfend1, stopfend1):
            map1 = mapping[fend1]
            if map1 == -1:
                continue
            k = 0
            index = map1 * (num_bins - 1) - map1 * (map1 + 1 - diag2) / 2 - 1 + diag
            if map1 == startbin1:
                start2 = max(startfend2, fend1 + 2)
            else:
                start2 = fend1 + 2
            if map1 == stopbin1:
                stop2 = stopfend2
            else:
                stop2 = num_fends
            for fend2 in range(start2, stop2):
                map2 = mapping[fend2]
                if map2 < 0 or (diag == 0 and map2 == map1):
                    continue
                if fend2 - fend1 == 3 and fend1 % 2 == 0:
                    continue
                 # give starting expected value
                value = 1.0
                # if finding fend, enrichment, or expected, and using binning bias correction, correct for fend
                for j in range(num_parameters):
                    afend1 = fend1 + startfend
                    afend2 = fend2 + startfend
                    if fend_indices[afend1, j, 0] < fend_indices[afend2, j, 0]:
                        value *= binning_corrections[fend_indices[afend1, j, 1] + fend_indices[afend2, j, 0]]
                    else:
                        value *= binning_corrections[fend_indices[afend2, j, 1] + fend_indices[afend1, j, 0]]
                # if finding enrichment, correct for distance
                if not parameters is None:
                    distance = log(<double>(mids[fend2] - mids[fend1]))
                    while distance > parameters[k, 0]:
                        k += 1
                    value *= exp(distance * parameters[k, 1] + parameters[k, 2] + chrom_mean)
                signal[index + map2] += value
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_binning_expected2(
        np.ndarray[DTYPE_int_t, ndim=1] mapping not None,
        np.ndarray[DTYPE_t, ndim=1] binning_corrections not None,
        np.ndarray[DTYPE_int_t, ndim=1] binning_num_bins not None,
        np.ndarray[DTYPE_int_t, ndim=3] fend_indices not None,
        np.ndarray[DTYPE_t, ndim=2] parameters,
        np.ndarray[DTYPE_int_t, ndim=1] mids not None,
        np.ndarray[DTYPE_t, ndim=1] signal not None,
        np.ndarray[DTYPE_int_t, ndim=2] ranges not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices0 not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices1 not None,
        int startfend,
        double chrom_mean):
    cdef long long int fend1, fend2, afend1, afend2, i, j, k, bin1, bin2, map1, map2
    cdef double distance, value
    cdef int num_parameters = fend_indices.shape[1]
    cdef long long int num_bins = indices0.shape[0]
    with nogil:
        for i in range(num_bins):
            bin1 = indices0[i]
            bin2 = indices1[i]
            for fend1 in range(ranges[bin1, 0], ranges[bin1, 1]):
                map1 = mapping[fend1]
                if map1 == -1:
                    continue
                k = 0
                for fend2 in range(max(fend1 + 2, ranges[bin2, 0]), ranges[bin2, 1]):
                    map2 = mapping[fend2]
                    if map2 == -1:
                        continue
                    if fend2 - fend1 == 3 and (fend1 + startfend) % 2 == 0:
                        continue
                     # give starting expected value
                    value = 1.0
                    # if finding fend, enrichment, or expected, and using binning bias correction, correct for fend
                    for j in range(num_parameters):
                        afend1 = fend1 + startfend
                        afend2 = fend2 + startfend
                        if fend_indices[afend1, j, 0] < fend_indices[afend2, j, 0]:
                            value *= binning_corrections[fend_indices[afend1, j, 1] + fend_indices[afend2, j, 0]]
                        else:
                            value *= binning_corrections[fend_indices[afend2, j, 1] + fend_indices[afend1, j, 0]]
                    # if finding enrichment, correct for distance
                    if not parameters is None:
                        distance = log(<double>(mids[fend2] - mids[fend1]))
                        while distance > parameters[k, 0]:
                            k += 1
                        value *= exp(distance * parameters[k, 1] + parameters[k, 2] + chrom_mean)
                    signal[i] += value
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dynamically_bin(
        np.ndarray[DTYPE_int_t, ndim=2] unbinned_c not None,
        np.ndarray[DTYPE_t, ndim=2] unbinned_e not None,
        np.ndarray[DTYPE_int_t, ndim=2] binned_c not None,
        np.ndarray[DTYPE_t, ndim=2] binned_e not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=1] X_coords not None,
        np.ndarray[DTYPE_int_t, ndim=1] Y_coords not None,
        int minobservations,
        int ratio):
    cdef long long int X, Y, lX, elX, uX, euX, eX, lY, elY, uY, euY, eY, i, j, k, iteration
    cdef long long int num_bins = binned_c.shape[0]
    cdef long long int num_coords = X_coords.shape[0]
    with nogil:
        for i in range(num_coords):
            X = X_coords[i]
            Y = Y_coords[i]
            if valid[X] == 0 or valid[Y] == 0:
                continue
            if unbinned_c[X, Y] >= minobservations:
                binned_c[X, Y] = unbinned_c[X, Y]
                binned_c[Y, X] = unbinned_c[X, Y]
                binned_e[X, Y] = unbinned_e[X, Y]
                binned_e[Y, X] = unbinned_e[X, Y]
                lX = X - 1
                uX = X + 1
                lY = Y - 1
                uY = Y + 1
                continue
            elif (i > 0 and Y == Y_coords[i - 1] + 1 and X == X_coords[i - 1] and
                  binned_c[X, Y - 1] > 0 and lX - uX > 3):
                lX += 1
                uX -= 1
                lY += 2
            else:
                lX = X - 1
                uX = X + 1
                lY = Y - 1
                uY = Y + 1
            elX = max(lX, 0)
            euX = min(uX, num_bins - 1)
            elY = max(lY, 0)
            euY = min(uY, num_bins - 1)
            for j in range(max(lX + 1, 0), euX):
                for k in range(max(j, lY + 1), euY):
                    binned_c[X, Y] += unbinned_c[j, k]
                    binned_e[X, Y] += unbinned_e[j, k]
            iteration  = 1
            while binned_c[X, Y] < minobservations:
                if lX >= 0 and valid[lX] == 1:
                    for j in range(elY, euY + 1):
                        binned_c[X, Y] += unbinned_c[lX, j]
                        binned_e[X, Y] += unbinned_e[lX, j]
                if uY < num_bins and valid[uY] == 1:
                    for j in range(elX + 1, euX + 1):
                        binned_c[X, Y] += unbinned_c[j, uY]
                        binned_e[X, Y] += unbinned_e[j, uY]
                if uX < uY and uX < num_bins and valid[uX] == 1:
                    eY = max(uX, lY)
                    for j in range(eY, euY):
                        binned_c[X, Y] += unbinned_c[uX, j]
                        binned_e[X, Y] += unbinned_e[uX, j]
                if lY > lX and lY >= 0 and valid[lY] == 1:
                    eX = min(uX - 1, lY)
                    for j in range(elX + 1, eX + 1):
                        binned_c[X, Y] += unbinned_c[j, lY]
                        binned_e[X, Y] += unbinned_e[j, lY]
                lX -= 1
                elX = max(lX, 0)
                uX += 1
                euX = min(uX, num_bins - 1)
                if iteration % ratio == 0:
                    lY -= 1
                    elY = max(lY, 0)
                    uY += 1
                    euY = min(uY, num_bins - 1)
                iteration += 1
            binned_c[Y, X] = binned_c[X, Y]
            binned_e[Y, X] = binned_e[X, Y]
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_boundary_sizes(
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=2] b_sizes not None,
        np.ndarray[DTYPE_int_t, ndim=1] d_sizes not None,
        int M):
    cdef int i, j, X, Y
    cdef int N = valid.shape[0]
    with nogil:
        for i in range(N - 1):
            if valid[i] == 0:
                continue
            for j in range(i + 1, min(i + M, N)):
                if valid[j] == 0:
                    continue
                for X in range(max(0, j - M + 1), i + 1):
                    for Y in range(j, min(X + M, N)):
                        b_sizes[X, 0] += 1
                        b_sizes[Y, 1] += 1
                        d_sizes[Y - X] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_boundary_expected(
        np.ndarray[DTYPE_64_t, ndim=2] boundaries not None,
        np.ndarray[DTYPE_64_t, ndim=1] distance not None,
        np.ndarray[DTYPE_64_t, ndim=2] expected not None,
        int M):
    cdef int i, X, Y
    cdef int N = expected.shape[0]
    with nogil:
        for X in range(N - 1):
            for i in range(min(N, X + M - 1) - X - 1):
                Y = min(N, X + M - 1) - i - 1
                expected[X, Y] = (boundaries[X, 0] + boundaries[Y, 1]) * distance[Y - X]
                if Y - X < M - 1:
                    if X > 0:
                        expected[X, Y] += expected[X - 1, Y]
                    if i > 0:
                        expected[X, Y] += expected[X, Y + 1]
                    if Y < N and X > 0 and Y - X < M - 2:
                        expected[X, Y] -= expected[X - 1, Y + 1]
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def learn_boundary_gradients(
        np.ndarray[DTYPE_64_t, ndim=2] hm not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_64_t, ndim=2] expected not None,
        np.ndarray[DTYPE_64_t, ndim=2] boundaries not None,
        np.ndarray[DTYPE_64_t, ndim=1] distance not None,
        np.ndarray[DTYPE_64_t, ndim=2] b_gradients not None,
        np.ndarray[DTYPE_64_t, ndim=1] d_gradients not None):
    cdef int i, j, X, Y
    cdef double cost, temp, D
    cdef int N = hm.shape[0]
    cdef int M = distance.shape[0]
    with nogil:
        cost = 0.0
        for i in range(N - 1):
            # 0 <= i <= N - 2
            if valid[i] == 0:
                continue
            for j in range(i + 1, min(i + M, N)):
                # i + 1 <= j <= min(i + M - 1, N - 1)
                if valid[j] == 0:
                    continue
                temp = hm[i, j] - expected[i, j]
                temp *= temp
                cost += temp
                temp *= 2
                for X in range(max(0, j - M + 1), i + 1):
                    for Y in range(j, min(X + M, N)):
                        D = temp * distance[Y - X]
                        b_gradients[X, 0] -= D
                        b_gradients[Y, 1] -= D
                        #d_gradients[Y - X] -= temp * (boundaries[X, 0] + boundaries[Y, 1])
    return cost


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def update_boundary_gradients(
        np.ndarray[DTYPE_64_t, ndim=2] boundaries not None,
        np.ndarray[DTYPE_64_t, ndim=1] distance not None,
        np.ndarray[DTYPE_64_t, ndim=2] b_gradients not None,
        np.ndarray[DTYPE_64_t, ndim=1] d_gradients not None,
        np.ndarray[DTYPE_int_t, ndim=2] b_sizes not None,
        np.ndarray[DTYPE_int_t, ndim=1] d_sizes not None,
        double learning_rate):
    cdef int i
    cdef double change, temp
    cdef int N = boundaries.shape[0]
    cdef int M = distance.shape[0]
    with nogil:
        change = 0.0
        for i in range(N):
            temp = learning_rate * b_gradients[i, 0] / b_sizes[i, 0]
            if temp > 0:
                change = max(change, temp)
            else:
                change = max(change, -temp)
            boundaries[i, 0] = max(0, boundaries[i, 0] + temp)
            temp = learning_rate * b_gradients[i, 1] / b_sizes[i, 1]
            if temp > 0:
                change = max(change, temp)
            else:
                change = max(change, -temp)
            boundaries[i, 1] = max(0, boundaries[i, 1] + temp)
        """
        for i in range(M):
            temp = learning_rate * d_gradients[i] / d_sizes[i]
            if temp > 0:
                change = max(change, temp)
            else:
                change = max(change, -temp)
            distance[i] = max(0.000001, distance[i] + temp)
        """
    return change


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smooth_distance_curve(
        np.ndarray[DTYPE_int_t, ndim=1] counts not None,
        np.ndarray[DTYPE_t, ndim=1] expected not None,
        np.ndarray[DTYPE_t, ndim=1] distance not None,
        np.ndarray[DTYPE_t, ndim=2] smoothed not None,
        int minobs):
    cdef int i, j, start, stop, count
    cdef double expect, dist
    cdef int num_bins = counts.shape[0]
    with nogil:
        for i in range(num_bins):
            start = i
            stop = i
            count = counts[i]
            expect = expected[i]
            dist = distance[i]
            while count < minobs:
                start -= 1
                stop += 1
                if start >= 0:
                    count += counts[start]
                    expect += expected[start]
                    dist += distance[start]
                if stop < num_bins:
                    count += counts[stop]
                    expect += expected[stop]
                    dist += distance[stop]
            smoothed[i, 0] = dist / (min(num_bins - 1, stop) - max(start, 0) + 1)
            smoothed[i, 1] = log(count / expect)
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_distance_parameters(
        np.ndarray[DTYPE_int_t, ndim=2] counts not None,
        np.ndarray[DTYPE_t, ndim=2] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] where not None,
        np.ndarray[DTYPE_int_t, ndim=1] where2 not None,
        np.ndarray[DTYPE_int_t, ndim=2] bounds not None,
        np.ndarray[DTYPE_int_t, ndim=1] dcounts not None,
        np.ndarray[DTYPE_64_t, ndim=1] dexpected not None,
        np.ndarray[DTYPE_int_t, ndim=1] binsizes not None,
        int within):
    cdef int i, j, x, y, start, dist
    cdef int where_size = where.shape[0]
    cdef int where_size2 = where2.shape[0]
    with nogil:
        for i in range(where_size):
            if within == 1:
                start = i
            else:
                start = 0
            for j in range(start, where_size2):
                for x in range(bounds[where[i], 0], bounds[where[i], 1]):
                    for y in range(bounds[where2[j], 0], bounds[where2[j], 1]):
                        if expected[x, y] == 0.0:
                            continue
                        dist = y - x
                        if dist < 0:
                            dist = -dist
                        dcounts[dist] += counts[x, y]
                        dexpected[dist] += expected[x, y]
                        binsizes[dist] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_distance_parameters2(
        np.ndarray[DTYPE_int_t, ndim=2] counts not None,
        np.ndarray[DTYPE_t, ndim=2] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=2] bounds not None,
        np.ndarray[DTYPE_int_t, ndim=2] dcounts not None,
        np.ndarray[DTYPE_64_t, ndim=2] dexpected not None,
        np.ndarray[DTYPE_int_t, ndim=2] binsizes not None,
        int mindist):
    cdef int i, j, x, y, start, dist, index
    cdef int num_bounds = bounds.shape[0]
    with nogil:
        for i in range(num_bounds):
            for j in range(i, num_bounds):
                if bounds[i, 2] == bounds[j, 2]:
                    if bounds[i, 2] == 1:
                        index = 1
                    else:
                        index = 0
                else:
                    index = 2
                for x in range(bounds[i, 0], bounds[i, 1]):
                    if valid[x] == 0:
                        continue
                    if i == j:
                        start = x
                    else:
                        start = max(x + mindist, bounds[j, 0])
                    for y in range(start, bounds[j, 1]):
                        if valid[y] == 0:
                            continue
                        if expected[x, y] == 0.0:
                            continue
                        dist = y - x
                        dcounts[dist, index] += counts[x, y]
                        dexpected[dist, index] += expected[x, y]
                        binsizes[dist, index] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_distance_parameters3(
        np.ndarray[DTYPE_int_t, ndim=1] counts not None,
        np.ndarray[DTYPE_t, ndim=1] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=1] states not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices0 not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices1 not None,
        np.ndarray[DTYPE_int_t, ndim=2] dcounts not None,
        np.ndarray[DTYPE_64_t, ndim=2] dexpected not None,
        np.ndarray[DTYPE_int_t, ndim=2] binsizes not None):
    cdef long long int i, j, x, y, start, dist, index, bin1, bin2
    cdef long long int num_bins = indices0.shape[0]
    with nogil:
        for i in range(num_bins):
            if valid[i] <= 0:
                continue
            bin1 = indices0[i]
            bin2 = indices1[i]
            if states[bin1] == states[bin2]:
                if states[bin1] == 1:
                    index = 1
                else:
                    index = 0
            else:
                index = 2
            dist = bin2 - bin1
            dcounts[dist, index] += counts[i]
            dexpected[dist, index] += expected[i]
            binsizes[dist, index] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_distance_parameters_clusters(
        np.ndarray[DTYPE_int_t, ndim=1] counts not None,
        np.ndarray[DTYPE_t, ndim=1] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=1] states not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices0 not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices1 not None,
        np.ndarray[DTYPE_int_t, ndim=3] dcounts not None,
        np.ndarray[DTYPE_64_t, ndim=3] dexpected not None,
        np.ndarray[DTYPE_int_t, ndim=3] binsizes not None):
    cdef long long int i, j, x, y, start, dist, index, bin1, bin2
    cdef long long int num_bins = indices0.shape[0]
    with nogil:
        for i in range(num_bins):
            if valid[i] <= 0:
                continue
            bin1 = indices0[i]
            bin2 = indices1[i]
            dist = bin2 - bin1
            dcounts[dist, states[bin1], states[bin2]] += counts[i]
            dexpected[dist, states[bin1], states[bin2]] += expected[i]
            binsizes[dist, states[bin1], states[bin2]] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_distance_parameters4(
        np.ndarray[DTYPE_int_t, ndim=2] reads not None,
        np.ndarray[DTYPE_int_t, ndim=1] mids not None,
        np.ndarray[DTYPE_int_t, ndim=1] states not None,
        np.ndarray[DTYPE_int64_t, ndim=2] read_sums not None,
        np.ndarray[DTYPE_64_t, ndim=2] distance_sums not None,
        np.ndarray[DTYPE_int64_t, ndim=2] count_sums not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        long long int binsize,
        long long int start,
        long long int stop):
    cdef long long int i, X, Y, distance, state
    cdef double ln_distance
    cdef long long int num_reads = reads.shape[0]
    cdef long long int num_fends = mids.shape[0]
    with nogil:
        for i in range(num_reads):
            X = reads[i, 0]
            Y = reads[i, 1]
            if X == -1 or Y == -1:
                continue
            distance = (mids[Y] - mids[X]) / binsize
            state = states[X] + states[Y]
            read_sums[distance, state] += reads[i, 2]
        for X in range(start, stop):
            for Y in range(X + 1, min(num_fends, X + 4)):
                if valid[Y] - valid[X] > 3 or valid[Y] - valid[X] == 2:
                    distance = (mids[Y] - mids[X])
                    state = states[X] + states[Y]
                    ln_distance = log10(distance)
                    distance /= binsize
                    distance_sums[distance, state] += ln_distance
                    count_sums[distance, state] += 1
            for Y in range(X + 4, num_fends):
                distance = (mids[Y] - mids[X])
                state = states[X] + states[Y]
                ln_distance = log10(distance)
                distance /= binsize
                distance_sums[distance, state] += ln_distance
                count_sums[distance, state] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_valid_bins(
        np.ndarray[DTYPE_int_t, ndim=2] counts not None,
        np.ndarray[DTYPE_t, ndim=2] expected not None,
        np.ndarray[DTYPE_int_t, ndim=2] bounds not None,
        np.ndarray[DTYPE_int_t, ndim=2] positions not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        int minsize,
        int binsize):
    cdef int i, j, X, Y, mid1, mid2, mid_dist
    cdef int num_bounds = bounds.shape[0]
    with nogil:
        for i in range(num_bounds):
            mid1 = (positions[bounds[i, 1] - 1, 1] - positions[bounds[i, 0], 0]) / 2
            for j in range(num_bounds):
                mid2 = (positions[bounds[j, 1] - 1, 1] - positions[bounds[j, 0], 0]) / 2
                mid_dist = mid1 - mid2
                if mid_dist < 0:
                    mid_dist = -mid_dist
                if mid_dist < minsize:
                    continue
                for X in range(bounds[i, 0], bounds[i, 1]):
                    for Y in range(bounds[j, 0], bounds[j, 1]):
                        if counts[X, Y] > 0 and expected[X, Y] > 0:
                            valid[X] += 1
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optimize_compartment_bounds(
        np.ndarray[DTYPE_int_t, ndim=2] counts not None,
        np.ndarray[DTYPE_t, ndim=2] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=2] bounds not None,
        np.ndarray[DTYPE_int_t, ndim=2] positions not None,
        np.ndarray[DTYPE_64_t, ndim=1] paramsA not None,
        np.ndarray[DTYPE_64_t, ndim=1] paramsB not None,
        np.ndarray[DTYPE_64_t, ndim=1] paramsAB not None,
        np.ndarray[DTYPE_64_t, ndim=2] scores not None,
        int minsize):
    cdef int i, j, X, Y, mid1, mid2, mid_dist, dist
    cdef double A_prob, B_prob, muA, muB, lambdaA, lambdaB
    cdef int num_bounds = bounds.shape[0]
    cdef int num_bins = paramsA.shape[0]
    with nogil:
        for i in range(num_bounds):
            mid1 = (positions[bounds[i, 1] - 1, 1] - positions[bounds[i, 0], 0]) / 2
            for j in range(num_bounds):
                mid2 = (positions[bounds[j, 1] - 1, 1] - positions[bounds[j, 0], 0]) / 2
                mid_dist = mid1 - mid2
                #if mid_dist < 0:
                #    mid_dist = -mid_dist
                #if mid_dist < minsize:
                #    continue
                for X in range(bounds[i, 0], bounds[i, 1]):
                    if valid[X] < 1:
                        continue
                    for Y in range(bounds[j, 0], bounds[j, 1]):
                        if valid[Y] < 1:
                            continue
                        mid_dist = positions[X, 0] - positions[Y, 0]
                        if mid_dist < 0:
                            mid_dist = -mid_dist
                        if mid_dist < minsize:
                            continue
                        dist = X - Y
                        if dist < 0:
                            dist = -dist
                        if bounds[j, 2] == 1:
                            muA = paramsAB[dist]
                            muB = paramsB[dist]
                        else:
                            muA = paramsA[dist]
                            muB = paramsAB[dist]
                        lambdaA = muA * expected[X, Y]
                        lambdaB = muB * expected[X, Y]
                        scores[X, 1] += counts[X, Y] * log(lambdaA) - lambdaA
                        scores[X, 0] += counts[X, Y] * log(lambdaB) - lambdaB
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optimize_compartment_bounds2(
        np.ndarray[DTYPE_int_t, ndim=1] counts not None,
        np.ndarray[DTYPE_t, ndim=1] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=1] states not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices0 not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices1 not None,
        np.ndarray[DTYPE_64_t, ndim=1] paramsA not None,
        np.ndarray[DTYPE_64_t, ndim=1] paramsB not None,
        np.ndarray[DTYPE_64_t, ndim=1] paramsAB not None,
        np.ndarray[DTYPE_64_t, ndim=2] scores not None):
    cdef long long int i, X, Y, dist
    cdef double muA, muB, lambdaA, lambdaB, temp1, temp2
    cdef long long int num_indices = indices0.shape[0]
    with nogil:
        for i in range(num_indices):
            if valid[i] <= 0:
                continue
            X = indices0[i]
            Y = indices1[i]
            dist = Y - X
            if states[Y] == 1:
                muA = paramsAB[dist]
                muB = paramsB[dist]
            else:
                muA = paramsA[dist]
                muB = paramsAB[dist]
            lambdaA = muA * expected[i]
            lambdaB = muB * expected[i]
            temp1 = counts[i] * log(lambdaA) - lambdaA
            temp2 = counts[i] * log(lambdaB) - lambdaB
            scores[X, 1] += temp1
            scores[X, 0] += temp2
            if states[X] == states[Y]:
                scores[Y, 1] += temp1
                scores[Y, 0] += temp2
            else:
                if states[X] == 1:
                    muA = paramsAB[dist]
                    muB = paramsB[dist]
                else:
                    muA = paramsA[dist]
                    muB = paramsAB[dist]
                lambdaA = muA * expected[i]
                lambdaB = muB * expected[i]
                scores[Y, 1] += counts[i] * log(lambdaA) - lambdaA
                scores[Y, 0] += counts[i] * log(lambdaB) - lambdaB
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def optimize_compartment_bounds_clusters(
        np.ndarray[DTYPE_int_t, ndim=1] counts not None,
        np.ndarray[DTYPE_t, ndim=1] expected not None,
        np.ndarray[DTYPE_int_t, ndim=1] valid not None,
        np.ndarray[DTYPE_int_t, ndim=1] states not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices0 not None,
        np.ndarray[DTYPE_int_t, ndim=1] indices1 not None,
        np.ndarray[DTYPE_64_t, ndim=3] dparams not None,
        np.ndarray[DTYPE_64_t, ndim=2] scores not None):
    cdef long long int i, X, Y, dist
    cdef double mu, Lambda, score
    cdef long long int num_states = dparams.shape[1]
    cdef long long int num_indices = indices0.shape[0]
    with nogil:
        for i in range(num_indices):
            if valid[i] <= 0:
                continue
            X = indices0[i]
            Y = indices1[i]
            dist = Y - X
            if states[X] == states[Y]:
                for j in range(num_states):
                    Lambda = dparams[dist, states[X], j] * expected[i]
                    score = counts[i] * log(Lambda) - Lambda
                    scores[X, j] += score
                    scores[Y, j] += score
            else:
                for j in range(num_states):
                    Lambda = dparams[dist, states[X], j] * expected[i]
                    score = counts[i] * log(Lambda) - Lambda
                    scores[Y, j] += score
                    Lambda = dparams[dist, states[Y], j] * expected[i]
                    score = counts[i] * log(Lambda) - Lambda
                    scores[X, j] += score
    return None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def find_domain_scores(
        np.ndarray[DTYPE_int_t, ndim=2] counts not None,
        np.ndarray[DTYPE_int64_t, ndim=1] sums not None,
        np.ndarray[DTYPE_int64_t, ndim=2] scores not None,
        np.ndarray[DTYPE_int_t, ndim=2] paths not None,
        np.ndarray[DTYPE_int64_t, ndim=1] temp not None,
        ):
    cdef int i, j, k, temp2
    cdef int num_bins = counts.shape[0]
    with nogil:
        for i in range(num_bins):
            for j in range(i):
                temp[i - j] = temp[i - j - 1]
            temp[0] = 0
            temp2 = 0
            for j in range(i + 1):
                if counts[i, i - j] == 1:
                    temp2 += 1
                else:
                    temp2 -= 1
                temp[j] += temp2 + sums[i]
            scores[i, 0] = temp[i]
            paths[i, 0] = -1
            for j in range(1, i + 1):
                scores[i, j] = temp[i - j] + scores[j - 1, j - 1]
                paths[i, j] = j - 1
                for k in range(j + 1, i + 1):
                    temp2 = temp[i - k] + scores[k - 1, j - 1]
                    if temp2 > scores[i, j]:
                        scores[i, j] = temp2
                        paths[i, j] = k - 1
    return None



















