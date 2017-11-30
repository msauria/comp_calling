#!/usr/bin/env python

import sys
import argparse

import numpy
import h5py
import hifive
from mpi4py import MPI
try:
    from pyx import *
    c = canvas.canvas()
    plot_results = True
except:
    plot_results = False
from scipy.sparse.linalg import eigs
import library

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()
numpy.seterr()


def main():
    if rank == 0:
        print >> sys.stderr, ("\r%s\rLoading data") % (' ' * 80),
        if plot_results:
            c = canvas.canvas()
        parser = generate_parser()
        args = parser.parse_args()
        args = comm.bcast(args, root=0)
        nres = int(numpy.ceil(numpy.log(args.min / args.max) / numpy.log(5))) + 1
        resolutions = numpy.zeros(nres, dtype=numpy.int32)
        resolutions[0] = args.min
        resolutions[-1] = args.max
        for i in range(1, nres - 1):
            resolutions[i] = resolutions[i - 1] / 5
        infile = h5py.File(args.input, 'r')
        hm_binsize = infile.attrs['binsize']
        N = int((0.25 + 2 * infile['%s.fend' % args.chrom].shape[0]) ** 0.5 - 0.5)
        indices = numpy.triu_indices(N, 0)
        raw_counts = numpy.zeros((N, N), dtype=numpy.int32)
        temp = infile['%s.counts' % args.chrom][...]
        raw_counts[temp[:, 0], temp[:, 1]] = temp[:, 2]
        del temp
        raw_counts[indices[1], indices[0]] = raw_counts[indices]
        raw_enrichment = numpy.zeros((N, N), dtype=numpy.float32)
        raw_enrichment[indices] = infile['%s.enrichment' % args.chrom][...]
        raw_enrichment[indices[1], indices[0]] = raw_enrichment[indices]
        raw_fend = numpy.zeros((N, N), dtype=numpy.float32)
        raw_fend[indices] = infile['%s.fend' % args.chrom][...]
        raw_fend[indices[1], indices[0]] = raw_fend[indices]
        raw_mids = numpy.arange(N).astype(numpy.int32) * hm_binsize + infile.attrs['%s.start' % args.chrom] + hm_binsize / 2
        print >> sys.stderr, ("\r%s\rInitial binning of data") % (' ' * 80),
        initial_counts, start = bin_data(raw_counts, raw_mids, args.min)
        initial_expected, start = bin_data(raw_enrichment, raw_mids, args.min)
        print >> sys.stderr, ("\r%s\rFinding correlations") % (' ' * 80),
        corr, valid_rows = find_correlations(initial_counts, initial_expected)
        corr2 = numpy.zeros((initial_counts.shape[0], initial_counts.shape[0]), dtype=numpy.float32)
        valid_rows2 = numpy.where(valid_rows)[0]
        for i, X in enumerate(valid_rows2):
            corr2[X, valid_rows2] = corr[i, :]
        if plot_results:
            c.insert(plot_heatmap(numpy.exp(corr2), corr2 != 0.0, None, None, initial_counts.shape[0], False),
                [trafo.translate(0, 0.5)])
        state_bounds, eigen = find_initial_states(corr, valid_rows, start, args.min)
        pos = numpy.zeros((eigen.shape[0], 2), dtype=numpy.int32)
        pos[:, 0] = start + numpy.arange(eigen.shape[0]) * args.min
        pos[:, 1] = start + numpy.arange(1, eigen.shape[0] + 1) * args.min
        hm_start = start
        hm_stop = pos[-1, 1]
        if plot_results:
            c.insert(plot_scores(pos, eigen, hm_start, hm_stop), [trafo.translate(0, 1 - resolutions.shape[0])])
    else:
        args = None
        args = comm.bcast(args, root=0)
        find_correlations()
        resolutions = None
    resolutions = comm.bcast(resolutions, root=0)
    for h, res in enumerate(resolutions[1:]):
        if rank == 0:
            print >> sys.stderr, ("\r%s\rBinning data to %iK") % (' ' * 80, res/1000),
            initial_counts, start = bin_data(raw_counts, raw_mids, res)
            initial_expected, start = bin_data(raw_fend, raw_mids, res)
            vrows = numpy.sum(initial_expected > 0, axis=1) > 0
            print >> sys.stderr, ("\r%s\rDistributing data") % (' ' * 80),
            counts, expected, vrows, indices0, indices1, N, start = distribute_data(
                initial_counts, initial_expected, vrows, args.mindist, res, start)
            states = initialize_states(N, state_bounds, start, res)
        else:
            counts, expected, vrows, indices0, indices1, N, start = distribute_data()
            states = initialize_states(N)
        stop = start + N * res
        positions, score_sums = optimize_bounds(counts, expected, states, indices0, indices1,
                                                start, stop, N, vrows, args.mindist / res)
        if rank == 0:
            if plot_results:
                c.insert(plot_scores(positions, score_sums, hm_start, hm_stop),
                     [trafo.translate(0, 2 + h -resolutions.shape[0])])
            state_bounds = find_state_bounds(score_sums, positions)
    if rank == 0:
        if plot_results:
            c.writePDFfile('%s.pdf' % (args.output))
        write_bg(positions, score_sums, args.chrom, "%s.bg" % (args.output))
        write_bed(state_bounds, args.chrom, "%s.bed" % (args.output))
    return None

def plot_distance_relationships(data, minsize):
    if rank > 0:
        return None
    X = []
    Y = []
    minX = numpy.inf
    maxX = -numpy.inf
    minY = numpy.inf
    maxY = -numpy.inf
    for i in range(len(data)):
        X.append(numpy.log(numpy.r_[0.5, numpy.arange(1, data[i].shape[0])].astype(numpy.float64)))
        Y.append(numpy.log(data[i]))
        minX = min(minX, numpy.amin(X[-1]))
        maxX = max(maxX, numpy.amax(X[-1]))
        minY = min(minY, numpy.amin(Y[-1]))
        maxY = max(maxY, numpy.amax(Y[-1]))
    c = canvas.canvas()
    c.stroke(path.rect(0, 0, 10, 5))
    grad = color.gradient.Rainbow
    for h in range(len(data)):
        col = grad.select(h, len(data))
        lpath = path.path(path.moveto((X[h][0] - minX) / (maxX - minX) * 10.,
                                      (Y[h][0] - minY) / (maxY - minY) * 5.))
        for i in range(1, X[h].shape[0]):
            lpath.append(path.lineto((X[h][i] - minX) / (maxX - minX) * 10.,
                                     (Y[h][i] - minY) / (maxY - minY) * 5.))
        c.stroke(lpath, [col])
    X = (numpy.log(minsize) - minX) / (maxX - minX) * 10.
    c.stroke(path.line(X, 0, X, 5.0))
    return c

def plot_heatmap(counts, expected, indices0, indices1, N, split=True):
    if rank == 0:
        c = canvas.canvas()
        hm = numpy.zeros((N, N, 2), dtype=numpy.float32)
        if split:
            hm[indices0, indices1, 0] = counts
            hm[indices0, indices1, 1] = expected
            for i in range(1, num_procs):
                counts = comm.recv(source=i, tag=11)
                expected = comm.recv(source=i, tag=11)
                indices0 = comm.recv(source=i, tag=11)
                indices1 = comm.recv(source=i, tag=11)
                hm[indices0, indices1, 0] = counts
                hm[indices0, indices1, 1] = expected
            hm += hm.transpose(1, 0, 2)
        else:
            hm[:, :, 0] = counts
            hm[:, :, 1] = expected
        img = hifive.plotting.plot_full_array(hm, symmetricscaling=False, silent=True)
        c.insert(bitmap.bitmap(0, 0, img, width=10))
        return c
    else:
        comm.send(counts, dest=0, tag=11)
        comm.send(expected, dest=0, tag=11)
        comm.send(indices0, dest=0, tag=11)
        comm.send(indices1, dest=0, tag=11)
    return None

def plot_scores(coords, scores, start, stop):
    if rank == 0:
        c = canvas.canvas()
        valid = numpy.where(numpy.logical_not(numpy.isnan(scores)))[0]
        signs = numpy.sign(scores[valid])
        breaks = numpy.where(signs[1:] != signs[:-1])[0] + 1
        mids = ((coords[valid[breaks], 0] + coords[valid[breaks - 1], 1]) / 2.0 - start) / float(stop - start) * 10.
        if signs[0] > 0:
            c.fill(path.rect(0, -0.5, mids[0], 1.0), [color.gray(0.7)])
        for i in range(breaks.shape[0] - 1):
            if signs[breaks[i]] > 0:
                c.fill(path.rect(mids[i], -0.5, mids[i + 1] - mids[i], 1.0), [color.gray(0.7)])
        if signs[-1] > 0:
            c.fill(path.rect(mids[-1], -0.5, 10.0 - mids[-1], 1.0), [color.gray(0.7)])
        temp = numpy.copy(numpy.abs(scores[valid]))
        temp.sort()
        cutoff = temp[int(0.95 * temp.shape[0])]
        scores2 = numpy.copy(scores)
        scores2[valid] = numpy.maximum(-cutoff, numpy.minimum(cutoff, scores2[valid]))
        scores2[valid] /= cutoff * 2
        lpath = path.path(path.moveto(0, 0))
        for i in valid:
            X = ((coords[i, 0] + coords[i, 1]) * 0.5 - start) / float(stop - start) * 10.
            lpath.append(path.lineto(X, scores2[i]))
        lpath.append(path.lineto(10., 0))
        lpath.append(path.closepath())
        c.fill(lpath)
        return c
    return None

def bin_data(raw, mids, binsize):
    start = (mids[0] / binsize) * binsize
    stop = ((mids[-1] - 1) / binsize + 1) * binsize
    mapping = ((mids - start) / binsize).astype(numpy.int64)
    indices = numpy.triu_indices(mids.shape[0], 0)
    M = (stop - start) / binsize
    index = mapping[indices[0]] * M + mapping[indices[1]]
    binned = numpy.bincount(index, weights=raw[indices], minlength=(M * M)).reshape(M, M).astype(raw.dtype)
    indices = numpy.triu_indices(M, 0)
    binned[indices[1], indices[0]] += binned[indices]
    return binned, start

def find_correlations(counts=None, expected=None):
    if rank == 0:
        valid_rows = numpy.sum(counts, axis=0) > 0
        valid_rows2 = numpy.where(valid_rows)[0]
        M = valid_rows2.shape[0]
        data = numpy.zeros((M, M), dtype=numpy.float64)
        data[:, :] = counts[valid_rows2, :][:, valid_rows2]
        where = numpy.where(data > 0)
        where2 = numpy.where(data == 0)
        data[where] /= expected[valid_rows2, :][:, valid_rows2][where]
        data[where] = numpy.log2(data[where])
        data[where2] = numpy.nan
        data[numpy.where(numpy.isnan(data))] = 0.0
        M = comm.bcast(M, root=0)
    else:
        M = 0
        M = comm.bcast(M, root=0)
        data = numpy.zeros((M, M), dtype=numpy.float64)
    comm.Bcast(data, root=0)
    R = M * (M - 1) / 2
    node_ranges = numpy.round(numpy.linspace(0, R, num_procs + 1)).astype(numpy.int64)
    indices = numpy.triu_indices(M, 1)
    index0 = indices[0][node_ranges[rank]:node_ranges[rank + 1]]
    index1 = indices[1][node_ranges[rank]:node_ranges[rank + 1]]
    corrs = numpy.zeros((M, M), dtype=numpy.float64)
    for i in range(index0.shape[0]):
        X = index0[i]
        Y = index1[i]
        where = numpy.where(numpy.logical_not(numpy.isnan(data[X, :])) &
                            numpy.logical_not(numpy.isnan(data[X, :])))[0]
        corrs[X, Y] = numpy.corrcoef(data[X, where], data[Y, where])[0, 1]
    if rank == 0:
        for i in range(1, num_procs):
            corrs[indices[0][node_ranges[i]:node_ranges[i + 1]],
                 indices[1][node_ranges[i]:node_ranges[i + 1]]] = comm.recv(source=i, tag=11)
        corrs[numpy.arange(M), numpy.arange(M)] = 1.0
        corrs[indices[1], indices[0]] = corrs[indices]
        return corrs, valid_rows
    else:
        comm.send(corrs[index0, index1], dest=0, tag=11)
        return None

def find_initial_states(corr, valid_rows, start, binsize):
    eigen = numpy.real(eigs(corr, k=1)[1][:, 0]).astype(numpy.float64)
    signs = numpy.maximum(0, numpy.sign(eigen))
    bounds = numpy.zeros((corr.shape[0], 2), dtype=numpy.int32)
    bounds[:, 0] = numpy.where(valid_rows)[0] * binsize + start
    bounds[:, 1] = bounds[:, 0] + binsize
    changes = numpy.where(signs[1:] != signs[:-1])[0]
    states = numpy.zeros((changes.shape[0] + 1, 3), dtype=numpy.int32)
    states[0, 0] = start
    states[-1, 1] = bounds[-1, 1]
    temp = (bounds[changes, 1] + bounds[changes + 1, 0]) / 2
    states[1:, 0] = temp
    states[:-1, 1] = temp
    states[:-1, 2] = signs[changes]
    states[-1, 2] = 1 - states[-2, 2]
    eigen2 = numpy.zeros(valid_rows.shape[0], dtype=numpy.float32)
    eigen2.fill(numpy.nan)
    eigen2[numpy.where(valid_rows)[0]] = eigen
    return states, eigen2

def distribute_data(initial_counts=None, initial_expected=None, vrows=None, mindist=None, binsize=None, start=None):
    if rank == 0:
        N = initial_counts.shape[0]
        indices = list(numpy.triu_indices(N, mindist / binsize))
        indices[0] = indices[0].astype(numpy.int32)
        indices[1] = indices[1].astype(numpy.int32)
        node_ranges = numpy.round(numpy.linspace(0, indices[0].shape[0], num_procs + 1)).astype(numpy.int64)
        N = comm.bcast(N, root=0)
        start = comm.bcast(start, root=0)
        comm.Bcast(vrows, root=0)
        for i in range(1, num_procs):
            comm.send(node_ranges[i + 1] - node_ranges[i], dest=i, tag=11)
            comm.Send(initial_counts[indices[0][node_ranges[i]:node_ranges[i + 1]],
                                     indices[1][node_ranges[i]:node_ranges[i + 1]]], dest=i, tag=12)
            comm.Send(initial_expected[indices[0][node_ranges[i]:node_ranges[i + 1]],
                                       indices[1][node_ranges[i]:node_ranges[i + 1]]], dest=i, tag=13)
            comm.Send(indices[0][node_ranges[i]:node_ranges[i + 1]], dest=i, tag=14)
            comm.Send(indices[1][node_ranges[i]:node_ranges[i + 1]], dest=i, tag=15)
        counts = initial_counts[indices[0][:node_ranges[1]], indices[1][:node_ranges[1]]]
        expected = initial_expected[indices[0][:node_ranges[1]], indices[1][:node_ranges[1]]]
        indices0 = indices[0][:node_ranges[1]]
        indices1 = indices[1][:node_ranges[1]]
        del indices
    else:
        N, start = 0, 0
        N = comm.bcast(N, root=0)
        start = comm.bcast(start, root=0)
        vrows = numpy.zeros(N, dtype=numpy.bool)
        comm.Bcast(vrows, root=0)
        M = comm.recv(source=0, tag=11)
        counts = numpy.zeros(M, dtype=numpy.int32)
        expected = numpy.zeros(M, dtype=numpy.float32)
        indices0 = numpy.zeros(M, dtype=numpy.int32)
        indices1 = numpy.zeros(M, dtype=numpy.int32)
        comm.Recv(counts, source=0, tag=12)
        comm.Recv(expected, source=0, tag=13)
        comm.Recv(indices0, source=0, tag=14)
        comm.Recv(indices1, source=0, tag=15)
    return counts, expected, vrows, indices0, indices1, N, start

def initialize_states(N, prev_states=None, start=None, res=None):
    states = numpy.zeros(N, dtype=numpy.int32)
    if rank == 0:
        mids = numpy.arange(N) * res + start
        indices = numpy.r_[numpy.searchsorted(mids, prev_states[:, 0]), mids.shape[0]]
        for i in range(indices.shape[0] - 1):
            states[indices[i]:indices[i + 1]] = prev_states[i, 2]
    comm.Bcast(states, root=0)
    return states

def find_state_bounds(scores, pos):
    where = numpy.where(numpy.logical_not(numpy.isnan(scores)))[0]
    signs = numpy.sign(scores[where])
    breaks = numpy.r_[numpy.where(signs[1:] != signs[:-1])[0] + 1, where.shape[0]]
    bed = numpy.zeros((breaks.shape[0] - 1, 3), dtype=numpy.int32)
    bed[:, 0] = pos[where[breaks[:-1]], 0]
    bed[:, 1] = pos[where[breaks[1:] - 1], 1]
    bed[:, 2] = numpy.maximum(0, signs[breaks[:-1]])
    return bed

def optimize_bounds(counts, expected, new_states, indices0, indices1, start, stop, N, vrows, minbin):
    valid = (expected > 0).astype(numpy.int32)
    positions = numpy.zeros((N, 2), dtype=numpy.int32)
    positions[:, 0] = numpy.linspace(start, stop, N + 1)[:-1].astype(numpy.int32)
    positions[:, 1] = numpy.linspace(start, stop, N + 1)[1:].astype(numpy.int32)
    iteration = 0
    max_iteration = 200
    burnin = 20
    valid1 = numpy.where(vrows > 0)[0]
    score_sums = numpy.zeros(N, dtype=numpy.float64)
    states = numpy.zeros(0, dtype=numpy.int32)
    new_scores = numpy.zeros(N, dtype=numpy.float64)
    score_changes = numpy.zeros(N, dtype=numpy.float64)
    new_score_changes = numpy.zeros(N, dtype=numpy.float64)
    where = numpy.zeros(2)
    while iteration < burnin or (iteration < max_iteration and not where.shape[0] == 0):
        prev_score_changes = score_changes
        states = new_states
        score_changes = new_score_changes
        scores = new_scores
        paramsA, paramsB, paramsAB = find_distance_relationship(counts, expected, states, valid,
                                                                indices0, indices1, minbin)
        new_states, new_scores = optimize_bound_iteration(counts, expected, states, valid, vrows, indices0, indices1,
                                                          paramsA, paramsB, paramsAB)
        if iteration >= burnin:
            score_sums += new_scores
        where = valid1[numpy.where(states[valid1] != new_states[valid1])[0]]
        new_score_changes = new_scores[where]
        if where.shape[0] > 0 and rank == 0:
            print >> sys.stderr, ("\rIter: %03i  Changed: %04i  Diff: %0.6f       ") % (iteration, where.shape[0], numpy.mean(numpy.abs(new_scores[where] - scores[where]))),
        iteration += 1
        if where.shape[0] == 0:
            score_sums = new_scores
            iteration = burnin
            break
        if numpy.array_equal(prev_score_changes, new_score_changes):
            score_sums = scores + new_scores
            iteration = burnin + 1
            break
    score_sums /= iteration - burnin + 1
    positions = positions[valid1, :]
    score_sums = score_sums[valid1]
    if rank == 0:
        print >> sys.stderr, ("\r%s\r") % (' '*80),
    return positions, score_sums

def find_distance_relationship(counts, expected, states, valid, indices0, indices1, minbin):
    N = states.shape[0]
    dist_counts = numpy.zeros((N, 3), dtype=numpy.int32)
    dist_expected = numpy.zeros((N, 3), dtype=numpy.float64)
    binsizes = numpy.zeros((N, 3), dtype=numpy.int32)
    library.find_distance_parameters3(
        counts,
        expected,
        valid,
        states,
        indices0,
        indices1,
        dist_counts,
        dist_expected,
        binsizes)
    if rank == 0:
        temp1 = numpy.zeros((N, 3), dtype=numpy.int32)
        temp2 = numpy.zeros((N, 3), dtype=numpy.float64)
        for i in range(1, num_procs):
            comm.Recv(temp1, source=i, tag=11)
            dist_counts += temp1
            comm.Recv(temp2, source=i, tag=12)
            dist_expected += temp2
            comm.Recv(temp1, source=i, tag=13)
            binsizes += temp1
        distances = numpy.log(numpy.r_[0.5, numpy.arange(1, N)].astype(numpy.float64))
        params = []
        mincount = 5000
        for i in range(3):
            data = []
            pos = minbin
            c_sum = 0
            e_sum = 0.0
            d_sum = 0.0
            n_sum = 0
            while pos < N:
                c_sum += dist_counts[pos, i]
                e_sum += dist_expected[pos, i]
                n_sum += binsizes[pos, i]
                d_sum += distances[pos] * binsizes[pos, i]
                if c_sum >= mincount:
                    data.append([d_sum / n_sum, numpy.log(c_sum / e_sum)])
                    c_sum = 0
                    e_sum = 0.0
                    d_sum = 0.0
                    n_sum = 0
                pos  += 1
            if n_sum > 0 and e_sum > 0 and c_sum > mincount:
                data.append([d_sum / n_sum, numpy.log(c_sum / e_sum)])
            data = numpy.array(data, dtype=numpy.float64)
            if data.shape[0] < 2:
                data = []
                pos = minbin
                c_sum = 0
                e_sum = 0.0
                d_sum = 0.0
                n_sum = 0
                while pos < N:
                    c_sum += numpy.sum(dist_counts[pos, :])
                    e_sum += numpy.sum(dist_expected[pos, :])
                    n_sum += numpy.sum(binsizes[pos, :])
                    d_sum += distances[pos] * numpy.sum(binsizes[pos, :])
                    if c_sum >= mincount:
                        data.append([d_sum / n_sum, numpy.log(c_sum / e_sum)])
                        c_sum = 0
                        e_sum = 0.0
                        d_sum = 0.0
                        n_sum = 0
                    pos  += 1
                if n_sum > 0 and e_sum > 0 and c_sum > mincount:
                    data.append([d_sum / n_sum, numpy.log(c_sum / e_sum)])
                data = numpy.array(data, dtype=numpy.float64)
            slopes = (data[1:, 1] - data[:-1, 1]) / (data[1:, 0] - data[:-1, 0])
            intercepts = data[1:, 1] - data[1:, 0] * slopes
            indices = numpy.searchsorted(data[1:-1, 0], distances)
            distance_parameters = numpy.exp(distances * slopes[indices] + intercepts[indices]).astype(numpy.float64)
            params.append(distance_parameters)
    else:
        comm.Send(dist_counts, dest=0, tag=11)
        comm.Send(dist_expected, dest=0, tag=12)
        comm.Send(binsizes, dest=0, tag=13)
        params = [numpy.zeros(N, dtype=numpy.float64), numpy.zeros(N, dtype=numpy.float64),
                  numpy.zeros(N, dtype=numpy.float64)]
    for i in range(3):
        comm.Bcast(params[i], root=0)
    return params

def optimize_bound_iteration(counts, expected, states, valid, vrows, indices0, indices1, paramsA, paramsB, paramsAB):
    N = vrows.shape[0]
    scores = numpy.zeros((N, 2), dtype=numpy.float64)
    library.optimize_compartment_bounds2(
        counts,
        expected,
        valid,
        states,
        indices0,
        indices1,
        paramsA,
        paramsB,
        paramsAB,
        scores)
    if rank == 0:
        for i in range(1, num_procs):
            scores += comm.recv(source=i, tag=11)
        scores = scores[:, 0] - scores[:, 1]
        temp_states = numpy.sign(scores)
        valid1 = numpy.where(vrows > 0)[0]
        where2 = valid1[numpy.where(states[valid1] != temp_states[valid1])[0]]
        temp = numpy.abs(numpy.copy(scores[where2]))
        temp.sort()
        new_states = numpy.copy(states)
        if temp.shape[0] > 0:
            #cutoff = temp[int(numpy.floor(0.75 * temp.shape[0]))]
            cutoff = temp[int(numpy.floor(0.9 * temp.shape[0]))]
            where3 = where2[numpy.where(numpy.abs(scores[where2]) >= cutoff)[0]]
            new_states[where3] = numpy.copy(temp_states[where3])
        scores[numpy.where(vrows == 0)] = numpy.nan
    else:
        comm.send(scores, dest=0, tag=11)
        scores = numpy.zeros(N, dtype=numpy.float64)
        new_states = numpy.zeros(N, dtype=numpy.int32)
    comm.Bcast(scores, root=0)
    comm.Bcast(new_states, root=0)
    return new_states, scores

def write_bg(pos, bg, chrom, fname):
    output = open(fname, 'w')
    where = numpy.where(numpy.logical_not(numpy.isnan(bg)))[0]
    for i in where:
        print >> output, "chr%s\t%i\t%i\t%f" % (chrom, pos[i, 0], pos[i, 1], bg[i])
    output.close()

def write_bed(bed, chrom, fname):
    output = open(fname, 'w')
    label = {0: 'A', 1: 'B'}
    strand = {0: '-', 1: '+'}
    for i in range(bed.shape[0]):
        print >> output, "chr%s\t%i\t%i\t%s\t%i\t%s" % (chrom, bed[i, 0], bed[i, 1], label[bed[i, 2]], bed[i, 2],
                                                        strand[bed[i, 2]])
    output.close()

def generate_parser():
    """Generate an argument parser."""
    description = "%(prog)s -- Find a quality score for a HiC dataset"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(dest="input", type=str, action='store', help="HiFive-based HM file name")
    parser.add_argument(dest="output", type=str, action='store', help="Output prefix")
    parser.add_argument('-c', dest="chrom", type=str, action='store', default='1', help="Chromosome to use")
    parser.add_argument('-m', dest="mindist", type=int, action='store', default=100000,
        help="Minimum interaction size to use in calculations")
    parser.add_argument('-i', dest="min", type=int, action='store', default=1000000, help="Initial resolution")
    parser.add_argument('-f', dest="max", type=int, action='store', default=10000, help="Final resolution")
    return parser


if __name__ == "__main__":
    main()
