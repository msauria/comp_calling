#!/usr/bin/env python

import sys
import os

import numpy
import hifive
import h5py
from library import find_binning_expected
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()
plot_results = False


def main():
    hic_fname, hdf5_fname, binsize = sys.argv[1:4]
    binsize = int(binsize)
    hic = hifive.HiC(hic_fname)
    chromosomes = list(hic.fends['chromosomes'][...])
    chroms = []
    for i in range(1, 24):
        if str(i) in chromosomes:
            chroms.append(str(i))
    for chrom in ['X', '2L', '2R', '3L', '3R']:
        if chrom in chromosomes:
            chroms.append(chrom)
    if rank == 0:
        infile = h5py.File(hdf5_fname, 'a')
        lengths = numpy.zeros(len(chroms), dtype=numpy.int32)
        for i, chrom in enumerate(chroms):
            chrint = hic.chr2int[chrom]
            lengths[i] = hic.fends['chrom_sizes'][chrint]
        infile.create_dataset(name='chromosomes', data=numpy.array(chroms))
        infile.create_dataset(name='chrom_sizes', data=lengths)
        infile.attrs['binsize'] = binsize
    else:
        infile = None
    for chrom in chroms:
        find_bin_probabilities(chrom, hic, infile, binsize, 'fend')
        find_bin_probabilities(chrom, hic, infile, binsize, 'enrichment')
        find_bin_counts(chrom, hic, infile, binsize)
    if rank == 0:
        infile.close()
        print >> sys.stderr, ("\r%s\r") % (" " * 80),

def find_bin_probabilities(chrom, hic, infile, binsize, datatype):
    if rank == 0:
        if '%s.%s' % (chrom, datatype) in infile:
            for i in range(1, num_procs):
                comm.send(0, dest=i, tag=11)
            return None
        elif '%s.%s' % (chrom, datatype) in infile['/'].attrs:
            for i in range(1, num_procs):
                comm.send(0, dest=i, tag=11)
            return None
        for i in range(1, num_procs):
            comm.send(1, dest=i, tag=11)
        print >> sys.stderr, ("\r%s\rFinding chrom %s %s array") % (' '*120, chrom, datatype),
    else:
        if comm.recv(source=0, tag=11) == 0:
            return None
    chrint = hic.chr2int[chrom]
    fends = hic.fends['fends'][...]
    chr_indices = hic.fends['chr_indices'][...]
    start_fend = chr_indices[chrint]
    stop_fend = chr_indices[chrint + 1]
    if stop_fend - start_fend == 0:
        if rank == 0:
            infile.attrs[chrom] = 'None'
        return None
    mids = fends['mid'][start_fend:stop_fend]
    start = (mids[0] / binsize) * binsize
    stop = ((mids[-1] - 1) / binsize + 1) * binsize
    if stop - start < 1000000:
        if rank == 0:
            infile.attrs[chrom] = 'None'
        return None
    N = (stop - start) / binsize
    binning_corrections = hic.binning_corrections
    binning_num_bins = hic.binning_num_bins
    fend_indices = hic.binning_fend_indices
    mapping = numpy.zeros(mids.shape[0], dtype=numpy.int32) - 1
    valid = numpy.where(hic.filter[start_fend:stop_fend])[0]
    mapping[valid] = (mids[valid] - start) / binsize
    if datatype == 'enrichment':
        distance_parameters = hic.distance_parameters
        chrom_mean = hic.chromosome_means[hic.chr2int[chrom]]
    else:
        distance_parameters = None
        chrom_mean = 1.0
    expected = numpy.zeros(N * (N + 1) / 2, dtype=numpy.float32)
    if rank == 0:
        indices = list(numpy.triu_indices(N, 0))
        indices[0] = indices[0].astype(numpy.int64)
        indices[1] = indices[1].astype(numpy.int64)
        node_ranges = numpy.round(numpy.linspace(0, indices[0].shape[0], num_procs + 1)).astype(numpy.int32)
        for i in range(1, num_procs):
            comm.send([node_ranges[i], node_ranges[i + 1], indices[0][node_ranges[i]],
                       indices[0][node_ranges[i + 1] - 1], indices[1][node_ranges[i]],
                       indices[1][node_ranges[i + 1] - 1]], dest=i, tag=11)
        start_index = node_ranges[0]
        stop_index = node_ranges[1]
        start1 = indices[0][start_index]
        stop1 = indices[0][stop_index - 1]
        start2 = indices[1][start_index]
        stop2 = indices[1][stop_index - 1]
    else:
        start_index, stop_index, start1, stop1, start2, stop2 = comm.recv(source=0, tag=11)
    startfend1 = numpy.searchsorted(mids, start1 * binsize + start)
    stopfend1 = numpy.searchsorted(mids, (stop1 + 1) * binsize + start)
    startfend2 = numpy.searchsorted(mids, start2 * binsize + start)
    stopfend2 = numpy.searchsorted(mids, (stop2 + 1) * binsize + start)
    find_binning_expected(
        mapping,
        binning_corrections,
        binning_num_bins,
        fend_indices,
        mids,
        distance_parameters,
        expected,
        chrom_mean,
        start_fend,
        startfend1,
        stopfend1,
        startfend2,
        stopfend2,
        start1,
        stop1,
        start2,
        stop2,
        1)
    if rank == 0:
        for i in range(1, num_procs):
            expected[node_ranges[i]:node_ranges[i + 1]] = comm.recv(source=i, tag=11)
        expected = expected.astype(numpy.float64)
        infile.create_dataset(name='%s.%s' % (chrom, datatype), data=expected)
        if plot_results:
            img = hifive.plotting.plot_upper_array(numpy.vstack((expected, expected > 0)).T, symmetricscaling=False)
            img.save("%s_%s.png" % (chrom, datatype))
        return None
    else:
        comm.send(expected[start_index:stop_index], dest=0, tag=11)
        return None

def find_bin_counts(chrom, hic, infile, binsize):
    if rank == 0:
        if '%s.counts' % (chrom) in infile:
            counts = infile['%s.counts' % chrom][...]
            return counts
        elif '%s.counts' % (chrom) in infile['/'].attrs:
            return None
        print >> sys.stderr, ("\r%s\rFinding chrom %s counts array") % (' '*120, chrom),
    else:
        return None
    chrint = hic.chr2int[chrom]
    chr_indices = hic.fends['fend_indices'][...]
    fends = hic.fends['fends'][...]
    if chr_indices[chrint] == chr_indices[chrint + 1]:
        return None
    start_fend = chr_indices[chrint]
    stop_fend = chr_indices[chrint + 1]
    if stop_fend - start_fend == 0:
        if rank == 0:
            infile.attrs[chrom] = 'None'
        return None
    start = (fends['mid'][start_fend] / binsize) * binsize
    stop = ((fends['mid'][stop_fend - 1] - 1) / binsize + 1) * binsize
    if stop - start < 1000000:
        if rank == 0:
            infile.attrs[chrom] = 'None'
        return None
    N = (stop - start) / binsize
    start_index = hic.data['cis_indices'][start_fend]
    stop_index = hic.data['cis_indices'][stop_fend]
    counts = hic.data['cis_data'][start_index:stop_index, :]
    if counts.shape[0] == 0:
        return None
    counts[:, 0] = (fends['mid'][counts[:, 0]] - start) / binsize
    counts[:, 1] = (fends['mid'][counts[:, 1]] - start) / binsize
    all_indices = counts[:, 0].astype(numpy.int64) * N + counts[:, 1].astype(numpy.int64)
    indices = numpy.unique(all_indices)
    new_counts = numpy.zeros((indices.shape[0], 3), dtype=numpy.int32)
    new_counts[:, 0] = indices / N
    new_counts[:, 1] = indices % N
    new_counts[:, 2] = numpy.bincount(numpy.searchsorted(indices, all_indices, side='left'), weights=counts[:, 2],
                                  minlength=indices.shape[0])
    infile.create_dataset(name="%s.counts" % chrom, data=new_counts)
    infile.attrs['%s.start' % chrom] = start
    data = numpy.zeros((N, N, 2), dtype=numpy.int32)
    data[new_counts[:, 0], new_counts[:, 1], 0] = new_counts[:, 2]
    data[new_counts[:, 1], new_counts[:, 0], 0] = new_counts[:, 2]
    data[:, :, 1] = data[:, :, 0] > 0
    if plot_results:
        img = hifive.plotting.plot_full_array(data, symmetricscaling=False)
        img.save("%s_counts.png" % (chrom))
    return None


if __name__ == "__main__":
    main()
