#!/usr/bin/env python


import sys
from argparse import ArgumentParser

import time
import numpy
import scipy.sparse.linalg
import hifive
try:
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  num_procs = comm.Get_size()
  rank = comm.Get_rank()
except:
  comm = None
  num_procs = 1
  rank = 0

def main(args):
  hic = hifive.HiC(args.HIC)
  if args.CHROMS == '':
    args.CHROMS = hic.fends['chromosomes'][...]
  else:
    args.CHROMS = args.CHROMS.split(',')
  bounds = {}
  args.CHROMS.sort()
  new_chr_indices = [0]
  if 'binned' in hic.__dict__ and hic.binned is not None:
    fends = hic.fends['bins'][...]
    chr_indices = hic.fends['bin_indices'][...]
  else:
    fends = hic.fends['fends'][...]
    chr_indices = hic.fends['chr_indices'][...]
  for chrom in args.CHROMS:
    chrint = hic.chr2int[chrom]
    sfend = chr_indices[chrint]
    efend = chr_indices[chrint + 1]
    valid = numpy.where(hic.filter[sfend:efend])[0] + sfend
    if valid.shape[0] < 2:
      print >> sys.stderr, ("Insufficient information for %s\n") % (chrom),
      continue
    sbin = (fends['mid'][valid[0]] / args.BINSIZE) * args.BINSIZE
    ebin = (fends['mid'][valid[-1]] / args.BINSIZE + 1) * args.BINSIZE
    N = (ebin - sbin) / args.BINSIZE
    bounds[chrom] = numpy.zeros((N, 2), dtype=numpy.int32)
    bounds[chrom][:, 0] = numpy.arange(N) * args.BINSIZE + sbin
    bounds[chrom][:, 1] = bounds[chrom][:, 0] + args.BINSIZE
    new_chr_indices.append(new_chr_indices[-1] + N)
  args.CHROMS = bounds.keys()
  args.CHROMS.sort()
  args = comm.bcast(args, root=0)
  chr_indices = new_chr_indices
  data = numpy.zeros((chr_indices[-1], chr_indices[-1], 2), dtype=numpy.float64)
  mapping = numpy.zeros((chr_indices[-1], 3), dtype=numpy.int32)
  for i, chrom in enumerate(args.CHROMS):
    mapping[chr_indices[i]:chr_indices[i + 1], 0] = i
    mapping[chr_indices[i]:chr_indices[i + 1], 1:] = bounds[chrom]
  indices = list(numpy.triu_indices(len(args.CHROMS), 1))
  if comm is not None:
    bounds = comm.bcast(bounds, root=0)
    node_ranges = numpy.round(numpy.linspace(0, indices[0].shape[0], num_procs + 1)).astype(numpy.int32)
    for i in range(1, num_procs):
      comm.send(indices[0][node_ranges[i]:node_ranges[i + 1]], dest=i)
      comm.send(indices[1][node_ranges[i]:node_ranges[i + 1]], dest=i)
    indices0 = indices[0][:node_ranges[1]]
    indices1 = indices[1][:node_ranges[1]]
  else:
    indices0 = indices[0]
    indices1 = indices[1]
  for i in range(indices0.shape[0]):
    X = indices0[i]
    Y = indices1[i]
    chrom = args.CHROMS[X]
    chrom2 = args.CHROMS[Y]
    data[chr_indices[X]:chr_indices[X + 1], chr_indices[Y]:chr_indices[Y + 1], :] = hic.trans_heatmap(
      chrom, chrom2, binsize=args.BINSIZE, start1=mapping[chr_indices[X], 1], stop1=mapping[chr_indices[X + 1] - 1, 2],
      start2=mapping[chr_indices[Y], 1], stop2=mapping[chr_indices[Y + 1] - 1, 2], datatype=args.DATATYPE)
  if comm is not None:
    for i in range(1, num_procs):
      for j in range(node_ranges[i], node_ranges[i + 1]):
        X = indices[0][j]
        Y = indices[1][j]
        temp = numpy.zeros((chr_indices[X + 1]- chr_indices[X]) * (chr_indices[Y + 1] - chr_indices[Y]) * 2, dtype=numpy.float32)
        comm.Recv(temp, source=i, tag=(X*len(args.CHROMS) + Y))
        data[chr_indices[X]:chr_indices[X + 1], chr_indices[Y]:chr_indices[Y + 1], :] = temp.reshape(chr_indices[X + 1] - chr_indices[X], -1, 2)
  N = data.shape[0]
  indices = list(numpy.triu_indices(N, 1))
  data[indices[1], indices[0], :] = data[indices[0], indices[1], :]
  valid = numpy.sum(data[:, :, 0], axis=1) > mapping.shape[0] / 2.
  ivrows = numpy.where(numpy.logical_not(valid))[0]
  data[ivrows, :, :] = 0
  data[:, ivrows, :] = 0

  if args.PLOT:
    img = hifive.plotting.plot_full_array(data, symmetricscaling=False)
    img.save("%s_enr.png" % args.OUTPUT)

  where = numpy.where((data[:, :, 0] > 0) & (data[:, :, 1] > 0))
  data[where[0], where[1], 0] /= data[where[0], where[1], 1]
  data[where[0], where[1], 1] = 1
  data[where[0], where[1], 0] = numpy.log(data[where[0], where[1], 0])
  scores = data[where[0], where[1], 0]
  scores.sort()
  data[where[0], where[1], 0] = numpy.maximum(scores[int(scores.shape[0]*0.05)], numpy.minimum(scores[int(scores.shape[0]*0.95)],
                                  data[where[0], where[1], 0]))
  data[where[0], where[1], 0] -= numpy.mean(data[where[0], where[1], 0])

  data2 = numpy.zeros(data.shape, dtype=data.dtype)
  indices[0] = indices[0].astype(numpy.int32)
  indices[1] = indices[1].astype(numpy.int32)
  if comm is not None:
    N = comm.bcast(N, root=0)
    comm.Bcast(data, root=0)
    node_ranges = numpy.round(numpy.linspace(0, indices[0].shape[0], num_procs + 1)).astype(numpy.int32)
    for i in range(1, num_procs):
      comm.send(node_ranges[i + 1] - node_ranges[i], dest=i)
      comm.Send(indices[0][node_ranges[i]:node_ranges[i + 1]], dest=i)
      comm.Send(indices[1][node_ranges[i]:node_ranges[i + 1]], dest=i)
    indices0 = indices[0][:node_ranges[1]]
    indices1 = indices[1][:node_ranges[1]]
  else:
    indices0, indices1 = indices
  for i in range(indices0.shape[0]):
    print >> sys.stderr, ("\r%s\rCorrelating %i of %i bins") % (' '*50, i, indices0.shape[0]),
    X = indices0[i]
    Y = indices1[i]
    try:
      where = numpy.where((data[X, :, 1] > 0) & (data[Y, :, 1] > 0))[0]
      if where.shape[0] < N / 10.:
        continue
      corr = numpy.corrcoef(data[X, where, 0], data[Y, where, 0])[0, 1]
      if corr != numpy.nan and abs(corr) < numpy.inf:
        data2[X, Y, 0] = corr
        data2[X, Y, 1] = 1
    except:
      pass
  if comm is not None:
    for i in range(1, num_procs):
      temp = numpy.zeros((node_ranges[i + 1] - node_ranges[i], 2), dtype=numpy.float64)
      comm.Recv(temp, source=i)
      data2[indices[0][node_ranges[i]:node_ranges[i + 1]], indices[1][node_ranges[i]:node_ranges[i + 1]], :] = temp
  data2[indices[1], indices[0], :] = data2[indices[0], indices[1], :]
  where = numpy.where(data2[:, :, 1])
  scores = data2[where[0], where[1], 0]
  scores.sort()
  data2[where[0], where[1], 0] = numpy.maximum(scores[int(scores.shape[0]*0.05)], numpy.minimum(scores[int(scores.shape[0]*0.95)],
                                  data2[where[0], where[1], 0])) - scores[int(scores.shape[0] / 2)]
  data2[where[0], where[1], 0] /= numpy.amax(numpy.abs(data2[where[0], where[1], 0]))

  valid = numpy.sum(data2[:, :, 1], axis=1) >= data2.shape[0] / 2
  vrows = numpy.where(valid)[0]
  ivrows = numpy.where(numpy.logical_not(valid))[0]
  eigen = numpy.real(scipy.sparse.linalg.eigs(data2[vrows, :, 0][:, vrows], k=1)[1][:, 0])

  output = open("%s.bg" % args.OUTPUT, 'w')
  output1 = open("%s.bed" % args.OUTPUT, 'w')
  start = mapping[vrows[0], 0]
  for i, X, in enumerate(vrows):
    print >> output, "%s\t%i\t%i\t%f" % (args.CHROMS[mapping[X, 0]], mapping[X, 1], mapping[X, 1], eigen[i])
    if i < vrows.shape[0] - 1:
      if mapping[X, 0] != mapping[vrows[i + 1], 0] or numpy.sign(eigen[i]) != numpy.sign(eigen[i + 1]):
        if eigen[i]  >= 0:
          score = 1
          sign = '+'
        else:
          score = -1
          sign = '-'
        print >> output1, "%s\t%i\t%i\t.\t%i\t%s" % (args.CHROMS[mapping[X, 0]], start, mapping[X, 2], score, sign)
        start = mapping[vrows[i + 1], 1]
    else:
      if eigen[i]  >= 0:
        score = 1
        sign = '+'
      else:
        score = -1
        sign = '-'
      print >> output1, "%s\t%i\t%i\t.\t%i\t%s" % (args.CHROMS[mapping[X, 0]], start, mapping[X, 2], score, sign)
  output.close()
  output1.close()

  if args.PLOT:
    data3 = numpy.zeros((data2.shape[0], data2.shape[0] + 42, 2), dtype=data2.dtype)
    data3[:, 42:, :] = data2
    eigen /= numpy.amax(numpy.abs(eigen)) / 20.5
    for i, X in enumerate(vrows):
      data3[X, :40, 1] = 1
      if eigen[i] >= 0:
        data3[X, 20:(20 + int(round(eigen[i]))), 0] = 1
      else:
        data3[X, (20 - int(round(-eigen[i]))):20, 0] = -1

    img = hifive.plotting.plot_full_array(data3, logged=False, symmetricscaling=True)
    img.save("%s_comp.png" % args.OUTPUT)

def worker():
  args = comm.bcast(None, root=0)
  hic = hifive.HiC(args.HIC)
  bounds = comm.bcast(None, root=0)
  indices0 = comm.recv(source=0)
  indices1 = comm.recv(source=0)
  for i in range(indices0.shape[0]):
    X = indices0[i]
    Y = indices1[i]
    chrom = args.CHROMS[X]
    chrom2 = args.CHROMS[Y]
    data = hic.trans_heatmap(
      chrom, chrom2, binsize=args.BINSIZE, start1=bounds[chrom][0, 0], stop1=bounds[chrom][-1, 1],
      start2=bounds[chrom2][0, 0], stop2=bounds[chrom2][-1, 1], datatype=args.DATATYPE)
    comm.Send(data.flatten(), dest=0, tag=(X * len(args.CHROMS) + Y))

  N = comm.bcast(None, root=0)
  data = numpy.zeros((N, N, 2), dtype=numpy.float64)
  comm.Bcast(data, root=0)
  M = comm.recv(source=0)
  indices0 = numpy.zeros(M, dtype=numpy.int32)
  indices1 = numpy.zeros(M, dtype=numpy.int32)
  comm.Recv(indices0, source=0)
  comm.Recv(indices1, source=0)
  data2 = numpy.zeros((M, 2), dtype=numpy.float64)
  for i in range(indices0.shape[0]):
    X = indices0[i]
    Y = indices1[i]
    try:
      where = numpy.where((data[X, :, 1] > 0) & (data[Y, :, 1] > 0))[0]
      if where.shape[0] < N / 2.:
        continue
      corr = numpy.corrcoef(data[X, where, 0], data[Y, where, 0])[0, 1]
      if corr != numpy.nan and abs(corr) < numpy.inf:
        data2[i, 0] = corr
        data2[i, 1] = 1
    except:
      pass
  comm.Send(data2, dest=0)
  del data
  del data2
  return None

def generate_parser():
  parser = ArgumentParser(prog='Compartment_finder', description='Find eigenvector-based compartment calls from HiFive Hi-C data.',
                          add_help=True)
  parser.add_argument(metavar='HIC', dest='HIC', action='store', type=str, help='Input HiFive Hi-C project file')
  parser.add_argument(metavar='OUTPUT', dest="OUTPUT", action='store', type=str, help='Output file prefix')
  parser.add_argument('-d', metavar='DATATYPE', dest="DATATYPE", action='store', choices=['raw', 'fend'], default='raw', help='Type of corrected data to use')
  parser.add_argument('-b', metavar='BINSIZE', dest="BINSIZE", action='store', type=int, default=0, help='Resolution to bin data at')
  parser.add_argument('-c', metavar='CHROMS', dest="CHROMS", action='store', type=str, default='', help='Comma-separated list of chromosomes to find comparments for')
  parser.add_argument('-p', dest="PLOT", action='store_true', help='Plot heatmaps')
  return parser

if __name__ == '__main__':
  if rank == 0:
    parser = generate_parser()
    args = parser.parse_args()
    main(args)
  else:
    worker()
