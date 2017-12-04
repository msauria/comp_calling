#!/usr/bin/env python


import sys
from argparse import ArgumentParser

import numpy
import scipy.sparse.linalg
import hifive

def main(args):
  hic = hifive.HiC(args.HIC)
  if args.CHROMS == '':
    args.CHROMS = hic.fends['chromosomes'][...]
  else:
    args.CHROMS = args.CHROMS.split(',')
  cis_data = {}
  cis_mapping = {}
  args.CHROMS.sort()
  chr_indices = [0]
  for chrom in args.CHROMS:
    temp = hic.cis_heatmap(chrom, binsize=args.BINSIZE, datatype='enrichment', arraytype='full', returnmapping=True)
    if temp is not None:
      cis_data[chrom] = temp[0]
      cis_mapping[chrom] = temp[1][:, :2]
      chr_indices.append(chr_indices[-1] + temp[1].shape[0])
  args.CHROMS = cis_data.keys()
  args.CHROMS.sort()
  data = numpy.zeros((chr_indices[-1], chr_indices[-1], 2), dtype=numpy.float64)
  mapping = numpy.zeros((chr_indices[-1], 3), dtype=numpy.int32)
  for i, chrom in enumerate(args.CHROMS):
    #data[chr_indices[i]:chr_indices[i + 1], chr_indices[i]:chr_indices[i + 1], :] = cis_data[chrom]
    del cis_data[chrom]
    mapping[chr_indices[i]:chr_indices[i + 1], 0] = i
    mapping[chr_indices[i]:chr_indices[i + 1], 1:] = cis_mapping[chrom]
    del cis_mapping[chrom]
    for j in range(i + 1, len(args.CHROMS)):
      chrom2 = args.CHROMS[j]
      trans = hic.trans_heatmap(chrom, chrom2, binsize=args.BINSIZE, start1=mapping[chr_indices[i], 1],
        stop1=mapping[chr_indices[i + 1] - 1, 2], start2=mapping[chr_indices[j], 1], stop2=mapping[chr_indices[j + 1] - 1, 2],
        datatype='enrichment')
      data[chr_indices[i]:chr_indices[i + 1], chr_indices[j]:chr_indices[j + 1], :] = trans
      data[chr_indices[j]:chr_indices[j + 1], chr_indices[i]:chr_indices[i + 1], :] = numpy.transpose(trans, (1, 0, 2))

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

  data2 = numpy.copy(data)
  data2.fill(0)
  for i in range(chr_indices[-1] - 1):
    print >> sys.stderr, ("\r%s\rCorrelating %i of %i bins") % (' '*50, i, chr_indices[-1]),
    for j in range(i + 1, chr_indices[-1]):
      try:
        where = numpy.where((data[i, :, 1] > 0) & (data[j, :, 1] > 0))[0]
        if where.shape[0] < 3:
          continue
        corr = numpy.corrcoef(data[i, where, 0], data[j, where, 0])[0, 1]
        if corr != numpy.nan and abs(corr) < numpy.inf:
          data2[i, j, 0] = corr
          data2[i, j, 1] = 1
      except:
        pass
      data2[j, i, :] = data2[i, j, :]
  where = numpy.where(data2[:, :, 1])
  scores = data2[where[0], where[1], 0]
  scores.sort()
  data2[where[0], where[1], 0] = numpy.maximum(scores[int(scores.shape[0]*0.05)], numpy.minimum(scores[int(scores.shape[0]*0.95)],
                                  data2[where[0], where[1], 0])) - scores[int(scores.shape[0] / 2)]
  data2[where[0], where[1], 0] /= numpy.amax(numpy.abs(data2[where[0], where[1], 0]))

  valid = numpy.sum(data2[:, :, 1], axis=1) >= data2.shape[0] / 2
  vrows = numpy.where(valid)[0]
  ivrows = numpy.where(numpy.logical_not(valid))[0]
  eigen = scipy.sparse.linalg.eigs(data2[vrows, :, 0][:, vrows], k=1)[1][:, 0]

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
    print numpy.amax(eigen), numpy.amin(eigen)
    for i, X in enumerate(vrows):
      data3[X, :40, 1] = 1
      if eigen[i] >= 0:
        data3[X, 20:(20 + int(round(eigen[i]))), 0] = 1
      else:
        data3[X, (20 - int(round(-eigen[i]))):20, 0] = -1

    img = hifive.plotting.plot_full_array(data3, logged=False, symmetricscaling=True)
    img.save("%s_comp.png" % args.OUTPUT)


def generate_parser():
  parser = ArgumentParser(prog='Compartment_finder', description='Find eigenvector-based compartment calls from HiFive Hi-C data.',
                          add_help=True)
  parser.add_argument(metavar='HIC', dest='HIC', action='store', type=str, help='Input HiFive Hi-C project file')
  parser.add_argument(metavar='OUTPUT', dest="OUTPUT", action='store', type=str, help='Output file prefix')
  parser.add_argument('-b', metavar='BINSIZE', dest="BINSIZE", action='store', type=int, default=0, help='Resolution to bin data at')
  parser.add_argument('-c', metavar='CHROMS', dest="CHROMS", action='store', type=str, default='', help='Comma-separated list of chromosomes to find comparments for')
  parser.add_argument('-p', dest="PLOT", action='store_true', help='Plot heatmaps')
  return parser

if __name__ == '__main__':
  parser = generate_parser()
  args = parser.parse_args()
  main(args)
