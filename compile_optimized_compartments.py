#!/usr/bin/env python

import sys
import glob

import numpy

def main():
    prefix, out_prefix, len_fname, score_fname, enrichment = sys.argv[1:6]
    sizes = load_sizes(len_fname)
    scores = load_scores(score_fname)
    bed = load_bed(prefix, sizes)
    bg = load_bg(prefix, sizes)
    chroms = bed.keys()
    for chrom in chroms:
        bed[chrom], bg[chrom] = orient_scores(bed[chrom], bg[chrom], scores[chrom], enrichment)
    write_bed(bed, "%s.bed" % out_prefix)
    write_bg(bg, "%s.bg" % out_prefix)

def load_sizes(fname):
    sizes = {}
    for line in open(fname):
        temp = line.rstrip('\n').split('\t')
        sizes[temp[0].strip('chr')] = int(temp[1])
    return sizes

def load_scores(fname):
    scores = {}
    for line in open(fname):
        temp = line.rstrip('\n').split('\t')
        temp[0] = temp[0].strip('chr')
        if temp[0] not in scores:
            scores[temp[0]] = []
        scores[temp[0]].append((int(temp[1]), int(temp[2]), float(temp[3])))
    for chrom in scores:
        scores[chrom] = numpy.array(scores[chrom], dtype=numpy.dtype([('start', numpy.int32), ('stop', numpy.int32),
                                                                      ('score', numpy.float32)]))
    return scores

def load_bg(prefix, sizes):
    fnames = glob.glob("%s*.bg" % prefix)
    bg = {}
    for fname in fnames:
        for line in open(fname):
            temp = line.rstrip('\n').split('\t')
            temp[0] = temp[0].strip('chr')
            if temp[0] not in bg:
                bg[temp[0]] = []
            temp[1], temp[2] = int(temp[1]), min(sizes[temp[0]], int(temp[2]))
            if temp[1] < sizes[temp[0]]:
                bg[temp[0]].append((temp[1], temp[2], float(temp[3])))
    for chrom in bg:
        bg[chrom] = numpy.array(bg[chrom], dtype=numpy.dtype([('start', numpy.int32), ('stop', numpy.int32),
                                                                      ('score', numpy.float32)]))
    return bg

def load_bed(prefix, sizes):
    fnames = glob.glob("%s*.bed" % prefix)
    bed = {}
    strand = {'+': 1, '-': 0}
    for fname in fnames:
        for line in open(fname):
            temp = line.rstrip('\n').split('\t')
            temp[0] = temp[0].strip('chr')
            if temp[0] not in bed:
                bed[temp[0]] = []
            temp[1], temp[2] = int(temp[1]), min(sizes[temp[0]], int(temp[2]), )
            if temp[1] < sizes[temp[0]]:
                bed[temp[0]].append((temp[1], temp[2], strand[temp[5]]))
    for chrom in bed:
        bed[chrom] = numpy.array(bed[chrom], dtype=numpy.int32)
    return bed

def orient_scores(bed, bg, scores, enrichment):
    mids = (scores['start'] + scores['stop']) / 2
    starts = numpy.searchsorted(mids, bed[:, 0])
    stops = numpy.searchsorted(mids, bed[:, 1])
    sums = numpy.zeros(bed.shape[0], dtype=numpy.float32)
    for i in range(sums.shape[0]):
        sums[i] = numpy.sum(scores['score'][starts[i]:stops[i]]) / max(1, (stops[i] - starts[i]))
    A = numpy.sum(sums[numpy.where(bed[:, 2] == 0)[0]])
    B = numpy.sum(sums[numpy.where(bed[:, 2] == 1)[0]])
    if enrichment == 'B' and B < A:
        bed[:, 2] = 1 - bed[:, 2]
        bg['score'] = -bg['score']
    elif enrichment == 'A' and B > A:
        bed[:, 2] = 1 - bed[:, 2]
        bg['score'] = -bg['score']
    return bed, bg

def write_bed(bed, fname):
    chroms = bed.keys()
    chroms.sort()
    labels = {0: 'A', 1: 'B'}
    strands = {0: '-', 1: '+'}
    output = open(fname, 'w')
    for chrom in chroms:
        for i in range(bed[chrom].shape[0]):
            print >> output, "chr%s\t%i\t%i\t%s\t%i\t%s" % (chrom, bed[chrom][i, 0], bed[chrom][i, 1],
                labels[bed[chrom][i, 2]], bed[chrom][i, 2], strands[bed[chrom][i, 2]])
    output.close()

def write_bg(bg, fname):
    chroms = bg.keys()
    chroms.sort()
    output = open(fname, 'w')
    for chrom in chroms:
        for i in range(bg[chrom].shape[0]):
            print >> output, "chr%s\t%i\t%i\t%f" % (chrom, bg[chrom]['start'][i], bg[chrom]['stop'][i],
                                                    bg[chrom]['score'][i])
    output.close()


if __name__ == "__main__":
    main()
