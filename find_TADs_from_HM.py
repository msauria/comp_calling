#!/usr/bin/env python

import sys
import os
from math import ceil

import numpy
import h5py
import hifive
#from pyx import *


def main():
    in_fname, out_prefix, width, window = sys.argv[1:5]
    width, window = int(width), int(window)
    DIs = {}
    infile = h5py.File(in_fname, 'r')
    chroms = infile['chromosomes'][...]
    sizes = infile['chrom_sizes'][...]
    binsize = infile.attrs['binsize']
    width /= binsize
    window /= binsize
    for h, chrom in enumerate(chroms):
        print >> sys.stderr, ("\rLoading chr%s data") % (chrom),
        N = int((0.25 + 2 * infile['%s.fend' % chrom].shape[0]) ** 0.5 - 0.5)
        start = infile.attrs['%s.start' % chrom]
        mapping = numpy.zeros((N, 2), dtype=numpy.int32)
        mapping[:, 0] = start + numpy.arange(N) * binsize
        mapping[:, 1] = mapping[:, 0] + binsize
        indices = numpy.triu_indices(N, 0)
        counts = numpy.zeros((N, N), dtype=numpy.int32)
        temp = infile['%s.counts' % chrom][...]
        counts[temp[:, 0], temp[:, 1]] = temp[:, 2]
        del temp
        counts[indices[1], indices[0]] = counts[indices]
        expected = numpy.zeros((N, N), dtype=numpy.float32)
        expected[indices] = infile['%s.fend' % chrom][...]
        expected[indices[1], indices[0]] = expected[indices]
        print >> sys.stderr, ("\r%s\rFinding chr%s DI") %  (' ' * 80, chrom),
        DI = numpy.zeros(N, dtype=numpy.float64)
        DI.fill(numpy.nan)
        for i in range(window / 4 - 1, N - window / 4):
            maxdist = min(i, N - i - width)
            down = numpy.sum(counts[(i + width + 1):min(i + window, i + maxdist), i:(i + width)])
            if down > 0:
                down /= numpy.sum(expected[(i + width + 1):min(i + maxdist, i + window), i:(i + width)])
                up = numpy.sum(counts[i:(i + width),(i - min(maxdist, window) + width + 1):i])
                if up > 0:
                    up /= numpy.sum(expected[i:(i + width),(i - min(maxdist, window) + width + 1):i])
                    DI[i] = numpy.log2(down / up)
        where = numpy.where(numpy.logical_not(numpy.isnan(DI)))[0]
        temp = numpy.zeros(where.shape[0], dtype=numpy.dtype([('position', numpy.int32),
                                                              ('score', numpy.float64)]))
        temp['position'] = mapping[where, 0] + (binsize * width) / 2
        temp['score'] = DI[where]
        DIs[chrom] = temp
        temp = numpy.abs(numpy.copy(DIs[chrom]['score']))
        temp.sort()
        cutoff = temp[int(0.95 * temp.shape[0])]
        DIs[chrom]['score'] = numpy.maximum(-cutoff, numpy.minimum(cutoff, DIs[chrom]['score']))
    infile.close()
    print >> sys.stderr, ("\r%s\rFinding TADs") %  (' ' * 80),
    TADs, states = find_TADs(DIs)
    #plot_DI(hm, '19', DIs, TADs, states).writePDFfile("%s.pdf" % out_prefix)
    write_Scores(chroms, DIs, sizes, binsize, "%s.bg" % out_prefix)
    write_TADs(chroms, TADs, sizes, binsize, "%s.bed" % out_prefix)
    print >> sys.stderr, ("\r%s\r") %  (' ' * 80),

def find_TADs(DIs):
    normed = []
    chroms = []
    for chrom in DIs:
        chroms.append(chrom)
        normed.append(numpy.copy(DIs[chrom]['score']))
        normed[-1] /= numpy.std(normed[-1])
    states = get_states(normed, chroms)
    TADs = {}
    for h, chrom in enumerate(chroms):
        TADs[chrom] = []
        pos = 0
        while states[chrom][pos] != 0:
            pos += 1
        TADs[chrom].append([pos, pos])
        state0 = True
        while pos < states[chrom].shape[0]:
            if state0:
                if states[chrom][pos] == 2:
                    state0 = False
                    TADs[chrom][-1][1] = pos
            else:
                if states[chrom][pos] == 0:
                    state0 = True
                    TADs[chrom].append([pos, pos])
                elif states[chrom][pos] == 2:
                    TADs[chrom][-1][1] = pos
            pos += 1
        if TADs[chrom][-1][0] == TADs[chrom][-1][1]:
            del TADs[chrom][-1]
        TADs[chrom] = numpy.array(TADs[chrom])
        positions = numpy.zeros((TADs[chrom].shape[0], 2), dtype=numpy.int32)
        positions[:, 0] = DIs[chrom]['position'][TADs[chrom][:, 0]]
        positions[:, 1] = DIs[chrom]['position'][TADs[chrom][:, 1]]
        TADs[chrom] = positions
    return TADs, states

def get_states(normed, chroms):
    pi = numpy.array([0.3, 0.4, 0.3])
    transitions = numpy.array([[0.8, 0.05, 0.15],
                               [0.25, 0.5, 0.25],
                               [0.15, 0.05, 0.8]])
    distributions = [[[0.33, 0.0, 1.0],
                      [0.33, 1.0, 1.0],
                      [0.33, 1.5, 1.0]],
                     [[0.33, -0.5, 1.0],
                      [0.33, 0.0, 1.0],
                      [0.33, 0.5, 1.0]],
                     [[0.33, -1.5, 1.0],
                      [0.33, -1.0, 1.],
                      [0.33, -0.0, 1.]]]
    hmm = hifive.libraries.hmm.HMM(
        seed=2001,
        num_states=3,
        num_distributions=3,
        pi=pi,
        transitions=transitions,
        distributions=distributions)
    hmm.train(normed)
    print "pi", hmm.pi
    print "transitions", hmm.transitions
    print "dists"
    for i in range(3):
        print hmm.distributions[i, :, :]
    states = {}
    for h, chrom in enumerate(chroms):    
        states[chrom] = hmm.find_path(normed[h])[0]
    return states

def plot_DI_tads(self, out_fname):
    if 'pyx' not in sys.modules:
        return None
    unit.set(defaultunit="cm")
    text.set(mode="latex")
    text.preamble(r"\usepackage{times}")
    text.preamble(r"\usepackage{sansmath}")
    text.preamble(r"\sansmath")
    text.preamble(r"\renewcommand*\familydefault{\sfdefault}")
    painter = graph.axis.painter.regular( labeldist=0.1, labelattrs=[text.size(-3)], titleattrs=[text.size(-3)] )
    pages = []
    for chrom in self.DIs:
        start = self.DIs[chrom]['position'][0]
        stop = self.DIs[chrom]['position'][-1]
        max_val = numpy.amax(numpy.abs(self.DIs[chrom]['score'])) * 1.05
        g = graph.graphxy( width=40, height=5,
            x=graph.axis.lin(min=start, max=stop, title='', painter=painter), 
            y=graph.axis.lin(min=-max_val, max=max_val, title='', painter=painter), 
            x2=graph.axis.lin(min=0, max=1, parter=None),
            y2=graph.axis.lin(min=0, max=1, parter=None))
        g.text(40.1, 2.5, chrom, [text.halign.center, text.valign.bottom, trafo.rotate(-90), text.size(-3)])
        g.plot(graph.data.points(zip(self.DIs[chrom]['position'], self.DIs[chrom]['score']), x=1, y=2),
            [graph.style.line([style.linewidth.THIN])])
        g.stroke(path.line(0, 2.5, 40, 2.5), [style.linestyle.dotted, style.linewidth.THIN])
        for i in range(self.TADs[chrom].shape[0]):
            X0 = (self.TADs[chrom][i, 0] - start) / float(stop - start) * 40.0
            X1 = (self.TADs[chrom][i, 1] - start) / float(stop - start) * 40.0
            if i % 2 == 0:
                Y = 1.25
            else:
                Y = 3.75
            g.stroke(path.line(X0, Y, X1, Y), [style.linewidth.THICK])
            g.stroke(path.line(X0, 1.25, X0, 3.75), [style.linewidth.THIN, style.linestyle.dotted])
            if i == self.TADs[chrom].shape[0] - 1 or self.TADs[chrom][i, 1] != self.TADs[chrom][i + 1, 0]:
                g.stroke(path.line(X1, 1.25, X1, 3.75), [style.linewidth.THIN, style.linestyle.dotted])
        pages.append(document.page(g))
    doc = document.document(pages)
    doc.writePDFfile(out_fname)    

def plot_DI(hm, chrom, scores, TADs, states):
    if 'pyx' not in sys.modules:
        return None
    mapping, rcounts, rexpected = hm
    N = mapping.shape[0]
    M = N / 5
    start = mapping[0, 0]
    stop = mapping[(N / 5) * 5 - 1, 1]
    width = (stop - start) / float(50000) / 72.
    counts = numpy.zeros((M, M), dtype=numpy.int32)
    expected = numpy.zeros((M, M), dtype=numpy.float32)
    for i in range(5):
        for j in range(5):
            counts += rcounts[i:(i + M * 5):5, j:(j + M * 5):5]
            expected += rexpected[i:(i + M * 5):5, j:(j + M * 5):5]
    maxval = numpy.amax(numpy.abs(scores[chrom]['score'])) * 1.05
    c = canvas.canvas()
    c.insert(bitmap.bitmap(0, 0, hifive.plotting.plot_full_array(numpy.dstack((counts, expected)), symmetricscaling=False), width=width))
    Xs = (scores[chrom]['position'] - start) / float(stop - start) * width
    lpath = None
    for i in range(Xs.shape[0]):
        Y = scores[chrom]['score'][i] / maxval
        if lpath is None:
            lpath = path.path(path.moveto(Xs[i], 0), path.lineto(Xs[i], Y))
        else:
            lpath.append(path.lineto(Xs[i], Y))
    lpath.append(path.lineto(Xs[-1], 0))
    lpath.append(path.closepath())
    c.fill(lpath, [trafo.translate(0, -1)])
    for i in range(TADs[chrom].shape[0]):
        X0 = (TADs[chrom][i, 0] - start) / float(stop - start) * width
        X1 = (TADs[chrom][i, 1] - start) / float(stop - start) * width
        c.stroke(path.line(X0, -2.4, X0, width - X0), [style.linewidth.THin])
        c.stroke(path.line(X1, -2.4, X1, width - X1), [style.linewidth.THin])
        c.stroke(path.line(0, width - X0, X0, width - X0), [style.linewidth.THin])
        c.stroke(path.line(0, width - X1, X1, width - X1), [style.linewidth.THin])
    lpath = path.path(path.moveto(0, 0))
    for i in range(Xs.shape[0]):
        Y = (2. - states[chrom][i]) / 6.
        lpath.append(path.lineto(Xs[i], Y))
    lpath.append(path.lineto(width, 0))
    lpath.append(path.closepath())
    c.stroke(lpath, [trafo.translate(0, -2.4), style.linewidth.THin])
    return c

def write_TADs(chroms, TADs, sizes, binsize, fname):
    output = open(fname, 'w')
    width = min(binsize / 2, 2500)
    for h, chrom in enumerate(chroms):
        TADs[chrom].sort()
        i = 0
        for domain in TADs[chrom]:
            print >> output, "chr%s\t%i\t%i\t.\t0\t+\t%i\t%i\t0,0,0\t2\t%i,%i\t0,%i" % (
                chrom, domain[0], min(sizes[h], domain[1]), domain[0], domain[0], width, width,
                domain[1] - domain[0] - width)
            i += 1
    output.close()

def write_Scores(chroms, scores, sizes, binsize, fname):
    binsize2 = binsize / 2
    output = open(fname, 'w')
    for h, chrom in enumerate(chroms):
        for i in range(scores[chrom].shape[0]):
            print >> output, "chr%s\t%i\t%i\t%f" % (chrom, scores[chrom]['position'][i] - binsize2,
                                                  min(sizes[h], scores[chrom]['position'][i] + binsize2),
                                                  scores[chrom]['score'][i])
    output.close()



if __name__ == "__main__":
    main()
