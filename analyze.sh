#!/bin/bash

# source analyze.sh PROJECT OUT_PREFIX

CHROMS=($(eval {1..19}) X)
NCPUS=1
BINSIZE=10000
# these should be multiples of the binsize
WIDTH=40000
WINDOW=500000
# bedgraph with strong enrichment in A-compartment or B-compartment
SCOREFILE=
ENRICHMENT=A
# chromosome length file (name\tlength)
LENFILE=

# compile cython functions
if [! -e library.so ]; then
  python setup.py --inplace
fi

# create heatmap file
mpirun -np ${NCPUS} python make_binning_HMs.py $1 $1.HM ${BINSIZE}

# find TADs
python find_TADs_from_HM.py $1.HM $2_TAD ${WIDTH} ${WINDOW}

# find individual chromosome compartment scores
for I in ${CHROMS[*]}; do
  mpirun -np ${NCPUS} python find_optimized_multires_compartment_from_HM.py $1.HM $2_Comp_chr${I} -c ${I} -f ${BINSIZE}
done

# compile compartment scores and orient using external data
python compile_optimized_compartments $2_Comp_chr $2_Comp_All $LENFILE $SCOREFILE $ENRICHMENT
