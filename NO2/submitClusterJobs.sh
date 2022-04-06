#!/bin/bash 

OUTDIR=./output/

if [ -z "$1" ]; then
  echo "ERROR SUBMITTING JOBS!!!   Must give the name number of jobs!"
  exit
fi

NJOBS=${1}

OUTPUTDIR=${OUTDIR}/logs/
for (( j=0; j<$NJOBS; j++ ))
do
  sbatch -p psanaq -o ${OUTPUTDIR}"job_ConS_3dof_Delta_ens"${j}".log" --wrap="python3 validate.py --multiProc_ind $j --do_2dof 0"
  #sbatch -p psanaq --nodes 1 --ntasks-per-node 10 -o ${OUTPUTDIR}"job_S2N_3dof_Gauss"${j}".log" --wrap="python3 validate.py --multiProc_ind $j --do_ensemble 1 --do_2dof 0"
  sleep 1
done
