#!/bin/bash 

OUTDIR=./output/

if [ -z "$1" ]; then
  echo "ERROR SUBMITTING JOBS!!!   Must give the name number of jobs!"
  exit
fi

NJOBS=${1}

declare -A array
for constant in 0 1 2 8 9 15 16 17 26 27 34 35 41 42 43 
do
  array[$constant]=1
done

OUTPUTDIR=${OUTDIR}/logs/
for (( j=0; j<$NJOBS; j++ ))
do
  if [[ $j -gt 25 ]]
  then
    sbatch -p psanaq --time 9-23:59 -o ${OUTPUTDIR}"job_mode_"${j}".log" --wrap="python3 mode_search.py --multiProc_ind $j"
  elif [[ ${array[$j]} ]]
  then
    sbatch -p psanaq --ntasks-per-node 16 --time 9-23:59 -o ${OUTPUTDIR}"job_mode_"${j}".log" --wrap="python3 mode_search.py --multiProc_ind $j"
  else
    sbatch -p psanaq --ntasks-per-node 10 --time 9-23:59 -o ${OUTPUTDIR}"job_mode_"${j}".log" --wrap="python3 mode_search.py --multiProc_ind $j"
  fi
  #sbatch -p psanaq --nodes 1 --ntasks-per-node 10 -o ${OUTPUTDIR}"job_S2N_3dof_Gauss"${j}".log" --wrap="python3 validate.py --multiProc_ind $j --do_ensemble 1 --do_2dof 0"
  sleep 1
done


#OUTPUTDIR=${OUTDIR}/logs/
#for (( j=0; j<$NJOBS; j++ ))
#do
#  sbatch -p psanaq --ntasks-per-node 10 --time 9-23:59 -o ${OUTPUTDIR}"job_mode"${j}".log" --wrap="python3 mode_search.py --multiProc_ind $j"
#  #sbatch -p psanaq --nodes 1 --ntasks-per-node 10 -o ${OUTPUTDIR}"job_S2N_3dof_Gauss"${j}".log" --wrap="python3 validate.py --multiProc_ind $j --do_ensemble 1 --do_2dof 0"
#  sleep 1
#done
