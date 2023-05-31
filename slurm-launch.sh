#!/bin/bash
set -ex

CONTAINER_NAME=${1:-"viskaria_mds"}
RUN_PROGRAM=${2:-"/fsx/users/viskaria/topology-aware-launcher/src/topology_aware_launcher.py"}
HOSTFILE="/nfs/node_alloc/$SLURM_JOBID/hostlist.txt"

MASTER=$(head -n 1 $HOSTFILE)

cat $HOSTFILE

# MPI run the herring job
RUN_CMD="/opt/conda/bin/python3 /fsx/users/viskaria/topology-aware-launcher/src/topology_aware_launcher.py \
        --pp-degree 1 \
        --dp-degree 4 \
        --optimize-for-pp True \
        --dp-major True \
        --entry-point /fsx/users/viskaria/topology-aware-launcher/src/hello_mpirun.py \
        --launcher mpirun"

docker exec ${CONTAINER_NAME} bash -c "${RUN_CMD}"

exit 0
