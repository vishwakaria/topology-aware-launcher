# from aws_topology import topology_aware_launcher
import topology_aware_launcher

hostfile = "/nfs/node_alloc/3368/hostlist.txt"

mp = topology_aware_launcher.ModelParallelParams(
    pp_degree=1, dp_degree=4, optimize_for_pp="True", dp_major="True"
)

topology_aware_launcher.launch_training(
    "/fsx/users/viskaria/topology-aware-launcher/scripts/hello.py",
    "torchrun",
    mp,
    ["--batch-size", "8"],
    hostfile,
)
