import subprocess
import os
import argparse
import json
import aws_topology.mpi_launcher_helper as mpi_launcher_helper
from collections import namedtuple

## Define constants
TOPOLOGY_FILE_NAME = "node_to_spine.txt"
LATENCY_CALCULATOR_EXE = "/bin/multispine_latency_calculator"
if os.environ.get("SM_TRAINING_ENV") is None:
    TOPOLOGY_OUTPUT_DIRECTORY = "/fsx/users/viskaria/topology-aware-launcher/src/"
else:
    TOPOLOGY_OUTPUT_DIRECTORY = "/opt/ml/code/"

ModelParallelParams = namedtuple(
    "ModelParallelParams",
    ["pp_degree", "dp_degree", "optimize_for_pp", "dp_major"],
)


def read_file_to_list(filename):
    with open(filename) as f:
        l = f.read().splitlines()
        return l

def get_hosts_info(hostfile=None):
    if os.environ.get("SM_TRAINING_ENV") is not None:
        print("Running on SageMaker Platform. Hostlist will be ignored if provided.")
        hosts = json.loads(os.environ["SM_HOSTS"])
        hosts.sort()
        my_host = os.environ["SM_CURRENT_HOST"]
        return hosts, my_host
    if hostfile is None:
        raise ValueError("Hostfile not provided on non-SageMaker environment.")
    else:
        hostlist = read_file_to_list(hostfile)
        my_host = subprocess.check_output(["hostname"]).decode("utf-8").strip()
        print(f'hostlist: {hostlist}, my_host: {my_host}')
        return hostlist, my_host


def compute_topology_mapping(hosts, my_host):
    master_hostname = hosts[0].strip()
    is_master = my_host == master_hostname
    script_directory = os.path.dirname(os.path.abspath(__file__))
    entry_point = script_directory + "/" + LATENCY_CALCULATOR_EXE
    print(f'latency entry_point: {entry_point}')
    if is_master:
        runner = mpi_launcher_helper.MasterRunner(
            user_entry_point=entry_point,
            user_output_dir=TOPOLOGY_OUTPUT_DIRECTORY,
            processes_per_host=1,
            master_hostname=master_hostname,
            hosts=hosts,
        )
    else:
        runner = mpi_launcher_helper.WorkerRunner(
            user_entry_point=entry_point,
            processes_per_host=1,
            master_hostname=master_hostname,
            current_host=my_host,
        )
    runner.run()


def get_spine_to_host_mapping(hosts, my_host):
    compute_topology_mapping(hosts, my_host)
    spine_to_host = {}
    filename = TOPOLOGY_OUTPUT_DIRECTORY + TOPOLOGY_FILE_NAME
    print(f"Reading topology mapping from file {filename}")
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            host, spinename = line.split()
            if spinename in spine_to_host:
                spine_to_host[spinename].append(host)
            else:
                spine_to_host[spinename] = [host]
    print(f"Output from topology compute: {spine_to_host}")
    return spine_to_host, len(lines)


def get_optimized_node_ranking(
    pp_degree, dp_degree, optimize_for_pp, dp_major, hosts, my_host
):
    spine_to_host, count = get_spine_to_host_mapping(hosts, my_host)
    assert (
        count == pp_degree * dp_degree
    ), "pp_degree * dp_degree must match number of hosts"
    if not dp_major:
        pp_degree, dp_degree = dp_degree, pp_degree
    flattened = []
    for spine in spine_to_host:
        for hostname in spine_to_host[spine]:
            flattened.append(hostname)
    if not optimize_for_pp:
        return flattened
    ranking = ["" for i in range(count)]
    k = 0
    for i in range(dp_degree):
        for j in range(pp_degree):
            ranking[i + j * dp_degree] = flattened[k]
            k += 1
    return ranking


def simulate_bad_node_ranking(pp_degree, dp_degree, dp_major, hosts, my_host):
    spine_to_host, count = get_spine_to_host_mapping(hosts, my_host)
    assert (
        count == pp_degree * dp_degree
    ), "pp_degree * dp_degree must match number of hosts"
    if not dp_major:
        pp_degree, dp_degree = dp_degree, pp_degree
    ranking = ["" for i in range(count)]
    spines = list(spine_to_host.keys())
    k = 0
    for i in range(dp_degree):
        for j in range(pp_degree):
            while len(spine_to_host[spines[k]]) == 0:
                k = (k + 1) % len(spines)
            ranking[i + j * dp_degree] = spine_to_host[spines[k]][0]
            spine_to_host[spines[k]].pop(0)
            k = (k + 1) % len(spines)
    return ranking


def get_training_info(
    pp_degree, dp_degree, optimize_for_pp, dp_major, bad_placement, hosts, my_host
):
    if bad_placement:
        ranking = simulate_bad_node_ranking(
            pp_degree, dp_degree, dp_major, hosts, my_host
        )
    else:
        ranking = get_optimized_node_ranking(
            pp_degree, dp_degree, optimize_for_pp, dp_major, hosts, my_host
        )
    print(f"ranking is {ranking}")
    master_addr = ranking[0]
    assert my_host in ranking
    return ranking, master_addr


def launch_mpirun_training(
    entry_point, ranking, master_addr, training_args, hosts, my_host
):
    master_hostname = hosts[0].strip()
    is_master = my_host == master_hostname
    if is_master:
        runner = mpi_launcher_helper.MasterRunner(
            user_entry_point=entry_point,
            processes_per_host=8,
            master_hostname=master_addr,
            hosts=ranking,
            no_python=False,
            training_args=training_args,
        )
    else:
        runner = mpi_launcher_helper.WorkerRunner(
            user_entry_point=entry_point,
            processes_per_host=8,
            master_hostname=master_addr,
            current_host=my_host,
        )
    runner.run()


def launch_torchrun_training(
    entry_point, ranking, master_addr, training_args, hosts, my_host
):
    count = len(ranking)
    # my_host = os.environ['SM_CURRENT_HOST']
    print(f"my_host = {my_host}")
    rank = ranking.index(my_host)
    command = [
        "torchrun",
        f"--nnodes={count}",
        f"--node_rank={rank}",
        "--nproc_per_node=8",
        f"--rdzv_endpoint={master_addr}:29400",
        "--rdzv_id=100",
        entry_point,
    ]
    if training_args:
        command.extend(training_args)
    command_str = " ".join(command)
    print(f"Launching job with command: {command_str}")
    subprocess.run(command)


def launch_training(
    entry_point,
    launcher,
    params: ModelParallelParams,
    training_args=None,
    hostfile=None,
):
    hosts, my_host = get_hosts_info(hostfile)
    ranking, master_addr = get_training_info(
        params.pp_degree,
        params.dp_degree,
        params.optimize_for_pp,
        params.dp_major,
        False,
        hosts,
        my_host,
    )
    if launcher == "mpirun":
        launch_mpirun_training(
            entry_point, ranking, master_addr, training_args, hosts, my_host
        )
    elif launcher == "torchrun":
        launch_torchrun_training(
            entry_point, ranking, master_addr, training_args, hosts, my_host
        )
    else:
        print("Argument launcher set to unsupported value, defaulting to torchrun.")
