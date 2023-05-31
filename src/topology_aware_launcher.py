import subprocess
import time
import os
import ast
import paramiko
import argparse
import json
import mpi_launcher_helper
import psutil

TOPOLOGY_FILE_NAME = "node_to_spine0.txt"
LATENCY_CALCULATOR_EXE = "bin/multispine_latency_calculator"

if os.environ.get("SM_TRAINING_ENV") is None:
    TOPOLOGY_OUTPUT_DIRECTORY = "/fsx/users/viskaria/topology-aware-launcher/"
else:
    TOPOLOGY_OUTPUT_DIRECTORY = "/opt/ml/code/"


def get_hosts_info():
    if os.environ.get("SM_TRAINING_ENV") is None:
        slurm_jobid = 3017
        with open(f"/nfs/node_alloc/{slurm_jobid}/hostlist.txt") as f:
            hosts = [line.rstrip("\n") for line in f.readlines()]
        hosts.sort()
        my_host = subprocess.check_output(['hostname']).decode('utf-8').strip()
    else:
        hosts = json.loads(os.environ['SM_HOSTS'])
        hosts.sort()
        my_host = os.environ['SM_CURRENT_HOST']
    return hosts, my_host


def compute_topology_mapping(hosts, my_host):
    master_hostname = hosts[0].strip()
    is_master = my_host == master_hostname
    entry_point = TOPOLOGY_OUTPUT_DIRECTORY + LATENCY_CALCULATOR_EXE
    if is_master:
        runner = mpi_launcher_helper.MasterRunner(
            user_entry_point=entry_point, 
            user_output_dir=TOPOLOGY_OUTPUT_DIRECTORY, 
            processes_per_host=1, 
            master_hostname=master_hostname, 
            hosts=hosts)
    else:
        runner = mpi_launcher_helper.WorkerRunner(
            user_entry_point=entry_point,
            processes_per_host=1,
            master_hostname=master_hostname,
            current_host=my_host
        )
    runner.run()


def read_spine_to_host(hosts, my_host):
    compute_topology_mapping(hosts, my_host)
    spine_to_host = {}
    filename = TOPOLOGY_OUTPUT_DIRECTORY + TOPOLOGY_FILE_NAME
    print(f'Reading topology mapping from file {filename}')
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            host, spinename = line.split()
            if spinename in spine_to_host:
                spine_to_host[spinename].append(host)
            else:
                spine_to_host[spinename] = [host]
    return spine_to_host, len(lines)


def construct_ranking(pp_degree, dp_degree, optimize_for_pp, dp_major, count, spine_to_host):
    assert count == pp_degree * dp_degree, "pp_degree * dp_degree must match number of hosts"
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


def construct_bad_ranking(pp_degree, dp_degree, dp_major, count, spine_to_host):
    assert count == pp_degree * dp_degree, "pp_degree * dp_degree must match number of hosts"
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


def get_training_info(pp_degree, dp_degree, optimize_for_pp, dp_major, bad_placement, hosts, my_host):
    print('get_training_info')
    spine_to_host, count = read_spine_to_host(hosts, my_host)
    print(f'Output from topology compute: {spine_to_host} count: {count}')
    if bad_placement:
        ranking = construct_bad_ranking(pp_degree, dp_degree, dp_major, count, spine_to_host)
    else:
        ranking = construct_ranking(pp_degree, dp_degree, optimize_for_pp, dp_major, count, spine_to_host)
    print(f'ranking is {ranking}')
    master_addr = ranking[0]
    assert my_host in ranking
    return ranking, master_addr


def launch_mpirun_training(entry_point, ranking, master_addr, training_args):
    master_hostname = hosts[0].strip()
    is_master = my_host == master_hostname
    if is_master:
        runner = mpi_launcher_helper.MasterRunner(
            user_entry_point=entry_point, 
            processes_per_host=8, 
            master_hostname=master_addr, 
            hosts=ranking,
            no_python=False,
            training_args=training_args)
    else:
        runner = mpi_launcher_helper.WorkerRunner(
            user_entry_point=entry_point,
            processes_per_host=8,
            master_hostname=master_addr,
            current_host=my_host
        )
    runner.run()


def launch_torchrun_training(entry_point, ranking, master_addr, training_args):
    count = len(ranking)
    # my_host = os.environ['SM_CURRENT_HOST']
    print(f'my_host = {my_host}')
    rank = ranking.index(my_host)
    command = ['torchrun', f'--nnodes={count}', f'--node_rank={rank}',
               '--nproc_per_node=8', f'--rdzv_endpoint={master_addr}:29400',
               '--rdzv_id=100', entry_point]
    command.extend(training_args)
    command_str = ' '.join(command)
    print(f'Launching job with command: {command_str}')
    subprocess.run(command)


def validate_args(args):
    if args.launcher != "mpirun" and args.launcher != "torchrun":
        print('Argument launcher set to unsupported value, defaulting to torchrun.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp-degree', type=int, required=True)
    parser.add_argument('--dp-degree', type=int, required=True)
    parser.add_argument('--optimize-for-pp', type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument('--dp-major', type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument('--bad-placement', action='store_true')
    parser.add_argument('--entry-point', type=str, required=True)
    parser.add_argument('--launcher', type=str, required=True)
    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = parse_args()
    print(f'unknown args: {unknown}')
    hosts, my_host = get_hosts_info()
    ranking, master_addr = get_training_info(args.pp_degree,
                                             args.dp_degree,
                                             args.optimize_for_pp,
                                             args.dp_major,
                                             args.bad_placement,
                                             hosts, 
                                             my_host)

    if args.launcher == "mpirun":
        launch_mpirun_training(args.entry_point, ranking, master_addr, unknown)
    else:
        launch_torchrun_training(args.entry_point, ranking, master_addr, unknown)

    print('Job finished!')