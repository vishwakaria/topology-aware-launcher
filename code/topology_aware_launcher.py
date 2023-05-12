import subprocess
import time
import os
import ast
import paramiko
import argparse
import json
import mpi_launcher_helper
import psutil

TOPOLOGY_OUTPUT_DIRECTORY = "/opt/ml/code/"
TOPOLOGY_FILE_NAME = "node_to_spine.txt"

sm_training_env = json.loads(os.environ['SM_TRAINING_ENV'])
is_master = sm_training_env.get("is_master")
master_hostname = sm_training_env.get("master_hostname")
my_host = os.environ['SM_CURRENT_HOST']
hosts = json.loads(os.environ['SM_HOSTS'])
hosts.sort()

def compute_topology_mapping():
    if is_master:
        runner = mpi_launcher_helper.MasterRunner(
            user_entry_point="bin/multispine_latency_calculator", 
            user_output_dir=TOPOLOGY_OUTPUT_DIRECTORY, 
            processes_per_host=1, 
            master_hostname=master_hostname, 
            hosts=hosts)
    else:
        runner = mpi_launcher_helper.WorkerRunner(
            user_entry_point="bin/multispine_latency_calculator",
            processes_per_host=1,
            master_hostname=master_hostname,
            current_host=my_host
        )
    runner.run()


def read_spine_to_host():
    compute_topology_mapping()
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


def get_training_info(pp_degree, dp_degree, optimize_for_pp, dp_major, bad_placement):
    spine_to_host, count = read_spine_to_host()
    print(f'Output from topology compute: {spine_to_host} count: {count}')
    my_host = os.environ['SM_CURRENT_HOST']
    if bad_placement:
        ranking = construct_bad_ranking(pp_degree, dp_degree, dp_major, count, spine_to_host)
    else:
        ranking = construct_ranking(pp_degree, dp_degree, optimize_for_pp, dp_major, count, spine_to_host)
    print(f'ranking is {ranking}')
    master_addr = ranking[0]
    assert my_host in ranking
    return ranking, master_addr


def launch_mpirun_training(entry_point, ranking, master_addr, training_args):
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
    my_host = os.environ['SM_CURRENT_HOST']
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp-degree', type=int, required=True)
    parser.add_argument('--dp-degree', type=int, required=True)
    parser.add_argument('--optimize-for-pp', type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument('--dp-major', type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument('--bad-placement', action='store_true')
    parser.add_argument('--entry-point', type=str, required=True)
    parser.add_argument('--launcher', type=str, required=True)
    args, unknown = parser.parse_known_args()
    print(f'unknown args: {unknown}')
    ranking, master_addr = get_training_info(args.pp_degree,
                                             args.dp_degree,
                                             args.optimize_for_pp,
                                             args.dp_major,
                                             args.bad_placement)

    if args.launcher == "mpirun":
        launch_mpirun_training(args.entry_point, ranking, master_addr, unknown)
    else:
        launch_torchrun_training(args.entry_point, ranking, master_addr, unknown)

    print('Job finished!')