import subprocess
import time
import os
import ast
import paramiko
import argparse
import json
import mpi


TOPOLOGY_OUTPUT_DIRECTORY = "/opt/ml/code/"
TOPOLOGY_FILE_NAME = "cluster_topology.txt"

def compute_topology_mapping():
    sm_training_env = json.loads(os.environ['SM_TRAINING_ENV'])
    is_master = sm_training_env.get("is_master")
    master_hostname = sm_training_env.get("master_hostname")
    my_host = os.environ['SM_CURRENT_HOST']
    hosts = json.loads(os.environ['SM_HOSTS'])
    hosts.sort()
    if is_master:
        print('master')
        runner = mpi.MasterRunner("bin/ring_latency_calculator", TOPOLOGY_OUTPUT_DIRECTORY, 1, master_hostname, hosts)
        print('master init done')
        runner.run()
    else:
        print('worker')
        runner = mpi.WorkerRunner(
            "bin/ring_latency_calculator",
            TOPOLOGY_OUTPUT_DIRECTORY,
            1,
            master_hostname,
            my_host
        )
        print('worker init done')
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


def construct_bad_ranking(count, spine_to_host):
    ranking = ["" for i in range(count)]
    while count > 0:
        for spine in spine_to_host:
            if spine_to_host[spine]:
                ranking[count - 1] = spine_to_host[spine][0]
                spine_to_host[spine].pop(0)
                count -= 1
    return ranking


def get_training_info(pp_degree, dp_degree, optimize_for_pp, dp_major, bad_placement):
    spine_to_host, count = read_spine_to_host()
    print(f'Output from topology compute: {spine_to_host} count: {count}')
    my_host = os.environ['SM_CURRENT_HOST']
    if False:
        ranking = construct_bad_ranking(count, spine_to_host)
    else:
        ranking = construct_ranking(pp_degree, dp_degree, optimize_for_pp, dp_major, count, spine_to_host)
    print(f'ranking is {ranking}')
    master_addr = ranking[0]
    assert my_host in ranking
    rank = ranking.index(my_host)
    return count, master_addr, rank


if __name__ == "__main__":
    print('In custom launcher')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp-degree', type=int, required=True)
    parser.add_argument('--dp-degree', type=int, required=True)
    parser.add_argument('--optimize-for-pp', type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument('--dp-major', type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument('--bad-placement', action = 'store_true')
    parser.add_argument('--entry-point', type=str, required=True)
    parser.add_argument('--conf', type=str, required=True)
    args, unknown = parser.parse_known_args()

    count, master_addr, rank = get_training_info(args.pp_degree, args.dp_degree, args.optimize_for_pp, args.dp_major, args.bad_placement)

    command = ['torchrun', f'--nnodes={count}', f'--node_rank={rank}', '--nproc_per_node=8', f'--rdzv_endpoint={master_addr}:29400', '--rdzv_id=100', args.entry_point, '--conf', args.conf]
    command_str = ' '.join(command)
    print(f'Launching job with command: {command_str}')
    subprocess.run(command)

    print('Job finished!')