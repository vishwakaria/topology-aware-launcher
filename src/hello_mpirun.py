import torch
import torch.distributed as dist
import os
import sys
import time
import subprocess

os.environ['RANK'] = os.getenv('OMPI_COMM_WORLD_RANK')
os.environ['WORLD_SIZE'] = os.getenv('OMPI_COMM_WORLD_SIZE')
os.environ['LOCAL_RANK'] = os.getenv('OMPI_COMM_WORLD_LOCAL_RANK')
os.environ['MASTER_PORT'] = str(12334)

master_addr = os.environ['MASTER_ADDR']
print(f'master_addr = {master_addr}')
print(f'arguments are {sys.argv}')

torch.distributed.init_process_group(backend="gloo")

# my_host = os.environ['SM_CURRENT_HOST']
my_host = subprocess.check_output(['hostname']).decode('utf-8').strip()

print(f'Hello from host {my_host} rank {dist.get_rank()}')
time.sleep(1)
print('hello.py exiting')
