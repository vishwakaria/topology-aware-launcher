import torch
import torch.distributed as dist
import os
import sys
import time


master_addr = os.environ['MASTER_ADDR']
print(f'master_addr = {master_addr}')
print(f'arguments are {sys.argv}')

torch.distributed.init_process_group(backend="gloo")

my_host = os.environ['SM_CURRENT_HOST']

print(f'Hello from host {my_host} rank {dist.get_rank()}')
time.sleep(1)
print('hello.py exiting')
