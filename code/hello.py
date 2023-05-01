import torch
import torch.distributed as dist
import os
import sys
import time

print(f'arguments are {sys.argv}')

torch.distributed.init_process_group(backend="gloo")

my_host = os.environ['SM_CURRENT_HOST']

print(f'Hello from host {my_host} rank {dist.get_rank()}')
time.sleep(5)
print('hello.py exiting')
