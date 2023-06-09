# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains functionality related to distributed training using
MPI (Message Passing Interface)."""
import os
import subprocess
import time
import psutil
import asyncio
from asyncio.subprocess import PIPE

if os.environ.get("SM_TRAINING_ENV") is None:
    SAGEMAKER_MODE = False
else:
    SAGEMAKER_MODE = True

MPI_FINISHED_STATUS_FILE = "/tmp/done"


class WorkerRunner:
    """Runner responsible for preparing MPI distributed training and waiting for MPI
    master execution to finish.
    """

    def __init__(
        self, 
        user_entry_point,  
        processes_per_host, 
        master_hostname, 
        current_host,
    ):
        """Initialize a WorkerRunner, which is responsible for preparing distributed
        training with MPI and waiting for MPI master execution to finish.

        Args:
            user_entry_point (str): The name of the user entry point.
            master_hostname (str): The master hostname.
            current_hostname (str): Current hostname.
        """
        print('Init worker runner')
        self._user_entry_point = user_entry_point
        self._processes_per_host = processes_per_host
        self._master_hostname = str(master_hostname)
        self._current_host = str(current_host)

    def run(self):  # type: (bool, bool) -> None # pylint: disable=unused-argument
        """The WorkerRunner proceeds as following:

        - wait for the MPI Master to create its SSH daemon
        - start its SSH daemon
        - monitor the MPI orted process and wait it to finish the MPI execution
        - wait for the status file from master
        - Exit once orted process is finished and status file is found.
        """
        print("Worker waiting for MPI Master to create SSH daemon.")
        self._wait_master_to_start()
        print("MPI Master online, creating SSH daemon.")

        print("Writing environment variables to /etc/environment for the MPI process.")
        _write_env_vars_to_file()

        self._sshd = _start_sshd_daemon()

        print("Waiting for MPI process to finish.")
        gone, alive = _wait_orted_process_to_finish()
        print(f"Reporting status for ORTEd process. gone: {gone} alive: {alive}")
        print("Orted process exited")
        print("MPI process finished, killing SSHD")
        self._sshd.kill()

    def _wait_master_to_start(self):  # type: () -> None
        while not _can_connect(self._master_hostname):
            time.sleep(1)


def _write_env_vars_to_file():  # type: () -> None
    with open("/etc/environment", "a") as f:
        for name in os.environ:
            f.write("{}={}\n".format(name, os.environ.get(name)))


def _on_terminate(proc):
    print("Invoked on_terminate from psutil.wait_for_procs")
    print("process {} terminated with exit code {}".format(proc, proc.returncode))


def _wait_orted_process_to_finish():  # type: () -> None
    orted = _orted_process()
    print("Orted process found %s", orted)
    print("Waiting for orted process %s", orted)
    if orted is not None:
        gone, alive = psutil.wait_procs(orted, callback=_on_terminate)
        return gone, alive
    else:
        print("orted is null")
    return None, None

def _orted_process():  # pylint: disable=inconsistent-return-statements
    """Wait a maximum of 20 minutes for orted process to start."""
    # the wait time here should be set to a dynamic value according to cluster size
    for _ in range(20 * 60):
        procs = [p for p in psutil.process_iter(attrs=["name"]) if p.info["name"] == "orted"]
        if procs:
            print("Process[es]: %s", procs)
            return procs
        time.sleep(0.01)


class MasterRunner():
    """Responsible for preparing MPI distributed training and synchronizing work
    with the Workers.
    """

    def __init__(
        self,
        user_entry_point,
        processes_per_host,
        master_hostname,
        hosts,
        training_args=None,
        user_output_dir=None,
        interval=1,
        timeout_in_seconds=60 * 60,
        num_processes=None,
        no_python=True
    ):
        print('Init master')
        self._user_entry_point = user_entry_point
        self._user_output_dir = user_output_dir
        self._processes_per_host = processes_per_host
        self._master_hostname = master_hostname
        self._hosts = hosts
        self._training_args = training_args
        self._processes_per_host = processes_per_host
        self._num_processes = num_processes
        self._interval = interval
        self.timeout_in_seconds = timeout_in_seconds
        self._no_python = no_python

    def _setup(self):  # type: () -> None
        print("Starting MPI run as master node.")
        print("Creating SSH daemon.")
        self._sshd = _start_sshd_daemon()

        self._wait_for_workers()

    def _wait_for_workers(self):  # type: () -> None
        print("Waiting for MPI workers to establish their SSH connections")

        workers = [host for host in self._hosts if host != self._master_hostname]
        for host in workers:
            print(f"master checking connection to {host}")
            while not _can_connect(host):
                time.sleep(self._interval)
            print("Worker %s available for communication", host)

    def _create_command(self):
        num_hosts = len(self._hosts)
        num_processes = self._processes_per_host * num_hosts
        if self._processes_per_host == 1:
            host_list = self._hosts
        else:
            host_list = ["%s:%s" % (host, self._processes_per_host) for host in self._hosts]
        msg = "Env Hosts: %s Hosts: %s process_per_hosts: %s num_processes: %s"
        print(msg, self._hosts, host_list, self._processes_per_host, num_processes)

        command = [
            "/opt/amazon/openmpi/bin/mpirun",
            "--host",
            ",".join(host_list),
            "-np",
            str(num_processes),
            "--allow-run-as-root",
            "--mca",
            "orte_abort_on_non_zero_status",
            "1",
            "--mca", 
            "btl_tcp_if_exclude", 
            "lo,docker0", 
            "-x",
            "MASTER_ADDR=%s" % self._master_hostname,
        ]
        if self._no_python:
            command.extend([self._user_entry_point])
        else:
            command.extend(["python", self._user_entry_point])

        if self._user_output_dir:
            command.extend([self._user_output_dir])

        if self._training_args:
            command.extend(self._training_args)

        command = " ".join(command)
        print(f'Running command: `{command}`')
        return command

    async def _run_async(self, cmd, processes_per_host):
        proc = await asyncio.create_subprocess_shell(cmd)
        print("Waiting for the process to finish and give a return code.")
        return_code = await proc.wait()
        print(f"Done waiting for a return code. Received {return_code} from exiting process.")
        return return_code, proc

    def _create_process(self, cmd, processes_per_host):
        rc, proc = asyncio.run(
            self._run_async(
                cmd,
                processes_per_host
            )
        )
        return rc, proc

    def run(self):
        self._setup()

        cmd = self._create_command()
        _, process_spawned = self._create_process(
            cmd,
            self._processes_per_host,
        )
        self._sshd.kill()
        return process_spawned


def _start_sshd_daemon():  # type: () -> None
    sshd_executable = "/usr/sbin/sshd"

    if not os.path.exists(sshd_executable):
        raise RuntimeError("SSH Daemon not found")
    proc = subprocess.Popen([sshd_executable, "-D"])
    return proc


def _can_connect(host, port=22):  # type: (str, int) -> bool
    ssh_command = f"ssh {host}"
    ssh_process = subprocess.Popen(ssh_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = ssh_process.communicate()
    if ssh_process.returncode == 0:
        print("Can connect to host %s", host)
        return True
    else:
        print("Cannot connect to host %s", host)
        return False

