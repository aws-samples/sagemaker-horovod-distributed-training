import stat

import shlex
import socket
import sys

import textwrap
from contextlib import contextmanager
import signal

import os
import subprocess
import time
import argparse

import sagemaker_containers

from retrying import retry
from sagemaker_containers import _logging
from sagemaker_containers.beta import framework

logger = _logging.get_logger()

# MPI files.
_MPI_SCRIPT = "/mpi_script.sh"
_MPI_IS_RUNNING = "/mpi_is_running"
_MPI_IS_FINISHED = "/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/libchangehostname.so"


def _change_hostname(current_host):
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.

    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("/change-hostname.sh {}".format(current_host))


def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _setup_mpi_environment(env):
    _change_hostname(env.current_host)
    _start_ssh_daemon()


def _can_connect(host, port, s):
    try:
        logger.debug("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        logger.debug("can connect to host %s", host)
        return True
    except socket.error:
        logger.debug("can't connect to host %s", host)
        return False


def _create_mpi_script(env, train_script):
    """Creates a MPI script with user provided information.

        For distributed training: the 'master node' runs mpirun with this script,
        '/mpi_script.sh'.

        This script creates a file '/mpi_is_running' that worker nodes use to
        determine whether training # (started by MPI from the master node) is still running.

        Processes on worker nodes use # /mpi_is_finished file to determine when to exit.

    Args:
        env (TrainingEnv): an instance of the training environment.
    """
    hyperparameters = framework.mapping.to_cmd_args(env.hyperparameters)
    channels = framework.mapping.to_cmd_args(env.channel_input_dirs)

    python_cmd = [sys.executable, train_script]
    # python_cmd = [sys.executable, '-m', 'mpi4py', '-m', train_script]
    python_cmd.extend(hyperparameters)
    python_cmd.extend(channels)

    content = textwrap.dedent("""#!/usr/bin/env bash
touch /mpi_is_running
%s
EXIT_CODE=$?
touch /mpi_is_finished
exit ${EXIT_CODE}
""" % ' '.join(python_cmd))

    with open(_MPI_SCRIPT, 'w') as w:
        w.write(content)

    st = os.stat(_MPI_SCRIPT)
    os.chmod(_MPI_SCRIPT, st.st_mode | stat.S_IEXEC)


class MPIMaster(object):
    """ MPI Master"""

    def __init__(self, env, process_per_host):
        self.env = env
        self.process_per_host = process_per_host

    def _wait_for_worker_nodes_to_start_sshd(self, hosts, interval=1, timeout_in_seconds=180):
        with timeout(seconds=timeout_in_seconds):
            while hosts:
                print("hosts that aren't SSHable yet: %s", str(hosts))
                for host in hosts:
                    ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    if _can_connect(host, 22, ssh_socket):
                        print(">>> Host: {} is sshable now.".format(host))
                        hosts.remove(host)
                time.sleep(interval)

    def _run_mpi_on_all_nodes(self):
        mpi_command = self._build_mpi_command()
        cmd = shlex.split(mpi_command)

        framework.logging.log_script_invocation(cmd, self.env.to_env_vars(), logger)

        print("MPI Command: {}".format(mpi_command))
        with open(_MPI_SCRIPT) as f:
            print('Running user script:\n\n%s', f.read())

        subprocess.check_call(cmd)

    def _build_mpi_command(self):

        is_gpu = self.env.num_gpus if self.env.num_gpus > 0 else 1

        num_hosts = len(self.env.hosts)
        num_processes = self.process_per_host * num_hosts

        # By default, use one process per GPU, or one process per node (if training with CPU).
        host_list = self.env.hosts if self.process_per_host == 1 else \
            [host + ':{}'.format(self.process_per_host) for host in self.env.hosts]

        print("Env Hosts: {} Hosts: {} process_per_hosts: {} num_processes: {}".format(self.env.hosts, host_list,
                                                                                       self.process_per_host,
                                                                                       num_processes))
        credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']

        print('network interface name: %s', self.env.network_interface_name)

        mpi_command = 'mpirun --host {}'.format(",".join(host_list)) \
                      + " -np {} ".format(num_processes) \
                      + " --allow-run-as-root" \
                      + " --display-map" \
                      + " --tag-output" \
                      + " -mca btl_tcp_if_include {}".format(self.env.network_interface_name) \
                      + " -mca oob_tcp_if_include {}".format(self.env.network_interface_name) \
                      + " -x NCCL_SOCKET_IFNAME={}".format(self.env.network_interface_name) \
                      + " --mca plm_rsh_no_tree_spawn 1" \
                      + " -mca orte_abort_on_non_zero_status 1" \
                      + " -x NCCL_MIN_NRINGS=8 -x NCCL_DEBUG=INFO" \
                      + " -x LD_LIBRARY_PATH -x PATH" \
                      + " -x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY)
        for v in credential_vars:
            if v in os.environ:
                mpi_command += " -x {}".format(v)

        for name, value in self.env.to_env_vars().items():
            mpi_command += ' -x {}="{}"'.format(name, value)

        mpi_command += " {}".format(_MPI_SCRIPT)

        print("MPI Command: {}".format(mpi_command))
        return mpi_command

    def __call__(self):
        self._wait_for_worker_nodes_to_start_sshd(self.env.hosts.copy())
        self._run_mpi_on_all_nodes()

    def is_master(self, hosts, current_host):
        print("Hosts: " + str(hosts) + " current host: " + str(current_host))
        return current_host == sorted(list(hosts))[0]


class MPIWorker(object):
    """ MPI Worker"""

    @retry(stop_max_delay=30000 * 1000, wait_fixed=1000, retry_on_result=lambda result: result is False)
    def _wait_for_mpi_to_start_running(self):
        return os.path.isfile(_MPI_IS_RUNNING)

    @retry(wait_fixed=5000, retry_on_result=lambda result: result is False)
    def _wait_until_mpi_stops_running(self):
        return os.path.isfile(_MPI_IS_FINISHED)

    def __call__(self, env):
        current_host = env.current_host

        print("Worker node %s is waiting for MPI to start training process", current_host)
        self._wait_for_mpi_to_start_running()

        print("MPI started training process on worker node %s", current_host)

        self._wait_until_mpi_stops_running()
        print("Training process started by MPI on worker node %s stopped", current_host)


def _horovod_run(env, processes_per_host, train_script):
    print("MPI requested with process per hosts: {}"
          .format(processes_per_host))

    _setup_mpi_environment(env)
    _create_mpi_script(env, train_script)

    mpi_master = MPIMaster(env, processes_per_host)
    if mpi_master.is_master(env.hosts, env.current_host):
        print("Inside Master")
        mpi_master()
    else:
        print("Inside Worker")
        MPIWorker()(env)


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.
    Usage:
    with timeout(seconds=5):
        my_slow_function(...)
    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):  # pylint: disable=W0613
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, limit)
        yield
    finally:
        signal.alarm(0)


def execute_horovod_script(train_script, processes_per_host):
    print("Starting Horovod training with Horovod train script: {} Num processes per host: {}".format(train_script,
                                                                                                      processes_per_host))
    env = sagemaker_containers.training_env()
    _horovod_run(env, processes_per_host, train_script)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--horovod-train-script', required=True, help="Name of the horovod training script to execute.")
    parser.add_argument('--num-processes-per-host', default=1, help="Number of processes per host.", type=int)

    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    execute_horovod_script(train_script=args.horovod_train_script,
                           processes_per_host=args.num_processes_per_host)


if __name__ == "__main__":
    main()
