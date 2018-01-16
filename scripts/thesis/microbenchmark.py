#!/usr/bin/env python3

import subprocess
import socket
import signal
import time
import shlex
import csv

class BMRServer:
    """Abstraction of the callback protocol between host and benchmark

    The benchmark opens a TCP connection to the host to signal it is ready.
    The host waits for this event using `wait_callback()` and signals
    to the benchmark that it can start.
    After the benchmark finishes running, it notifies the host.

    Use case: allows host to profile only while the actual benchmark
              is running.
    request """

    class CallbackConn:

        def __init__(self, conn):
            self.conn = conn

        def close(self):
            print("bmrserver: conn: closing connection")
            self.conn.close()

        def start_and_wait(self):
            self.conn.send("START\n".encode("ascii"))
            msg = self.conn.recv(1024)
            msg = msg.decode("ascii")
            msg.strip()
            print("bmrserver: conn: received: {}".format(msg))
            self.close()
            if msg == "FAIL":
                raise Exception("benchmark signaled failure")

    def __init__(self, listen_ip="192.168.122.1", listen_port=1235):
        print("bmrserver: start opening socket")
        try:
            ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            #ls.settimeout(10) #benchmark must be ready after 10s
            ls.bind((listen_ip, listen_port))
            ls.listen(1)
            self.ls = ls
        except Exception as e:
            raise Exception("error setting up callback socket: {}".format(e))
        print("bmrserver: listening")

    def wait_callback(self):
        print("bmrserver: waiting for callback from benchmark")
        conn, client = self.ls.accept()
        print("bmrserver: callback connection from client {}".format(client))
        return BMRServer.CallbackConn(conn)


class PerfProfiler:

    csv_delimiter = ";"

    def __init__(self, events, cores, tempfile_path):

        self.tempfile_path = tempfile_path
        self.cores = cores
        self.events = events

        core_string = ",".join(["{}".format(i) for i in cores])
        events_string = ",".join([ "{}".format(e) for e in events ])

        cmd = [ "perf", "stat",
                "-o", tempfile_path,
                "-x", self.csv_delimiter,
                "-C", core_string,
                "--per-core",
                "-e", events_string,
              ]
        self.cmd = cmd
        self.proc = None
        self.result = None

    def __enter__(self):
        print(self.cmd)
        self.proc = subprocess.Popen(self.cmd)

    def __exit__(self, exc_type, exc, tb):
        self.proc.send_signal(signal.SIGINT)
        self.proc.wait()

        # FIXME: if perf didn't run long enough (really fast benchmark), it won't produce a csv
        # FIXME  the data will probably not be very useful in that case,
        #        but currently, we just crash with an exception.
        with open(self.tempfile_path, 'r') as f:
            r = csv.reader(f, delimiter=self.csv_delimiter)
            # skip first two lines, they are comments
            next(r)
            next(r)

            per_core = []
            # grouped by core, same order of events
            for ci in range(0, len(self.cores)): #!ordering with hyperthreading
                rlist = []
                for ei in range(0,len(self.events)):
                    row = next(r)
                    #print(row)
                    core, _, val_str, _, event, sample_time, perc  = row[0:7]
                    if perc != "100.00":
                        print(row)
                        raise Exception("counter for {} was not running full time".format(event))
                    if event != self.events[ei]:
                        raise Exception("row has unexpected event")
                    rlist.append(int(val_str))
                per_core.append(rlist)
            self.result = per_core


class OSVRun:
    """Abstraction for profiling a benchmark running in OSv
    The KVM threads are pinned to specific cores.
    """

    def __init__(self, profiler, cores, osv_cmdline, osv_image=None):

        core_map = []
        for c in cores:
            core_map.append((len(core_map), c))

        affinity_args = ["{}:{}".format(v,p) for v, p in core_map]
        affinity_args = " ".join(affinity_args)

        cmd = ["scripts/run.py", "-n", "--with-signals"]
        cmd += [ "-c", "{}".format(len(cores)) ]
        cmd += [ "--qemu-affinity-args", " -v -k {}".format(affinity_args) ]
        cmd += [ "-e", osv_cmdline ]
        if osv_image:
            cmd += [ "-i", osv_image ]

        srv = BMRServer()

        print(cmd)
        qemu = subprocess.Popen(cmd)

        try:
            conn = srv.wait_callback()
            with profiler:
                conn.start_and_wait()
        except Exception as e:
            print("error during benchmark: {}".format(e))
            import traceback
            traceback.print_exc()

        print("waiting for qmeu to stop")
        #print("killing scripts/run.py")
        ## it's scripts/run.py and it's meant to be used interactively
        #qemu.send_signal(signal.SIGINT)
        qemu.wait()

class CommandRun:
    def __init__(self, profiler, cores, command):

        taskset = ["taskset", "-c", ",".join([ "{}".format(c) for c in cores ])]

        srv = BMRServer()

        print(taskset + command)
        bench = subprocess.Popen(taskset + command)

        conn = srv.wait_callback()
        with profiler:
            conn.start_and_wait()

        bench.wait()

