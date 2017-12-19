#!/usr/bin/env python3

# This tool executes

#####################################

import subprocess
import socket
import signal
import time
import shlex

class LikwidPerfctr:
    """Abstraction around likwid-perfctr to configure & read PMU events

    Usable as a context manager, starts profiling at __enter__,
    stops profiling at  __exit__.

    Profiling results are available in the .result attribute afterwards.
    """

    def __init__(self, COUNTERS, cores, tmpfile, stethoscope_time="3600s"):
        self.cores = cores # int array
        self.counters = COUNTERS
        self.tmpfile = tmpfile

        groupstring = []
        pmcid = 0
        for counter in COUNTERS:
            # use KERNEL counters to capture OSv events since everything
            # in the OSv VM executes in ring0
            groupstring.append("{}:PMC{}:KERNEL".format(counter, pmcid))
            pmcid += 1
        groupstring=",".join(groupstring)
        self.cmd = ["/usr/local/bin/likwid-perfctr", "-f"]
        self.cmd += ["-o", tmpfile]
        self.cmd += ["-C", ",".join([str(c) for c in cores])]
        self.cmd += ["-g", groupstring]
        self.cmd += ["-S", stethoscope_time]
        self.result = None

    def __enter__(self):
        print(self.cmd)
        self.proc = subprocess.Popen(self.cmd)
        time.sleep(1) # FIXME let's hope perfctr is recording after 1s

    def __exit__(self, exc_type, exc, tb):
        self.proc.send_signal(signal.SIGINT)
        self.proc.wait()

        if exc:
            print("exception during likwid-perfctr: {}".format(exc))
            raise exc

        result = {}
        # parse the crazy CSV format
        # FIXME find alternative output format that is safer
        #       hint: there's an "xml filter", maybe that's interesting
        with open(self.tmpfile) as f:
            found_raw_table = False
            found_csv_header = False
            expected_num_of_comps =  2 + len(self.cores)
            for line in f:
                line = line.strip()
#                print(line)
                if not found_raw_table:
                    if line.startswith("TABLE,Group 1 Raw,Custom,"):
                        found_raw_table = True
                    continue

                if not found_csv_header:
                    if line.startswith("Event,Counter,Core"):
                        # TODO validate cores are ordered as specified via -C
                        found_csv_header = True
                    continue

                if line.startswith("TABLE,"):
                    break # end of results table

#                print(line + "\n")
                comps = line.split(",", expected_num_of_comps)
#                if len(comps) != expected_num_of_comps :
#                    raise Exception("unexpected CSV format")

                counter, pmc = comps[0:2]
                core_vals = comps[2:]
                try:
                    result[counter] = [ int(val) for val in core_vals ]
                except:
                    result[counter] = None


        self.result = result

class ShellRun:
    """Abstraction for profiling a shell command"""

    def __init__(self, profiler, command):
        with profiler:
            print(command)
            subprocess.call(command, shell=True)

class OSVRun:
    """Abstraction for profiling a benchmark running in OSv

    The KVM threads are pinned to specific cores.

    The benchmark running inside OSv controls the profiling time
        through a TCP callback.
    """

    def __init__(self, profiler, core_pairs, osv_cmdline,
                 callback_ip="192.168.122.1", callback_port=1235):

        affinity_args = ["{}:{}".format(v,p) for v, p in core_pairs]
        affinity_args = " ".join(affinity_args)

        cmd = ["scripts/run.py", "-n", "-V", "--with-signals"]
        cmd += [ "-c", "{}".format(len(core_pairs)) ]
        cmd += [ "--qemu-affinity-args", " -v -k {}".format(affinity_args) ]
        cmd += [ "-e", osv_cmdline ]

        print("setup callback socket")
        try:
            ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            ls.settimeout(10) #benchmark must be ready after 10s
            ls.bind((callback_ip, callback_port))
            ls.listen(1)
        except Exception as e:
            print("error setting up callback socket: {}".format(e))
            return

        try:
            print(cmd)
            print(" ".join(cmd))
            qemu = subprocess.Popen(cmd)
        except Exception as e:
            print("could not start OSv VM: {}".format(e))
            ls.close()
            return

        callback_conn = None
        try:
            print("waiting for callback from benchmark")
            callback_conn, client = ls.accept()
            print("callback connection from client {}".format(client))
            print("starting profiler")
            with profiler:
                print("profiler started, signalling benchmark start")
                callback_conn.send("START\n".encode("ascii"))
                msg = callback_conn.recv(1024)
                msg = msg.decode("ascii")
                msg.strip()
                print("received: {}".format(msg))
                if msg == "FAIL":
                    raise Exception("benchmark signalled failure")

        except Exception as e:
            print("error during benchmark: {}".format(e))
        finally:
            if callback_conn:
                callback_conn.close()
            ls.close()
            print("killing scripts/run.py")
            # it's scripts/run.py and it's meant to be used interactively
            qemu.send_signal(signal.SIGINT)
            qemu.wait()

######################################

# TODO make this a reusable script with argparse / click
# TODO right now we need to modify it for benchmark runs

import tempfile

LIKWID_TMPFILE = "/tmp/likwidout.csv"

COUNTERS = [
        "MEM_LOAD_UOPS_RETIRED_L1_MISS",
        "MEM_LOAD_UOPS_RETIRED_L2_MISS",
        "MEM_LOAD_UOPS_RETIRED_L3_MISS",
        "MEM_UOPS_RETIRED_LOADS",
        ]

cores = [6,7,8,9]

csvout = tempfile.NamedTemporaryFile()

likwid = LikwidPerfctrStethoscope(COUNTERS, cores, csvout.name)
print("likwid-perfctr writes to temporary {}, will persist if this process crashes".format(csvout.name))

core_map = []
for c in cores:
    core_map.append((len(core_map), c))
OSVRun(likwid, core_map, "/cs_microbench.so --bench=cachestress --iterations=8 random 512")

# Alternative for running the microbenchmark on the linux host
#ShellRun(likwid, "sudo taskset -c 6 modules/cs_microbench/cs_microbench.wrapper --bench cachestress --iterations=4 --nobmr random 512".format(",".join([str(c) for c in cores])))

print("benchmark finished, results:")
import pprint
pprint.pprint(likwid.result)
