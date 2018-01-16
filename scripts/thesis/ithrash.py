#!/usr/bin/env python3

# About this Meta-Benchmark
#
# Does multiple runs of the ithrash microbenchmark, either on Linux or in OSv
# The resulting CSVs can be compared and values like
#   - total on-core instruction cache misses
#   - cycles per instruction
# can be derived (and compared).
#
# The OSv version has stages enabled, so we expect a clearly visible benefit
# of staging on OSv wrt the above derived values.
#
# TODO: implement derived values directly in this benchmark

from microbenchmark import *

print("Wrapper around cs_microbench --bench=ithrash")

import tempfile
import sys

class ParametrizedRun:
    def __init__(self, param, resultfile=None, result=None):
        self.param = param
        self.resultfile = resultfile
        self.result = result

    def list_from_params(params):
        return list([ ParametrizedRun(p, None) for p in params ])

def usage():
    print("usage: [linux | osv IMAGE] OUTFILE")
    sys.exit(1)

try:
    args = sys.argv
    args.pop(0)
    platform = args.pop(0)
    if platform not in ["linux", "osv"]:
        print("invalid platform")
        usage()
    if platform == "osv":
        osv_image = args.pop(0)
    outfilepath = args.pop(0)
except:
    usage()

runs = ParametrizedRun.list_from_params([8,16,32,64,128,256,512,1024])

events = [ "cycles",
           "instructions",
           "l2_rqsts.code_rd_hit",
           "l2_rqsts.code_rd_miss",
           "icache.hit",
           "icache.misses",
         ]
events = [ "{}:ukh".format(e) for e in events ]

cores = [ 12,13,14,15 ]
outfile = open(outfilepath, "w")
out = csv.writer(outfile)
out.writerow(["size"] + events)

for run in runs:

    tmp = tempfile.NamedTemporaryFile()
    run.resultfile = "/tmp/perf_tmp"
    print(run.resultfile)

    perf = PerfProfiler(events, cores, run.resultfile)

    # keep run time bounded
    ws_size = run.param
    iterations =  int(1000000 / ws_size)

    if platform == "linux":
        CommandRun(perf, cores, ["modules/cs_microbench/cs_microbench.wrapper",
                                 "--bench=ithrash", "{}".format(ws_size), "4", "{}".format(iterations), "0", "0", "0"])
    elif platform == "osv":
        cmdline = "--ip=eth0,192.168.122.10,255.255.255.0 --defaultgw=192.168.122.1 --nameserver=192.168.122.1 -- /cs_microbench.so --bench=ithrash {} 4 {} 0 0 1".format(ws_size, iterations)
        print("executing " + cmdline)
        OSVRun(perf, cores, cmdline, osv_image)
    else:
        raise Exception()

    run.result = perf.result

    # produce csv output
    print(run.result)
    import numpy as np
    # aggregate core results
    agg = np.zeros(len(events))
    for coreres in run.result:
        agg += np.array(coreres)
    # only process first core
    row = [ws_size] + agg.tolist()
    out.writerow(row)
    outfile.flush()
    print(row)

    tmp.close()

print("benchmark results in " + outfilepath)
