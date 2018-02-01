#!/usr/bin/env python3

# How to use this piece of art^Wduct tape
#
# 1. Create an OSv image with the MySQL OSv port (modules/mysql_stagesched for stageschedified OSv),
#  	 launch it and preseed the database for tpcc with a configuration like the one im tmplstr below.
#    Cleanly shut down the VM.
# 2. Use this script to run tpcc against a fresh copy of the preseeded OSv image:
#    NOTE: needs to run as root
#
# Example command for stagesched OSv:
#
#   sudo scripts/thesis/oltp.py run --oltpdir ../benchmarks/oltpbench/ --terminals 2:40:10 \
# 		--max-assignment-age 5000 --duration 30 \
#		--outdir /tmp/out_pageaccess3
#
# Example command for upstream OSv: (note the isupstream flag)
#
#	sudo scripts/thesis/oltp.py run --oltpdir ../benchmarks/oltpbench/ --terminals 2:40:10 \
#		--max-assignment-age 5000 --duration 30 \
#		--image ~/osv_upstream/build/last/usr.img --isupstream=true \
#  		--outdir /tmp/out_upstream
#
#

import subprocess
import string
import pathlib
import os
import MySQLdb
import signal
import time
import click
import csv
import pickle
import sys
import shutil

def popen_tpcc(oltpbenchmark_dir, outdir, terminals, time_sec):

    tmplstr = """<?xml version="1.0"?>
<parameters>
	
    <!-- Connection details -->
    <dbtype>mysql</dbtype>
    <driver>com.mysql.jdbc.Driver</driver>
    <DBUrl>jdbc:mysql://192.168.122.10:3306/tpcc</DBUrl>
    <username>root</username>
    <password></password>
    <isolation>TRANSACTION_SERIALIZABLE</isolation>
    
    <!-- Scale factor is the number of warehouses in TPCC -->
    <scalefactor>2</scalefactor>
    
    <!-- The workload -->
    <terminals>${CS_THESIS_TERMINALS}</terminals>
    <works>
        <work>
          <time>${CS_THESIS_TIME}</time>
          <rate>10000</rate>
          <weights>45,43,4,4,4</weights>
        </work>
	</works>
	
	<!-- TPCC specific -->  	
   	<transactiontypes>
    	<transactiontype>
    		<name>NewOrder</name>
    	</transactiontype>
    	<transactiontype>
    		<name>Payment</name>
    	</transactiontype>
    	<transactiontype>
    		<name>OrderStatus</name>
    	</transactiontype>
    	<transactiontype>
    		<name>Delivery</name>
    	</transactiontype>
    	<transactiontype>
    		<name>StockLevel</name>
    	</transactiontype>
   	</transactiontypes>	
</parameters>
"""
    tmpl = string.Template(tmplstr)
    configstr = tmpl.substitute({"CS_THESIS_TERMINALS":terminals, "CS_THESIS_TIME":time_sec})

    oltpconfig = pathlib.Path(outdir) / 'tpcc.oltp.config.xml'
    with open(oltpconfig, 'w') as o:
        o.write(configstr)

    cmd = [ "perf", "stat", "-e", "l2_rqsts.code_rd_hit:ukhG,l2_rqsts.code_rd_miss:ukhG,instructions:ukhG,cycles:ukhG,icache.hit:ukhG,icache.misses:ukhG" ]
    cmd += [ "-C", "10-15" ]
    cmd += [ "taskset", "-c", "8,9" ]
    cmd += [ './oltpbenchmark' ,  "--config", oltpconfig, "--bench=tpcc", "--execute=true", "-s", "2" ]
    cmd += [ "--directory", outdir, "-o", "result" ]

    stdredirpath = pathlib.Path(outdir) / "perf_oltp.stdx.txt"
    stdredir = open(stdredirpath, "w")
    return subprocess.Popen(cmd, stdout=stdredir, stderr=stdredir, cwd=oltpbenchmark_dir)


def popen_osv(stdredir, image_path, isupstream, max_assignment_age):
    cmd = [ "scripts/run.py", "-n"]
    cmd += ["-c", "6", "-V", "--qemu-affinity-args", " -v -k 0:12 1:13 2:14 3:15 4:11 5:10" ]
    osv_cmdline = "--ip=eth0,192.168.122.10,255.255.255.0 "
    osv_cmdline += "--defaultgw=192.168.122.1 "
    osv_cmdline += "--nameserver=192.168.122.1 "
    if not isupstream:
        osv_cmdline += "--stage.max_assignment_age={} ".format(max_assignment_age)
    osv_cmdline += "-- "
    osv_cmdline += "/usr/bin/mysqld --basedir /usr --datadir data --user root"
    cmd += [ "-e", osv_cmdline ]
    cmd += ["-i", image_path ]
    stdredirfile = open(stdredir, "w")
    return subprocess.Popen(cmd, stdout=stdredirfile, stderr=stdredirfile)

def check_mysql_ready():
    try:
        cnx = MySQLdb.connect(host="192.168.122.10", user='root', database='tpcc', connect_timeout=1)
        cursor = cnx.cursor()
        query = ("SELECT * FROM CUSTOMER LIMIT 1")
        cursor.execute(query) # todo check expected response
        cursor.close()
        cnx.close()
        return True
    except Exception as e:
        print("check_mysql_ready: error: " + str(e))
        return False

class Params:
    def __init__(self, max_assignment_age, terminals, time_sec, varying):
        self.max_assignment_age = max_assignment_age
        self.terminals = terminals
        self.time_sec = time_sec
        self.varying = varying # name of property that varies

    def dirname(self):
        return "{}_{}_{}".format(self.max_assignment_age, self.terminals, self.time_sec)

def run_benchmark(all_outdir, oltpbenchmark_dir, image_path, isupstream, params):
    outdir = pathlib.Path(all_outdir) / params.dirname()
    print("run benchmark dir " + str(outdir))
    os.mkdir(outdir)

    with open(outdir/"params.marshal", "wb") as p:
        pickle.dump(params, p)

    ramimage = "/tmp/cs_thesis_img_cur.img"
    print("copy {} to {}".format(image_path, ramimage))
    shutil.copyfile(image_path, ramimage)

    osv = popen_osv(outdir/"osv_stdx.txt", ramimage, isupstream, params.max_assignment_age)
    retries = 20
    for i in range(0, retries):
        if check_mysql_ready():
            break
        osvret = osv.poll()
        if osvret:
            raise Exception("osv exited before benchmark, return code " + str(osvret))
        if i == retries-1:
            osv.kill()
            raise Exception("MySQL not ready after {} retries".format(retries))

    print("sleep 2 secs to let things stabilize")
    time.sleep(2)
    print("start benchmark")
    tpcc_timeout = None

    try:
        tpcc = popen_tpcc(oltpbenchmark_dir, outdir, params.terminals, params.time_sec)
        started_tpcc = True
    except Exception as e:
        print("could not start tpcc: " + str(e))
        tpcc_timeout = e

    if not tpcc_timeout:
        try:
            tpcc_ret = tpcc.wait(params.time_sec + 10)
            if tpcc_ret != 0:
                raise Exception("tpcc exited with " + str(tpcc_ret))
            tpcc_timeout = None
        except Exception as e:
            tpcc.kill()
            tpcc_timeout = e

    os.kill(osv.pid, signal.SIGINT)
    try:
        osv.wait(5)
    except:
        osv.kill()

    if tpcc_timeout:
        raise Exception("tpcc benchmark error: " + str(tpcc_timeout))

@click.group()
def cli():
    pass

@cli.command()
@click.option('--outdir', required=True, help='path to (nonexistent) results directory')
@click.option('--image', default="build/last/usr.img", help='path to OSv image (will be copied to tmpfs before use')
@click.option('--oltpdir', required=True, help='path to oltpbenchmark git checkout (directory must contain compiled oltpbenchmark "binary")')
@click.option('--duration', default=20, help='duration argument of OLTP / TPCC')
@click.option('--terminals', required=True, type=str)
@click.option('--max-assignment-age', required=True, type=str)
@click.option('--isupstream', type=bool, default=False, help="do not set --stage.max_assignment_age in OSv command line")
def run(outdir, image, oltpdir, duration, terminals, max_assignment_age, isupstream):

    try:
        os.mkdir(outdir)
    except FileExistsError as e:
        # try delete it if empty and re-create it
        os.rmdir(outdir)
        os.mkdir(outdir)

    def parse_steps(stepspec):
        try:
            if int(stepspec):
                return None
        except:
            pass
        comps = stepspec.split(":")
        if len(comps) != 3:
            raise Exception("invalid step spec")
        return [int(i) for i in comps]

    terminal_steps = parse_steps(terminals)
    max_assignment_age_steps = parse_steps(max_assignment_age)
    if terminal_steps and  max_assignment_age_steps:
        raise Exception("only support one varying argument")

    if terminal_steps:
        f,t,s = terminal_steps
        for v in [ f + int(((t-f)/s)*j) for j in range(0, s)]:
            p = Params(int(max_assignment_age), v, duration, "terminals")
            run_benchmark(outdir, oltpdir, image, isupstream, p)

    if max_assignment_age_steps:
        f,t,s = max_assignment_age_steps
        for v in [ f + int(((t-f)/s)*j) for j in range(0, s)]:
            p = Params(v, int(terminals), duration, "max_assignment_age")
            run_benchmark(outdir, oltpdir, image, isupstream, p)


@cli.command()
@click.option('--outdir', required=True, help='path to (nonexistent) results directory')
def throughput(outdir):
    times = None
    results = []
    for dirent in os.listdir(outdir):
        dirent = pathlib.Path(outdir) / dirent
        with open(dirent/"params.marshal", "rb") as p:
            params = pickle.load(p)
        with open(dirent/"result.res", "r") as c:
            throughput = []
            r = csv.reader(c)
            if not times:
                times = []
                build_times = True
            else:
                build_times = False
            for row in r:
                throughput.append(row[1])
                if build_times:
                    times.append(row[0])
            if not (len(times) == len(throughput)):
                raise Exception("non-uniform result file detected: " + str(dirent))

        results.append((params, throughput))

    results.sort(key=lambda r: getattr(r[0], r[0].varying))

    w = csv.writer(sys.stdout)
    headers = [ times[0] ]
    headers += [ "{}={}".format(p.varying, getattr(p, p.varying)) for p, t in results ]
    w.writerow(headers)
    for i in range(1, len(times)):
        row = [ times[i] ]
        row += [ t[i] for p, t in results ]
        w.writerow(row)

if __name__ == "__main__":
    cli()

