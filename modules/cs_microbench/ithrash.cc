/** About this benchmark
 *
 * Experiment with icache thrashing and show the usefulness of stages.
 *
 * Allocate WS_COUNT working sets of icache_work of WS_SIZE.
 * Cycle between them for ITERATIONS and do one unit of work each.
 *
 * Can use perf or likwid to view icache misses or use the HIST_(START|END)
 * arguments to get a histogram of the duration it took to work one iteration
 * of the current WS> Note that this value does not include the migration cost.
 *
 * When profiling using perf counters, ITERATIONS needs to be pretty big (run
 * several seconds) to marginalize all the icache misses introduced by
 * generating the icache_work + possibly PF handling.
 *
 * Configuration fitting L1 cache on Intel(R) Xeon(R) CPU E5-2618L v3 @ 2.30GHz:

 *   perf stat -e icache.hit,icache.misses ./cs_microbench.wrapper --nobmr --bench=ithrash 8*1024 2 10000000 0 0 0
 *
 *        1,238,751,566      icache.hit:u
 *              130,240      icache.misses:u
 *
 *        1.221487074 seconds time elapsed
 *
 * Configuration thrashing L1 cache on Intel(R) Xeon(R) CPU E5-2618L v3 @ 2.30GHz
 * (notice ratio and execution time):

 *   perf stat -e icache.hit,icache.misses ./cs_microbench.wrapper --nobmr --bench=ithrash 32*1024 2 10000000 0 0 0

 *      37,757,666,715      icache.hit:u
 *       2,396,263,636      icache.misses:u

 *        33.856042099 seconds time elapsed
 *
 */
#include "cs_microbench.hh"
#include "work.hh"
#include "histogram.hh"

#include <osv/stagesched.h>

#include <stdexcept>
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>
#include <chrono>

#include <assert.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>

using namespace cs_microbench;

struct work_item {
    sched::stage *stage;
    icache_work *work;
};

int IThrash::run(bench_args_type args) {
    using namespace std;

    if (args.size() != 6) {
        throw runtime_error("usage: WS_SIZE[KiB] WS_COUNT ITERATIONS HIST_START HIST_END USE_STAGES");
    }

    int size = std::stoi(args[0])*1024;
    int work_count = std::stoi(args[1]);
    int iterations = std::stoi(args[2]);
    int hist_start = std::stoi(args[3]);
    int hist_end = std::stoi(args[4]); // == 0 disables histogram
    int use_stages = std::stoi(args[5]);
    std::vector<work_item> work;

    for (int s = 0; s < work_count; s++) {
        work_item wi;
        if (use_stages) {
            wi.stage = sched::stage::define("stage" + s);
            if (wi.stage == nullptr) {
                throw std::runtime_error("cannot define stage");
            }
        }
        wi.work = new icache_work(size);
        work.push_back(wi);
    }

    register bool use_hist = hist_end > 0;
    util::hist_linear hist(20, hist_start, use_hist ? hist_end : 1);

    chrono::time_point<chrono::high_resolution_clock> before, after;
    chrono::duration<double, std::nano> delta;

    for (int i = 0; i < iterations; i++) {
        auto w = work[i%work.size()];
        if (use_stages) {
            w.stage->enqueue();
        }
        if (use_hist) before = chrono::high_resolution_clock::now();
        w.work->work(1);
        if (use_hist) {
            after = chrono::high_resolution_clock::now();
            delta  = after - before;
            hist(((double)delta.count()));
        }
    }

    if (use_hist)
        hist.dump();

}

