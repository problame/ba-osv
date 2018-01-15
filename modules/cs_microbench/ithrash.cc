/** About this benchmark
 *
 * Experiment with icache thrashing:
 *
 * Allocate WS_COUNT working sets of icache_work of WS_SIZE.
 * Cycle between them for ITERATIONS.
 *
 * Can use perf or likwid to view icache misses.
 *
 * ITERATIONS needs to be pretty big (run several seconds) to marginalize
 * all the icache misses introduced by generating the icache_work + possibly PF handling.
 *
 * Configuration fitting L1 cache:

 *   perf stat -e icache.hit,icache.misses ./cs_microbench.wrapper --nobmr --bench=ithrash $((8*1024)) 2 10000000 0
 *
 *        1,238,751,566      icache.hit:u
 *              130,240      icache.misses:u
 *
 *        1.221487074 seconds time elapsed
 *
 * Configuration thrashing L1 cache (notice ratio and execution time)

 *   perf stat -e icache.hit,icache.misses ./cs_microbench.wrapper --nobmr --bench=ithrash $((32*1024)) 2 10000000 0

 *      37,757,666,715      icache.hit:u
 *       2,396,263,636      icache.misses:u

 *        33.856042099 seconds time elapsed
 *
 */
#include "cs_microbench.hh"
#include "work.hh"

#include <osv/stagesched.h>

#include <stdexcept>
#include <algorithm>
#include <array>
#include <vector>
#include <stdexcept>

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

    if (args.size() != 4) {
        throw runtime_error("usage: WS_SIZE WS_COUNT ITERATIONS USE_STAGES");
    }

    int size = std::stoi(args[0]);
    int work_count = std::stoi(args[1]);
    int iterations = std::stoi(args[2]);
    int use_stages = std::stoi(args[3]);
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

    for (int i = 0; i < iterations; i++) {
        auto w = work[i%work.size()];
        if (use_stages) {
            w.stage->enqueue();
        }
        w.work->work(1);
    }

}

