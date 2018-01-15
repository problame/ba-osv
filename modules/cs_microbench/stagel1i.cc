/** About this Benchmark
 *
 * 4 i-working sets of 32 KiB, each associated with its own stage
 *
 * A loop iterates over the working sets:
 *  (1) if use_stage==1, migrate to the working set's stage
 *  (2) touch every cache line in the working set exactly once
 *
 * => Pick working set size is that of the L1 i-cache (32 KiB)
 * => we switch working set every iteration of the loop,
 *    provoking capacity misses
 * => i-cache misses are visible in the time (2) takes
 * => measure time of (2) and produce a histogram
 *
 * Expected Result:
 *   use_stage == 0
 *     Idealized: full replacement of L1 i-cache every iteration of the loop
 *     => will be fed from L2
 *   use_stage == 1
 *     Idealized: working set fits in L1 i-cache because
 *     working sets are fully partitioned over cores
 *      => no L1 cache misses
 *
 * => according to Timing benchmark, L1 accesses are ~3 x faster then L2 accesses
 * => => use_stage == 1 should have ~3 x faster run time for (2)
 *
 **/

#include "cs_microbench.hh"
#include "work.hh"
#include "histogram.hh"

#include <osv/stagesched.h>

#include <stdexcept>
#include <array>
#include <chrono>
#include <iostream>

using namespace cs_microbench;

template <class W>
struct work_item {
    W work;
    sched::stage *stage;

    void run(int iterations) {
        if (stage != nullptr) {
            stage->enqueue();
        }
        work->work(iterations);
    }
    void enqueue() {
        if (stage != nullptr) {
            stage->enqueue();
        }
    }
};

int StageL1I::run(bench_args_type args) {
    using namespace std;

    if (args.size() != 1) {
        throw runtime_error("usage: use_stage[0|1]");
    }

    int work_size = 32*1024;
    int use_stages = stoi(args[0]);

    // FIXME: hardcoded num stages
    std::array<work_item<icache_work*>, 6> work_items;
    for (int c = 0; c < work_items.size(); c++) {
        work_items[c].work = new icache_work(work_size);
        work_items[c].stage = nullptr;
        if (use_stages) {
            work_items[c].stage = sched::stage::define("stage" + c);
            if (!work_items[c].stage) {
                throw runtime_error("could not define stage");
            }
        }
    }

    util::hist_linear hist(20, 200, 5000);

    auto start = std::chrono::steady_clock::now();
    for (int c = 0; c < 10000; c++) {
        // FIXME: don't use the 0th stage since CPU0 is showing 40% idle load
        work_items[1+(c%4)].enqueue(); // do not count the stage switch

        auto before = std::chrono::steady_clock::now();
        work_items[1+(c%4)].work->work(1);
        auto after = std::chrono::steady_clock::now();

        std::chrono::duration<double, std::nano> delta  = after - before;
        hist(((double)delta.count()));
    }
    hist.dump();

}
