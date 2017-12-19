#include <iostream>
#include <thread>
#include <cstdio>
#include <list>
#include <vector>
#include <string>
#include <osv/stagesched.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "cs_microbench.hh"
#include "histogram.hh"

using namespace cs_microbench;
using namespace std;

static constexpr int stages_len = 8;
static constexpr int threads_len = 200;
static std::vector<sched::stage*> stages;

static int work(int stage, int time) {
    printf("doing work %d@stage%d\n", time, stage);

    stages[stage]->enqueue();
    for (int c = 0; c < time*1000000000; c++) {
        c = c;
        if (c % 100000000 == 0)
            sched_yield();
    }
}

int Hopper::run(bench_args_type args) {

    if (args.size() != 4) {
        throw std::runtime_error("usage: BUCKETS MIN[ns] MAX[ns] ITERATIONS");
    }

    int hist_num_buckets, iterations;
    double hist_min, hist_max;

    hist_num_buckets = stoi(args[0]);
    hist_min = stod(args[1]);
    hist_max = stod(args[2]);
    iterations = stoi(args[3]);

    for (int c = 0; c < stages_len; c++){
        auto s = sched::stage::define("stage" + c);
        if (!s) {
            printf("could not create stage %d\n", c);
            exit(1);
        }
        stages.push_back(s);
    }

    namespace a = boost::accumulators;
    using namespace a;
    accumulator_set<double, features<tag::min, tag::max, tag::count, tag::mean>> acc;
    util::hist_linear hist(hist_num_buckets, hist_min, hist_max);

    std::chrono::duration<double, std::nano> delta;
    for(int c = 0; c < iterations; c++) {
        auto s = stages[c%stages_len];
        auto before = std::chrono::steady_clock::now();
        s->enqueue();
        auto after = std::chrono::steady_clock::now();
        delta = after - before;
        if (c > 1000) {
            acc(delta.count());
            hist(delta.count());
        }
    }

    using namespace std;
    cout << "count\t" << a::count(acc) << endl;
    cout << "mean\t" << a::mean(acc) << endl;
    cout << "min\t" << a::min(acc) << endl;
    cout << "max\t" << a::max(acc) << endl;

    hist.dump();

    return 0;
}
