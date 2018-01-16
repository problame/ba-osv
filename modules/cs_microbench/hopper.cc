/** About this Benchmark
 *
 * measures the time a stage switch takes
 * and puts the resulting values into a histogram
 * whose parameters are configurable via bench args
 *
 * TODO: investigate footprint of boost accumulators
 **/
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

#include "cs_microbench.hh"
#include "histogram.hh"

using namespace cs_microbench;
using namespace std;

static std::vector<sched::stage*> stages;

int Hopper::run(bench_args_type args) {

    if (args.size() != 5) {
        throw std::runtime_error("usage: NUM_STAGES BUCKETS MIN[ns] MAX[ns] ITERATIONS");
    }

    int hist_num_buckets, iterations, num_stages;
    double hist_min, hist_max;

    num_stages = stoi(args[0]);
    hist_num_buckets = stoi(args[1]);
    hist_min = stod(args[2]);
    hist_max = stod(args[3]);
    iterations = stoi(args[4]);

    printf("defining stages\n");

    for (int c = 0; c < num_stages; c++){
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

    printf("performing stage switches\n");
    std::chrono::duration<double, std::nano> delta;
    for(int c = 0; c < iterations; c++) {
        auto s = stages[c%num_stages];
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
