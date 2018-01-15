/** About this Benchmark
 *
 * measures and compares timing difference between
 * - a cold Lx cache that needs to fetch from a warm L(x-1),
 *   e.g. cold L1 fetches from warm L2
 * - a warm Lx cache
 *
 */
#include "cs_microbench.hh"
#include "work.hh"

#include <chrono>
#include <iostream>

using namespace cs_microbench;


struct cache {
    std::string name;
    int size;
};

void usage() {
        throw std::runtime_error("usage: l1|l2");
}

int ICacheLatency::run(bench_args_type args) {
    using namespace std;

    if (args.size() != 1) {
        usage();
    }

    cache c;
    if (args[0] == "l1") {
        c = {"L1", 32*1024};
    } else if (args[0] == "l2") {
        c = {"L2", 256*1024};
    } else {
        usage();
    }

    icache_work w_prime(c.size);
    icache_work w(c.size);

    // cache-line sized, but still before priming
    static_assert(sizeof(double) == 8);
    std::vector<double> res(8, -1.0);

    // bring both into cache below c
    w.work(100);
    w_prime.work(100);

    for (int c = 0; c < res.size(); c++){
        auto start = std::chrono::system_clock::now();
        w.work(1);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::nano> d = end-start;
        res[c] = (double)d.count();
    }

    auto first = res[0];
    auto second = res[1];
    double avg_next = 0.0;
    auto it = next(res.begin());
    for (int c = 1; c < res.size(); c++) {
        avg_next += res[c];
    }
    avg_next /= res.size()-1;
    cout << "1st access to working set          = " << first << "ns" << endl;
    cout << "2nd access to working set          = " << second << "ns" << endl;
    cout << "avg access to working set in cache = " << avg_next << "ns" << endl;
    cout<< "(" << c.name << "-to-lower-cache relative latency (how much faster is " << "c.name" << ") (1st/2nd) = " << first / second << endl;
    cout<< "(" << c.name << "-to-lower-cache relative latency (how much faster is " << "c.name" << ") (1st/avg) = " << first / avg_next << endl;

    return 0;
}
