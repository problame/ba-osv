#include <algorithm>
#include <iostream>
#include <sys/mman.h>
#include <cassert>

#include "cs_microbench.hh"

using namespace cs_microbench;

constexpr int arena_size =  256*1024*1024;  //256MB
static uint8_t arena[arena_size];

static void seqwalk(int lines){

    if (64*lines > arena_size)
        throw std::runtime_error("arena size exceeded");

    register int last = 0;
    for (int times = 0; times < 1000000*16; times++) {
            arena[last] = arena[last] + 1;
            last = (last + 1) % (64*lines);
    }

}

void randomwalk(int lines) {

    if (64*lines > arena_size)
        throw std::runtime_error("arena size exceeded");

    std::array<int,16> offsets;
    for (int c = 1; c <= offsets.size(); c++) {
        offsets[c-1] = 64*c;
    }

    std::random_shuffle(offsets.begin(), offsets.end());

    register int last = 0;
    for (auto times = 0ULL; times < 10000000ULL; times++) {
        for (auto o : offsets) {
            arena[last] = arena[last] + 1;
            last = (last + o) % (64*lines);
            assert(last < arena_size);
        }
    }

    // introduce data dependency to prohibit optimizations from removing above loop
    // TODO is this necessary? we need to compile at O0 anyways
    asm volatile(
            ""
          ::""(arena[0]):"memory"
            );

}

int Cachestress::run(bench_args_type args) {

    if (args[0] == "seq") {
        seqwalk(std::stoi(args[1]));
    }
    else if (args[0] == "random") {
        randomwalk(std::stoi(args[1]));
    } else if (args[0] == "loop") {
        while(true);
    } else {
        throw "invalid walk\n";
    }

    return 0;

}
