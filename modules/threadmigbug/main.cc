#include <thread>
#include <sched.h>
#include <error.h>
#include <osv/sched.hh>
#include <iostream>

int main(int argc, char *argv[]) {

    int ncpus = atoi(argv[1]);
    int nthreads = atoi(argv[2]);

    std::map<std::string, std::function<void(int)>> apis;
    apis["osv"] = [](int cpu){
        sched::thread::pin(sched::cpus[cpu]);
    };
    apis["linux"] = [](int cpu){
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) == -1)
            perror("sched_setaffinity failed");
    };

    auto hopper = apis[argv[3]];

    std::vector<std::thread> ts(nthreads);
    for (int t = 0; t < nthreads; t++) {
        ts[t] = std::thread([&]{
            for (int c = 0; ; c = (c + 1) % ncpus) {
                hopper(c);
            }
         });
    }

    std::cout << "waiting..." << std::endl;

    for (std::thread& t : ts) {
        t.join();
    }

}
