#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <iostream>
#include <exception>
#include <memory>

#include "cs_microbench.hh"
#include "build.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

using namespace cs_microbench;
using namespace std;

namespace bmr {

    namespace msg {
        const char* start = "START\n";
        const char* fail = "FAIL\n";
        const char* end = "END\n";
    }

    typedef std::function<int()> benchmark_function;

    // FIXME refactor socket client code into C++ class
    class client {

        private:
            struct sockaddr_in _host;

        public:

            client(string host, uint16_t port) {
                _host = {
                    .sin_family = AF_INET,
                    .sin_port = htons(port),
                };
                if (!inet_aton(host.c_str(), &_host.sin_addr)) {
                    perror("error setting sin_addr");
                    throw std::runtime_error("error setting sin_addr");
                }
            }

            int run(benchmark_function f) {

                int sock = socket(AF_INET, SOCK_STREAM, 0);
                if (sock == -1) {
                    perror("error socket\n");
                    return -1;
                }
                cout << "connecting to bmrserver\n";
                if (connect(sock, (const sockaddr*)&_host, sizeof(_host))) {
                    perror("error connecting to benchmark runner");
                    return -1;
                }
                static char buf[20] = {0};
                if (read(sock, buf, 19) == -1) {
                    perror("error reading start message from benchmark runner");
                    return -1;
                }
                if (strcmp(buf, msg::start)) {
                    // expect read() received one line...
                    printf("received unexpected message from benchmark runner: %s", buf);
                    return -1;
                }

                auto benchresult = f();

                auto resmsg = benchresult ? msg::fail : msg::end;
                if (write(sock, resmsg, strlen(resmsg)) != strlen(resmsg)) {
                    perror("error sending result message to benchmark runner");
                    return -1;
                }
                close(sock);
                return 0;
            };

    };


}

struct options {
    bool show_help;
    string microbenchmark;
    bool nobenchmarkrunner;
    unsigned int bench_iterations;
};


int main(int argc, char *argv[]) {

    namespace po = boost::program_options;

    options o;
    po::options_description cmdline_options("Allowed options");
    cmdline_options.add_options()
        ("help", po::bool_switch(&o.show_help), "produce help message")
        ("nobmr", po::bool_switch(&o.nobenchmarkrunner), "do not connect to benchmark runner")
        ("iterations", po::value<unsigned int>(&o.bench_iterations)->default_value(1), "how many times the benchmark shall be executed")
        ("bench", po::value<string>(&o.microbenchmark), "which micro benchmark to execute");

    po::variables_map vm;
    auto parsed = po::parse_command_line(argc, argv, cmdline_options);
    po::store(parsed, vm);
    po::notify(vm);

    if (o.show_help) {
        cout << cmdline_options << endl;
        return 0;
    }

    auto microbench_options = po::collect_unrecognized(parsed.options, po::collect_unrecognized_mode::include_positional);

    unique_ptr<Microbenchmark> bench;
    if (o.microbenchmark == "hopper")
        bench = unique_ptr<Hopper>(new Hopper());
    else if (o.microbenchmark == "cachestress")
        bench = unique_ptr<Cachestress>(new Cachestress());
    else if (o.microbenchmark == "ithrash")
        bench = unique_ptr<IThrash>(new IThrash());
    else if (o.microbenchmark == "icachelatency")
        bench = unique_ptr<ICacheLatency>(new ICacheLatency());
    else if (o.microbenchmark == "stagel1i")
        bench = unique_ptr<StageL1I>(new StageL1I());
    else
        throw runtime_error("unknown microbenchmark " + o.microbenchmark);

    cout << SCM_VERSION << endl;

    int res;
    bmr::benchmark_function runbenchmark = [&] {
        for (unsigned int c = 0; c < o.bench_iterations; c++) {
            bench->run(microbench_options);
        }
        return 0;
    };
    if (o.nobenchmarkrunner) {
        res = runbenchmark();
    } else {
        bmr::client client("192.168.122.1", 1235);
        res = client.run(runbenchmark);
        if (res == -1)
            throw runtime_error("benchmarkrunner client error occured");
    }
    return res;
}
