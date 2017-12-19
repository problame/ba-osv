#include <vector>
#include <array>

namespace cs_microbench {

    typedef std::vector<std::string> bench_args_type;

    class Microbenchmark {
        public:
            virtual int run(bench_args_type args) = 0;
    };

    class Hopper : public Microbenchmark {
        public:
            int run(bench_args_type args);
    };

    class Cachestress: public Microbenchmark {
        public:
            int run(bench_args_type args);
    };


}
