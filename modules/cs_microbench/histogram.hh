#include <cstdio>
#include <cassert>

namespace cs_microbench {
    namespace util {

        class hist_linear {
            public:
                hist_linear (int anum_buckets, double alower, double aupper) :
                    lower(alower), upper(aupper), num_buckets(anum_buckets), buckets(anum_buckets, 0)
            {
                if (lower >= upper)
                    throw std::runtime_error("invalid range");

                range = upper - lower;
            }

                bool operator()(double val) {
                    int idx = bucket_idx_for_val(val);
                    buckets[idx]++;
                }
                void dump() {
                    using namespace std;
                    for (int b = 0; b < num_buckets; b++) {
                        auto begin = lower + b*(range/num_buckets);
                        auto end = begin + range/num_buckets;
                        printf("%07.2f\t%07.2f\t%d\n", begin, end, buckets[b]);
                    }
                }
            private:
                int bucket_idx_for_val(double val) {
                    // fit into bucket
                    double bucket = val - lower;
                    if (bucket < 0)
                        return 0;

                    bucket /= range;

                    if (bucket >= 1.0)
                        return num_buckets-1;

                    assert(bucket >= 0.0 && bucket < 1.0);
                    return (int)(bucket * num_buckets);
                }
            private:
                double lower;
                double upper;
                double range;
                int num_buckets;
                std::vector<int> buckets;
        };

    }
}

