#include <vector>
#include <array>
#include <algorithm>

#include <unistd.h>
#include <sys/mman.h>
#include <assert.h>
#include <string.h>

namespace cs_microbench {

/**
 * d-cache-latency bound work
 *
 * TODO: verify offsets.size() == 16 is actualy enough
 **/
class dcache_work {
    public:
    dcache_work(int size) : d(size) {
        for (int c = 0; c < offsets.size(); c++) {
            // span PAGE_SIZE / (64 * offsets.size()) pages
            offsets[c] = 64 * (c+1);
        }
        std::random_shuffle(offsets.begin(), offsets.end());
    }

    void work(int iterations) {
        int idx = 0;
        for (int c = 0; c < iterations; c++) {
            d[idx] = d[idx] + 1;
            idx = (idx + offsets[c%offsets.size()]) % d.size();
        }
    }

    private:
    std::vector<uint8_t> d;
    std::array<int, 16> offsets;
};

/**
 * i-cache latency bound work
 *
 * A chain of jmp instructions, i.e. from one jmp to the next jmp.
 * Each jmp is located on a different cache line.
 * Each cache line is touched exactly once, first and last one in
 * memory are also first and last ones to be touched.
 */
class icache_work {
    uint8_t *code;
    int size;

public:
    icache_work(int asize) : size(asize) {

        int ret;
        if (size % getpagesize() != 0) {
            throw std::runtime_error("size must be multiple of a page size");
        }
        if (size % 64 != 0) {
            throw std::runtime_error("size must be multiple of a cache line size");
        }

        code = (uint8_t*) mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        if (code == MAP_FAILED) {
            perror("cannot mmap");
            throw std::runtime_error("cannot allocate");
        }
        ret = mprotect(code, size, PROT_READ | PROT_WRITE | PROT_EXEC);
        if (ret) {
            perror("cannot mprotect");
            throw std::runtime_error("cannot mprotect");
        }

        build_random_work();

    }

    void work(int iterations) {
        void (*call_code)() = (void (*)()) code;
        for (int c = 0; c < iterations; c++)
            call_code();
    }

private:
    void build_random_work() {

        // array of start of cache lines in open interval (code, code+size)
        std::vector<uint8_t*> jmp_cls(size/64 - 2);
        for (int c = 0; c < jmp_cls.size(); c++) {
            jmp_cls[c] = code + 64*(c+1);
            assert(jmp_cls[c] >= code + 64);
            assert(jmp_cls[c] < code + size - 64);
        }
        std::random_shuffle(jmp_cls.begin(), jmp_cls.end());

        // 0ed memory causes segfaults (?)
        memset(code, 0, size);

        // start a random walk using jmp in [code, code+size-64)
        // i.e. visit each cache line exactly once
        uint8_t *src = code;
        for (uint8_t *next : jmp_cls) {
            // jmp rel32
            *src = 0xe9;
            uint32_t *offset = (uint32_t*)(src+1);
            *offset = next - src - 5;
            src = next;
        }

        // finish with a jump to the last cache line
        // jmp rel32
        *src = 0xe9;
        uint32_t *offset = (uint32_t*)(src+1);
        *offset = (code + size - 64 - 5) - src;

        // return to caller
        // ret
        code[size-64] = 0xc3;

    };

    /* unused, prefetcher hoses the results here... */
    void build_linear_work() {

        // 0ed memory causes segfaults (?)
        memset(code, 0, size);

        for (int pos  = 0; pos < size ; pos += 64) {
            // jmp rel32
            code[pos] = 0xe9;
            uint32_t *offset = ((uint32_t*)&code[pos+1]);
            *offset = 64 - 5; // jump one cache line
         }

        // ret
        code[size-64] = 0xc3;
    };

 
};

}
