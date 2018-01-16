#include "cs_microbench.hh"
#include "work.hh"

#include <iostream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>

#include <osv/stagesched.h>

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

static work_item<icache_work*> wi1;
static work_item<icache_work*> wi2;
static work_item<icache_work*> wi3;
static work_item<icache_work*> wi4;

/**
 * Poor man's SPMC queue.
 * FIXME: benchmark might degenerate to benchmarking
 * FIXME  the implementation of this queue
 */
struct connqueue {
    std::condition_variable cond;
    std::mutex mtx;
    std::queue<int> q;
    int _limit;
    connqueue(int limit) : _limit(limit) { }

    void push(int conn) {
        std::unique_lock<std::mutex> ul(mtx);
        cond.wait(ul, [this]{return q.size() < _limit;});
        q.push(conn);
        cond.notify_one();
    }

    int pop() {
        std::unique_lock<std::mutex> ul(mtx);
        cond.wait(ul, [this]{return !q.empty();});
        int conn = q.front();
        q.pop();
        return conn;
    }

};

static void conn_handler(int conn, int iterations) {
    for (int c = 0; c < iterations; c++) {
        // FIXME: don't use wi1 for measurements since in current (no-)scheduling policy, it is on 40% busy core0
        wi2.run(1);
        wi3.run(1);
        wi4.run(1);
    }
    wi1.enqueue();
    close(conn);
}

int Server::run(bench_args_type args) {

    using namespace std;

    if (args.size() != 6) {
        throw runtime_error("usage: read the source :(");
    }

    string listen_addr = args[0], listen_port = args[1];
    int work_kbytes = stoi(args[2]);
    int use_stages = stoi(args[3]); // 0 or 1
    int iterations = stoi(args[4]);
    int thread_count = stoi(args[5]); // 0  for thread-per-conn
                                      // >0 for home-grown thread pool

    if (use_stages) {
        wi1.stage = sched::stage::define("stage1");
        wi2.stage = sched::stage::define("stage2");
        wi3.stage = sched::stage::define("stage3");
        wi4.stage = sched::stage::define("stage4");
    }
    wi1.work = new icache_work(work_kbytes*1024);
    wi2.work = new icache_work(work_kbytes*1024);
    wi3.work = new icache_work(work_kbytes*1024);
    wi4.work = new icache_work(work_kbytes*1024);

    int ret;

    // FIXME refactor socket server code into C++ class
    struct addrinfo hints;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;    /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_STREAM; /* Datagram socket */
    hints.ai_flags = AI_PASSIVE;    /* For wildcard IP address */
    hints.ai_protocol = 0;          /* Any protocol */
    hints.ai_canonname = NULL;
    hints.ai_addr = NULL;
    hints.ai_next = NULL;

    struct addrinfo *result;
    ret = getaddrinfo(listen_addr.c_str(), listen_port.c_str(), &hints, &result);
    if (ret != 0) {
        return 1;
    }

    int lsock = -1;
    for (; result != NULL; result = result->ai_next) {
        int s = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (s == -1)
            continue;

        int enable = 1;
        ret = setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
        if (ret)
            continue;

        ret = bind(s, result->ai_addr, result->ai_addrlen);
        if (ret != 0) {
            close(s);
            continue;
        }

        if (listen(s, 10)) {
            close(s);
            continue;
        }

        lsock = s;
        break;
    }

    if (lsock == -1)
        return 2;

    cout << "listening\n";

    if (thread_count == 0) {
        cout << "spawn a new thread per connection\n";
        while(true) {
            int conn = accept(lsock, NULL, NULL);
            if (conn == -1) {
                perror("error accepting");
                break;
            }
            thread handler_thread(conn_handler, conn, iterations);
            handler_thread.detach();
        }
    } else {
        cout << "use thread pool\n";
        // build thread pool
        connqueue cq(thread_count);
        for (int t = 0; t < thread_count; t++) {
            thread worker([&]{
                while(true) {
                    int conn = cq.pop();
                    conn_handler(conn, iterations);
                }
            });
            worker.detach();
        }
        // run loop
        while (true) {
             int conn = accept(lsock, NULL, NULL);
             if (conn == -1) {
                 perror("error accepting");
                 break;
             }
             cq.push(conn);
        }
    }

    cout << "finished\n";

}
