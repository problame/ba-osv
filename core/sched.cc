/*
 * Copyright (C) 2013 Cloudius Systems, Ltd.
 *
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */

#include <osv/sched.hh>
#include <list>
#include <osv/mutex.h>
#include <osv/rwlock.h>
#include <mutex>
#include <osv/debug.hh>
#include <osv/irqlock.hh>
#include <osv/align.hh>
#include <osv/interrupt.hh>
#include <smp.hh>
#include "osv/trace.hh"
#include <osv/percpu.hh>
#include <osv/prio.hh>
#include <osv/elf.hh>
#include <stdlib.h>
#include <math.h>
#include <unordered_map>
#include <osv/wait_record.hh>
#include <osv/preempt-lock.hh>
#include <osv/app.hh>
#include <osv/symbols.hh>

MAKE_SYMBOL(sched::thread::current);
MAKE_SYMBOL(sched::cpu::current);
MAKE_SYMBOL(sched::get_preempt_counter);
MAKE_SYMBOL(sched::preemptable);
MAKE_SYMBOL(sched::preempt);
MAKE_SYMBOL(sched::preempt_disable);
MAKE_SYMBOL(sched::preempt_enable);

__thread char* percpu_base;

extern char _percpu_start[], _percpu_end[];

using namespace osv;
using namespace osv::clock::literals;

namespace sched {

TRACEPOINT(trace_sched_idle, "");
TRACEPOINT(trace_sched_idle_ret, "");
TRACEPOINT(trace_sched_switch, "to %p vold=%g vnew=%g", thread*, float, float);
TRACEPOINT(trace_sched_wait, "");
TRACEPOINT(trace_sched_wait_ret, "");
TRACEPOINT(trace_sched_wake, "wake %p", thread*);
TRACEPOINT(trace_sched_migrate, "thread=%p cpu=%d", thread*, unsigned);
TRACEPOINT(trace_sched_queue, "thread=%p", thread*);
TRACEPOINT(trace_sched_load, "load=%d", size_t);
TRACEPOINT(trace_sched_preempt, "");
TRACEPOINT(trace_sched_ipi, "cpu %d", unsigned);
TRACEPOINT(trace_sched_yield, "");
TRACEPOINT(trace_sched_yield_switch, "");
TRACEPOINT(trace_sched_sched, "");
TRACEPOINT(trace_timer_set, "timer=%p time=%d", timer_base*, s64);
TRACEPOINT(trace_timer_reset, "timer=%p time=%d", timer_base*, s64);
TRACEPOINT(trace_timer_cancel, "timer=%p", timer_base*);
TRACEPOINT(trace_timer_fired, "timer=%p", timer_base*);
TRACEPOINT(trace_thread_create, "thread=%p", thread*);
TRACEPOINT(trace_sched_stage_enqueue, "stage=%p scpu=%d tcpu=%d thread=%p", stage*, unsigned, unsigned, thread*);
TRACEPOINT(trace_sched_stage_dequeue, "dcpu=%d thread=%p", unsigned, thread*);
TRACEPOINT(trace_sched_stage_dequeue_stagemig, "dcpu=%d thread=%p", unsigned, thread*);

std::vector<cpu*> cpus __attribute__((init_priority((int)init_prio::cpus)));

thread __thread * s_current;
cpu __thread * current_cpu;

unsigned __thread preempt_counter = 1;
bool __thread need_reschedule = false;

elf::tls_data tls;

inter_processor_interrupt wakeup_ipi{IPI_WAKEUP, [] {}};

constexpr float cmax = 0x1P63;
constexpr float cinitial = 0x1P-63;

static inline float exp_tau(thread_runtime::duration t) {
    // return expf((float)t/(float)tau);
    // Approximate e^x as much faster 1+x for x<0.001 (the error is O(x^2)).
    // Further speed up by comparing and adding integers as much as we can:
    static constexpr int m = tau.count() / 1000;
    static constexpr float invtau = 1.0f / tau.count();
    if (t.count() < m && t.count() > -m)
        return (tau.count() + t.count()) * invtau;
    else
        return expf(t.count() * invtau);
}

// fastlog2() is an approximation of log2, designed for speed over accuracy
// (it is accurate to roughly 5 digits).
// The function is copyright (C) 2012 Paul Mineiro, released under the
// BSD license. See https://code.google.com/p/fastapprox/.
static inline float
fastlog2 (float x)
{
    union { float f; u32 i; } vx = { x };
    union { u32 i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 124.22551499f - 1.498030302f * mx.f
            - 1.72587999f / (0.3520887068f + mx.f);
}

static inline float taulog(float f) {
    //return tau * logf(f);
    // We don't need the full accuracy of logf - we use this in time_until(),
    // where it's fine to overshoot, even significantly, the correct time
    // because a thread running a bit too much will "pay" in runtime.
    // We multiply by 1.01 to ensure overshoot, not undershoot.
    static constexpr float tau2 = tau.count() * 0.69314718f * 1.01;
    return tau2 * fastlog2(f);
}

static constexpr runtime_t inf = std::numeric_limits<runtime_t>::infinity();

mutex cpu::notifier::_mtx;
std::list<cpu::notifier*> cpu::notifier::_notifiers __attribute__((init_priority((int)init_prio::notifiers)));

}

#include "arch-switch.hh"

namespace sched {

class thread::reaper {
public:
    reaper();
    void reap();
    void add_zombie(thread* z);
private:
    mutex _mtx;
    std::list<thread*> _zombies;
    std::unique_ptr<thread> _thread;
};

cpu::cpu(unsigned _id)
    : id(_id)
    , idle_thread()
    , terminating_thread(nullptr)
    , c(cinitial)
    , renormalize_count(0)
{
    auto pcpu_size = _percpu_end - _percpu_start;
    // We want the want the per-cpu area to be aligned as the most strictly
    // aligned per-cpu variable. This is probably CACHELINE_ALIGNED (64 bytes)
    // but we'll be even stricter, and go for page (4096 bytes) alignment.
    percpu_base = (char *) aligned_alloc(4096, pcpu_size);
    memcpy(percpu_base, _percpu_start, pcpu_size);
    percpu_base -= reinterpret_cast<size_t>(_percpu_start);
    if (id == 0) {
        ::percpu_base = percpu_base;
    }
}

void cpu::init_idle_thread()
{
    running_since = osv::clock::uptime::now();
    std::string name = osv::sprintf("idle%d", id);
    idle_thread = thread::make([this] { idle(); }, thread::attr().pin(this).name(name));
    idle_thread->set_priority(thread::priority_idle);
}

// Estimating a *running* thread's total cpu usage (in thread::thread_clock())
// requires knowing a pair [running_since, cpu_time_at_running_since].
// Since we can't read a pair of u64 values atomically, nor want to slow down
// context switches by additional memory fences, our solution is to write
// a single 64 bit "_cputime_estimator" which is atomically written with
// 32 bits from each of the above values. We arrive at 32 bits by dropping
// the cputime_shift=10 lowest bits (so we get microsec accuracy instead of ns)
// and the 22 highest bits (so our range is reduced to about 2000 seconds, but
// since context switches occur much more frequently than that, we're ok).
constexpr unsigned cputime_shift = 10;
void thread::cputime_estimator_set(
        osv::clock::uptime::time_point running_since,
        osv::clock::uptime::duration total_cpu_time)
{
    u32 rs = running_since.time_since_epoch().count() >> cputime_shift;
    u32 tc = total_cpu_time.count() >> cputime_shift;
    _cputime_estimator.store(rs | ((u64)tc << 32), std::memory_order_relaxed);
}
void thread::cputime_estimator_get(
        osv::clock::uptime::time_point &running_since,
        osv::clock::uptime::duration &total_cpu_time)
{
    u64 e = _cputime_estimator.load(std::memory_order_relaxed);
    u64 rs = ((u64)(u32) e) << cputime_shift;
    u64 tc = (e >> 32) << cputime_shift;
    // Recover the (64-32-cputime_shift) high-order bits of rs and tc that we
    // didn't save in _cputime_estimator, by taking the current values of the
    // bits in the current time and _total_cpu_time, respectively.
    // These high bits usually remain the same if little time has passed, but
    // there's also the chance that the old value was close to the cutoff, and
    // just a short passing time caused the high-order part to increase by one
    // since we saved _cputime_estimator. We recognize this case, and
    // decrement the high-order part when recovering the saved value. To do
    // this correctly, we need to assume that less than 2^(32+cputime_shift-1)
    // ns have passed since the estimator was saved. This is 2200 seconds for
    // cputime_shift=10, way longer than our typical context switches.
    constexpr u64 ho = (std::numeric_limits<u64>::max() &
            ~(std::numeric_limits<u64>::max() >> (64 - 32 - cputime_shift)));
    u64 rs_ref = osv::clock::uptime::now().time_since_epoch().count();
    u64 tc_ref = _total_cpu_time.count();
    u64 rs_ho = rs_ref & ho;
    u64 tc_ho = tc_ref & ho;
    if ((rs_ref & ~ho) < rs) {
        rs_ho -= (1ULL << (32 + cputime_shift));
    }
    if ((tc_ref & ~ho) < tc) {
        tc_ho -= (1ULL << (32 + cputime_shift));
    }
    running_since = osv::clock::uptime::time_point(
            osv::clock::uptime::duration(rs_ho | rs));
    total_cpu_time = osv::clock::uptime::duration(tc_ho | tc);
}

// Note that this is a static (class) function, which can only reschedule
// on the current CPU, not on an arbitrary CPU. Allowing to run one CPU's
// scheduler on a different CPU would be disastrous.
void cpu::schedule()
{
    WITH_LOCK(irq_lock) {
        current()->reschedule_from_interrupt();
    }
}

void cpu::reschedule_from_interrupt(bool complete_stage_migration)
{
    trace_sched_sched();
    assert(sched::exception_depth <= 1);
    need_reschedule = false;
    handle_incoming_wakeups();
    if (!complete_stage_migration) {
        stage::dequeue();
    }

    auto now = osv::clock::uptime::now();
    auto interval = now - running_since;
    running_since = now;
    if (interval <= 0) {
        // During startup, the clock may be stuck and we get zero intervals.
        // To avoid scheduler loops, let's make it non-zero.
        // Also ignore backward jumps in the clock.
        interval = context_switch_penalty;
    }
    thread* p = thread::current();

    const auto p_status = p->_detached_state->st.load();
    assert(p_status != thread::status::queued);
    assert(!complete_stage_migration || p_status == thread::status::stagemig);

    p->_total_cpu_time += interval;

    if (p_status == thread::status::running) {
        if (runqueue.empty()) {
            /* we are the idle thread, let it run */
            return;
        }
        /* TODO work-conservation for running threads
         *
         * we should give global stage scheduling an opportunity to balance load
         * between CPUs by enqueing this thread into it's _detached_state->_stage
         */
        p->_detached_state->st.store(thread::status::queued);
        trace_sched_preempt();
        p->stat_preemptions.incr();
        enqueue(*p);
    } else if(p_status != thread::status::stagemig) {
        /* TODO work-conservation for blocking threads
         *
         * p is not runnable and not in flight for stage migration
         *
         * once it unblocks, it will be woken up on its detached_state cpu (this one)
         * via the wake_impl + handle_incoming_wakeups duo
         *
         * however, the stage scheduler should dispatch this unblocked thread
         * to ensure work-conservation
         *
         * When implementing this feature
         *  - try to de-duplicate the code for status::running above
         *  - take a look at how timers are implemented
         **/
    }

    /* Find a new thread from CPU-local runqueue
     * (system threads + already dequeued stage-aware threads) */
dequeue:
    auto ni = runqueue.begin();
    auto n = &*ni;
    runqueue.erase(ni);
    assert(n->_detached_state->st.load() == thread::status::queued);
    if (runqueue.size() >= 2 && n == idle_thread) {
        /* Poor-man's thread priorities */
        enqueue(*n);
        goto dequeue;
        return;
    }

    n->cputime_estimator_set(now, n->_total_cpu_time);

    if (n == idle_thread) {
        trace_sched_idle();
    } else if (p == idle_thread) {
        trace_sched_idle_ret();
    }
    n->stat_switches.incr();

    trace_sched_load(runqueue.size());

    n->_detached_state->st.store(thread::status::running);

    if (app_thread.load(std::memory_order_relaxed) != n->_app) { // don't write into a cache line if it can be avoided
        app_thread.store(n->_app, std::memory_order_relaxed);
    }
    if (lazy_flush_tlb.exchange(false, std::memory_order_seq_cst)) {
        mmu::flush_tlb_local();
    }
    n->switch_to(complete_stage_migration);

    // Note: after the call to n->switch_to(), we should no longer use any of
    // the local variables, nor "this" object, because we just switched to n's
    // stack and the values we can access now are those that existed in the
    // reschedule call which scheduled n out, and will now be returning.
    // So to get the current cpu, we must use cpu::current(), not "this".
    if (cpu::current()->terminating_thread) {
        cpu::current()->terminating_thread->destroy();
        cpu::current()->terminating_thread = nullptr;
    }
}

void cpu::timer_fired()
{
    // nothing to do, preemption will happen if needed
}

struct idle_poll_lock_type {
    explicit idle_poll_lock_type(cpu& c) : _c(c) {}
    void lock() { _c.idle_poll_start(); }
    void unlock() { _c.idle_poll_end(); }
    cpu& _c;
};

void cpu::idle_poll_start()
{
    idle_poll.store(true, std::memory_order_relaxed);
}

void cpu::idle_poll_end()
{
    idle_poll.store(false, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void cpu::send_wakeup_ipi()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!idle_poll.load(std::memory_order_relaxed) && runqueue.size() <= 1) {
        trace_sched_ipi(id);
        wakeup_ipi.send(this);
    }
}

void cpu::do_idle()
{
    do {
        // FIXME: mwait on stagesched_incoming, see mwait branch
        handle_incoming_wakeups();
        stage::dequeue();
        if (!runqueue.empty()) {
            return;
        }
    } while (runqueue.empty());
}

void start_early_threads();

void cpu::idle()
{
    // The idle thread must not sleep, because the whole point is that the
    // scheduler can always find at least one runnable thread.
    // We set preempt_disable just to help us verify this.
    preempt_disable();

    if (id == 0) {
        start_early_threads();
    }

    while (true) {
        do_idle();
        // We have idle priority, so this runs the thread on the runqueue:
        schedule();
    }
}

void cpu::handle_incoming_wakeups()
{
    cpu_set queues_with_wakes{incoming_wakeups_mask.fetch_clear()};
    if (!queues_with_wakes) {
        return;
    }
    for (auto i : queues_with_wakes) {
        irq_save_lock_type irq_lock;
        WITH_LOCK(irq_lock) {
            auto& q = incoming_wakeups[i];
            while (!q.empty()) {
                auto& t = q.front();
                q.pop_front();
                if (&t == thread::current()) {
                    // Special case of current thread being woken before
                    // having a chance to be scheduled out.
                    t._detached_state->st.store(thread::status::running);
                } else if (t.tcpu() != this) {
                    // Thread was woken on the wrong cpu. Can be a side-effect
                    // of sched::thread::pin(thread*, cpu*). Do nothing.
                } else {
                    t._detached_state->st.store(thread::status::queued);
                    enqueue(t);
                    t.resume_timers();
                }
            }
        }
    }

    trace_sched_load(runqueue.size());
}

void cpu::enqueue(thread& t)
{
    trace_sched_queue(&t);
    runqueue.push_back(t);
}

void cpu::init_on_cpu()
{
    arch.init_on_cpu();
    clock_event->setup_on_cpu();
}

unsigned cpu::load()
{
    return runqueue.size();
}

stage* stage::define(const std::string name) {

    static mutex _stages_mtx;
    static stage stages[stage::max_stages];
    static int stages_next;

    std::lock_guard<mutex> guard(_stages_mtx);

    if (stages_next == stage::max_stages)
        return nullptr;

    auto& next = stages[stages_next];
    next._id = stages_next;
    stages_next++;
    next._name = name;

    return &next;
}

void stage::enqueue()
{
    cpu *target_cpu = sched::cpus[_id]; // TODO policy

    /* prohibit migration of this thread off this cpu */
    irq_save_lock_type irq_lock;
    std::lock_guard<irq_save_lock_type> guard(irq_lock);

    cpu *source_cpu = cpu::current();
    thread *t = thread::current();

    trace_sched_stage_enqueue(this, source_cpu->id, target_cpu->id, t);

    // must be called from a thread executing on a CPU
    assert(t->_runqueue_link.is_linked() == false);
    //must be called from a runnable thread
    assert(t->_detached_state->st.load() == thread::status::running);

    if (target_cpu->id == source_cpu->id) {
        source_cpu->reschedule_from_interrupt(); // releases guard
        return;
    }

    t->_detached_state->st.store(thread::status::stagemig);

    /* status::stagemig prohibits target_cpu from executing current thread
       which is critical because we are still executing it right now on this cpu */

    /* thread migration code adopted + extended from thread::pin */
    t->stat_migrations.incr();
    t->suspend_timers();
    t->_detached_state->_cpu = target_cpu;
    percpu_base = target_cpu->percpu_base;
    current_cpu = target_cpu;

    // enqueue as late as possible to minimize the time c is in status::stagemig
    // but target_cpu->stagesched_incoming avoid target_cpu
    target_cpu->stagesched_incoming.push(t);

    /* find another thread to run on source_cpu and make sure that c is marked
     * runnable once source_cpu doesn't execute it anymore so that target_cpu
     * stops re-enqueuing it to its stagesched_incoming
     */
    source_cpu->reschedule_from_interrupt(true); // releases guard

    /* from here on, the calling thread is in target_cpu->stagesched_incoming
       or already in target_cpu->runqueue */
}

void stage::dequeue()
{

    /* prohibit migration of this thread off this cpu while dequeuing */
    irq_save_lock_type irq_lock;
    std::lock_guard<irq_save_lock_type> guard(irq_lock);

    auto inq = &cpu::current()->stagesched_incoming;

    /* fully drain inq
     * FIXME the runtime of the loop is unbounded.
     *       can only fix this once do_idle uses mwait */
    thread *t;
    while ((t = inq->pop()) != nullptr) {
        auto state = t->_detached_state->st.load();
        if (state == thread::status::stagemig_comp) {
            t->_detached_state->st.store(thread::status::queued);
            trace_sched_stage_dequeue(cpu::current()->id, t);
            cpu::current()->enqueue(*t);
            t->resume_timers();
        } else {
            /* This situation is unlikely:
             * t's source CPU has not completed the context switch yet.
             * The source_cpu is likely somewhere between stagesched_incoming.push() and
             * thread::switch_to's */
            assert(state == thread::status::stagemig);
            trace_sched_stage_dequeue_stagemig(cpu::current()->id, t);
            inq->push(t);
            /* When we pop t the next time, it will probably be status::stagemig_comp
             * so busy waiting is bounded here. */
        }
    }

}


// function to pin the *current* thread:
void thread::pin(cpu *target_cpu)
{
    // Note that this code may proceed to migrate the current thread even if
    // it was protected by a migrate_disable(). It is the thread's own fault
    // for doing this to itself...
    thread &t = *current();
    if (!t._pinned) {
        // _pinned comes with a +1 increase to _migration_counter.
        migrate_disable();
        t._pinned = true;
    }
    cpu *source_cpu = cpu::current();
    if (source_cpu == target_cpu) {
        return;
    }
    // We want to wake this thread on the target CPU, but can't do this while
    // it is still running on this CPU. So we need a different thread to
    // complete the wakeup. We could re-used an existing thread (e.g., the
    // load balancer thread) but a "good-enough" dirty solution is to
    // temporarily create a new ad-hoc thread, "wakeme".
    bool do_wakeme = false;
    std::unique_ptr<thread> wakeme(thread::make([&] () {
        wait_until([&] { return do_wakeme; });
        t.wake();
    }, sched::thread::attr().pin(source_cpu)));
    wakeme->start();
    WITH_LOCK(irq_lock) {
        trace_sched_migrate(&t, target_cpu->id);
        t.stat_migrations.incr();
        t.suspend_timers();
        t._detached_state->_cpu = target_cpu;
        percpu_base = target_cpu->percpu_base;
        current_cpu = target_cpu;
        t._detached_state->st.store(thread::status::waiting);
        // Note that wakeme is on the same CPU, and irq is disabled,
        // so it will not actually run until we stop running.
        wakeme->wake_with([&] { do_wakeme = true; });
        source_cpu->reschedule_from_interrupt();
    }
    // wakeme will be implicitly join()ed here.
}

void thread::unpin()
{
    // Unpinning the current thread is straightforward. But to work on a
    // different thread safely, without risking races with concurrent attempts
    // to pin, unpin, or migrate the same thread, we need to run the actual
    // unpinning code on the same CPU as the target thread.
    if (this == current()) {
        WITH_LOCK(preempt_lock) {
            if (_pinned) {
                _pinned = false;
                 std::atomic_signal_fence(std::memory_order_release);
                _migration_lock_counter--;
            }
        }
        return;
    }
    std::unique_ptr<thread> helper(thread::make([this] {
        WITH_LOCK(preempt_lock) {
            // helper thread started on the same CPU as "this", but by now
            // "this" might migrated. If that happened helper need to migrate.
            while (sched::cpu::current() != this->tcpu()) {
                DROP_LOCK(preempt_lock) {
                    thread::pin(this->tcpu());
                }
            }
            if (_pinned) {
                _pinned = false;
                 std::atomic_signal_fence(std::memory_order_release);
                _migration_lock_counter--;
            }
        }
    }, sched::thread::attr().pin(tcpu())));
    helper->start();
    helper->join();
}

void cpu::bring_up() {
    notifier::fire();
}

cpu::notifier::notifier(std::function<void ()> cpu_up)
    : _cpu_up(cpu_up)
{
    WITH_LOCK(_mtx) {
        _notifiers.push_back(this);
    }
}

cpu::notifier::~notifier()
{
    WITH_LOCK(_mtx) {
        _notifiers.remove(this);
    }
}

void cpu::notifier::fire()
{
    WITH_LOCK(_mtx) {
        for (auto n : _notifiers) {
            n->_cpu_up();
        }
    }
}

void thread::yield(thread_runtime::duration preempt_after)
{
    trace_sched_yield();
    auto t = current();
    std::lock_guard<irq_lock_type> guard(irq_lock);
    // FIXME: drive by IPI
    cpu::current()->handle_incoming_wakeups();
    // FIXME: what about other cpus?
    if (cpu::current()->runqueue.empty()) {
        return;
    }
    assert(t->_detached_state->st.load() == status::running);
    // Do not yield to a thread with idle priority
    thread &tnext = *(cpu::current()->runqueue.begin());
    if (tnext.priority() == thread::priority_idle) {
        return;
    }
    trace_sched_yield_switch();

    cpu::current()->reschedule_from_interrupt();
}

void thread::set_priority(float priority)
{
    // NOOP
}

float thread::priority() const
{
    return priority_default;
}

sched::thread::status thread::get_status() const
{
    return _detached_state->st.load(std::memory_order_relaxed);
}

thread::stack_info::stack_info()
    : begin(nullptr), size(0), deleter(nullptr)
{
}

thread::stack_info::stack_info(void* _begin, size_t _size)
    : begin(_begin), size(_size), deleter(nullptr)
{
    auto end = align_down(begin + size, 16);
    size = static_cast<char*>(end) - static_cast<char*>(begin);
}

void thread::stack_info::default_deleter(thread::stack_info si)
{
    free(si.begin);
}

// thread_map is used for a list of all threads, but also as a map from
// numeric (4-byte) threads ids to the thread object, to support Linux
// functions which take numeric thread ids.
static mutex thread_map_mutex;
using id_type = std::result_of<decltype(&thread::id)(thread)>::type;
std::unordered_map<id_type, thread *> thread_map
    __attribute__((init_priority((int)init_prio::threadlist)));

static thread_runtime::duration total_app_time_exited(0);

thread_runtime::duration thread::thread_clock() {
    if (this == current()) {
        WITH_LOCK (preempt_lock) {
            // Inside preempt_lock, we are running and the scheduler can't
            // intervene and change _total_cpu_time or _running_since
            return _total_cpu_time +
                    (osv::clock::uptime::now() - tcpu()->running_since);
        }
    } else {
        auto status = _detached_state->st.load(std::memory_order_acquire);
        if (status == thread::status::running) {
            // The cputime_estimator set before the status is already visible.
            // Even if the thread stops running now, cputime_estimator will
            // remain; Our max overshoot will be the duration of this code.
            osv::clock::uptime::time_point running_since;
            osv::clock::uptime::duration total_cpu_time;
            cputime_estimator_get(running_since, total_cpu_time);
            return total_cpu_time +
                    (osv::clock::uptime::now() - running_since);
        } else {
            // _total_cpu_time is set before setting status, so it is already
            // visible. During this code, the thread might start running, but
            // it doesn't matter, total_cpu_time will remain. Our maximum
            // undershoot will be the duration that this code runs.
            // FIXME: we assume reads/writes to _total_cpu_time are atomic.
            // They are, but we should use std::atomic to guarantee that.
            return _total_cpu_time;
        }
    }
}

// Return the total amount of cpu time used by the process. This is the amount
// of time that passed since boot multiplied by the number of CPUs, from which
// we subtract the time spent in the idle threads.
// Besides the idle thread, we do not currently account for "steal time",
// i.e., time in which the hypervisor preempted us and ran other things.
// In other words, when a hypervisor gives us only a part of a CPU, we pretend
// it is still a full CPU, just a slower one. Ordinary CPUs behave similarly
// when faced with variable-speed CPUs.
osv::clock::uptime::duration process_cputime()
{
    // FIXME: This code does not handle the possibility of CPU hot-plugging.
    // See issue #152 for a suggested solution.
    auto ret = osv::clock::uptime::now().time_since_epoch();
    ret *= sched::cpus.size();
    for (sched::cpu *cpu : sched::cpus) {
        ret -= cpu->idle_thread->thread_clock();
    }
    // idle_thread->thread_clock() may make tiny (<microsecond) temporary
    // mistakes when racing with the idle thread's starting or stopping.
    // To ensure that process_cputime() remains monotonous, we monotonize it.
    static std::atomic<osv::clock::uptime::duration> lastret;
    auto l = lastret.load(std::memory_order_relaxed);
    while (ret > l &&
           !lastret.compare_exchange_weak(l, ret, std::memory_order_relaxed));
    if (ret < l) {
        ret = l;
    }
    return ret;
}

std::chrono::nanoseconds osv_run_stats()
{
    thread_runtime::duration total_app_time;

    WITH_LOCK(thread_map_mutex) {
        total_app_time = total_app_time_exited;
        for (auto th : thread_map) {
            thread *t = th.second;
            total_app_time += t->thread_clock();
        }
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(total_app_time);
}

int thread::numthreads()
{
    SCOPE_LOCK(thread_map_mutex);
    return thread_map.size();
}

// We reserve a space in the end of the PID space, so we can reuse those
// special purpose ids for other things. 4096 positions is arbitrary, but
// <<should be enough for anybody>> (tm)
constexpr unsigned int tid_max = UINT_MAX - 4096;
unsigned long thread::_s_idgen = 0;

thread *thread::find_by_id(unsigned int id)
{
    auto th = thread_map.find(id);
    if (th == thread_map.end())
        return NULL;
    return (*th).second;
}

void* thread::do_remote_thread_local_var(void* var)
{
    auto tls_cur = static_cast<char*>(current()->_tcb->tls_base);
    auto tls_this = static_cast<char*>(this->_tcb->tls_base);
    auto offset = static_cast<char*>(var) - tls_cur;
    return tls_this + offset;
}

thread::thread(std::function<void ()> func, attr attr, bool main, bool app)
    : _func(func)
    , _detached_state(new detached_state(this))
    , _attr(attr)
    , _migration_lock_counter(0)
    , _pinned(false)
    , _id(0)
    , _cleanup([this] { delete this; })
    , _app(app)
    , _joiner(nullptr)
{
    trace_thread_create(this);

    if (!main && sched::s_current) {
        auto app = application::get_current().get();
        if (override_current_app) {
            app = override_current_app;
        }
        if (_app && app) {
            _app_runtime = app->runtime();
        }
    }
    setup_tcb();
    // module 0 is always the core:
    assert(_tls.size() == elf::program::core_module_index);
    _tls.push_back((char *)_tcb->tls_base);
    if (_app_runtime) {
        auto& offsets = _app_runtime->app.lib()->initial_tls_offsets();
        for (unsigned i = 1; i < offsets.size(); i++) {
            if (!offsets[i]) {
                _tls.push_back(nullptr);
            } else {
                _tls.push_back(reinterpret_cast<char*>(_tcb) + offsets[i]);
            }
        }
    }

    WITH_LOCK(thread_map_mutex) {
        if (!main) {
            auto ttid = _s_idgen;
            auto tid = ttid;
            do {
                tid++;
                if (tid > tid_max) { // wrap around
                    tid = 1;
                }
                if (!find_by_id(tid)) {
                    _s_idgen = _id = tid;
                    thread_map.insert(std::make_pair(_id, this));
                    break;
                }
            } while (tid != ttid); // One full round trip is enough
            if (tid == ttid) {
                abort("Can't allocate a Thread ID");
            }
        }
    }
    // setup s_current before switching to the thread, so interrupts
    // can call thread::current()
    // remote_thread_local_var() doesn't work when there is no current
    // thread, so don't do this for main threads (switch_to_first will
    // do that for us instead)
    if (!main && sched::s_current) {
        remote_thread_local_var(s_current) = this;
    }
    init_stack();

    if (_attr._detached) {
        _detach_state.store(detach_state::detached);
    }

    if (_attr._pinned_cpu) {
        ++_migration_lock_counter;
        _pinned = true;
    }

    if (main) {
        _detached_state->_cpu = attr._pinned_cpu;
        _detached_state->st.store(status::running);
        if (_detached_state->_cpu == sched::cpus[0]) {
            s_current = this;
        }
        remote_thread_local_var(current_cpu) = _detached_state->_cpu;
    }

    // For debugging purposes, it is useful for threads to have names. If no
    // name was set for this one, set one by prepending ">" to parent's name.
    if (!_attr._name[0] && s_current) {
        _attr._name[0] = '>';
        strncpy(_attr._name.data()+1, s_current->_attr._name.data(),
                sizeof(_attr._name) - 2);
    }
}

static std::list<std::function<void ()>> exit_notifiers
        __attribute__((init_priority((int)init_prio::threadlist)));
static rwlock exit_notifiers_lock
        __attribute__((init_priority((int)init_prio::threadlist)));
void thread::register_exit_notifier(std::function<void ()> &&n)
{
    WITH_LOCK(exit_notifiers_lock.for_write()) {
        exit_notifiers.push_front(std::move(n));
    }
}
static void run_exit_notifiers()
{
    WITH_LOCK(exit_notifiers_lock.for_read()) {
        for (auto& notifier : exit_notifiers) {
            notifier();
        }
    }
}

// not in the header to avoid double inclusion between osv/app.hh and
// osv/sched.hh
osv::application *thread::current_app() {
    auto cur = current();

    if (!cur->_app_runtime) {
        return nullptr;
    }

    return &(cur->_app_runtime->app);
}

thread::~thread()
{
    cancel_this_thread_alarm();

    if (!_attr._detached) {
        join();
    }
    WITH_LOCK(thread_map_mutex) {
        thread_map.erase(_id);
        total_app_time_exited += _total_cpu_time;
    }
    if (_attr._stack.deleter) {
        _attr._stack.deleter(_attr._stack);
    }
    for (unsigned i = 1; i < _tls.size(); i++) {
        if (_app_runtime) {
            auto& offsets = _app_runtime->app.lib()->initial_tls_offsets();
            if (i < offsets.size() && offsets[i]) {
                continue;
            }
        }
        delete[] _tls[i];
    }
    free_tcb();
    rcu_dispose(_detached_state.release());
}

void thread::start()
{
    assert(_detached_state->st == status::unstarted);

    if (!sched::s_current) {
        _detached_state->st.store(status::prestarted);
        return;
    }

    _detached_state->_cpu = _attr._pinned_cpu ? _attr._pinned_cpu : current()->tcpu();
    remote_thread_local_var(percpu_base) = _detached_state->_cpu->percpu_base;
    remote_thread_local_var(current_cpu) = _detached_state->_cpu;
    _detached_state->st.store(status::waiting);
    wake();
}

void thread::prepare_wait()
{
    // After setting the thread's status to "waiting", we must not preempt it,
    // as it is no longer in "running" state and therefore will not return.
    preempt_disable();
    assert(_detached_state->st.load() == status::running);
    _detached_state->st.store(status::waiting);
}

// This function is responsible for changing a thread's state from
// "terminating" to "terminated", while also waking a thread sleeping on
// join(), if any.
// This function cannot be called by the dying thread, because waking its
// joiner usually triggers deletion of the thread and its stack, and it
// must not be running at the same time.
// TODO: rename this function, perhaps to wake_joiner()?
void thread::destroy()
{
    // thread can't destroy() itself, because if it decides to wake joiner,
    // it will delete the stack it is currently running on.
    assert(thread::current() != this);

    assert(_detached_state->st.load(std::memory_order_relaxed) == status::terminating);
    // Solve a race between join() and the thread's completion. If join()
    // manages to set _joiner first, it will sleep and we need to wake it.
    // But if we set _joiner first, join() will never wait.
    sched::thread *joiner = nullptr;
    WITH_LOCK(rcu_read_lock_in_preempt_disabled) {
        auto ds = _detached_state.get();
        // Note we can't set status to "terminated" before the CAS on _joiner:
        // As soon as we set status to terminated, a concurrent join might
        // return and delete the thread, and _joiner will become invalid.
        if (_joiner.compare_exchange_strong(joiner, this)) {
            // In this case, the concurrent join() may have already noticed it
            // lost the race, returned, and the thread "this" may have been
            // deleted. But ds is still valid because of RCU lock.
            ds->st.store(status::terminated);
        } else {
            // The joiner won the race, and will wait. We need to wake it.
            joiner->wake_with([&] { ds->st.store(status::terminated); });
        }
    }
}

// Must be called under rcu_read_lock
//
// allowed_initial_states_mask *must* contain status::waiting, and
// *may* contain status::sending_lock (for waitqueue wait morphing).
// it will transition from one of the allowed initial states to the
// waking state.
void thread::wake_impl(detached_state* st, unsigned allowed_initial_states_mask)
{
    status old_status = status::waiting;
    trace_sched_wake(st->t);
    while (!st->st.compare_exchange_weak(old_status, status::waking)) {
        if (!((1 << unsigned(old_status)) & allowed_initial_states_mask)) {
            return;
        }
    }
    auto tcpu = st->_cpu;
    WITH_LOCK(preempt_lock_in_rcu) {
        unsigned c = cpu::current()->id;
        // we can now use st->t here, since the thread cannot terminate while
        // it's waking, but not afterwards, when it may be running
        irq_save_lock_type irq_lock;
        WITH_LOCK(irq_lock) {
            tcpu->incoming_wakeups[c].push_back(*st->t);
        }
        if (!tcpu->incoming_wakeups_mask.test_all_and_set(c)) {
            // FIXME: avoid if the cpu is alive and if the priority does not
            // FIXME: warrant an interruption
            if (tcpu != current()->tcpu()) {
                tcpu->send_wakeup_ipi();
            } else {
                need_reschedule = true;
            }
        }
    }
}

void thread::wake()
{
    WITH_LOCK(rcu_read_lock) {
        wake_impl(_detached_state.get());
    }
}

void thread::wake_lock(mutex* mtx, wait_record* wr)
{
    // must be called with mtx held
    WITH_LOCK(rcu_read_lock) {
        auto st = _detached_state.get();
        // We want to send_lock() to this thread, but we want to be sure we're the only
        // ones doing it, and that it doesn't wake up while we do
        auto expected = status::waiting;
        if (!st->st.compare_exchange_strong(expected, status::sending_lock, std::memory_order_relaxed)) {
            // make sure the thread can see wr->woken() == true.  We're still protected by
            // the mutex, so so need for extra protection
            wr->clear();
            // let the thread acquire the lock itself
            return;
        }
        // Send the lock to the thread, unless someone else already woke the us up,
        // and we're sleeping in mutex::lock().
        if (mtx->send_lock_unless_already_waiting(wr)) {
            st->lock_sent = true;
        } else {
            st->st.store(status::waiting, std::memory_order_relaxed);
            wr->clear();
        }
        // since we're in status::sending_lock, no one can wake us except mutex::unlock
    }
}

bool thread::unsafe_stop()
{
    WITH_LOCK(rcu_read_lock) {
        auto st = _detached_state.get();
        auto expected = status::waiting;
        return st->st.compare_exchange_strong(expected,
                status::terminated, std::memory_order_relaxed)
                || expected == status::terminated;
    }
}


void thread::main()
{
    _func();
}

void thread::wait()
{
    trace_sched_wait();
    cpu::schedule();
    trace_sched_wait_ret();
}

void thread::stop_wait()
{
    // Can only re-enable preemption of this thread after it is no longer
    // in "waiting" state (otherwise if preempted, it will not be scheduled
    // in again - this is why we disabled preemption in prepare_wait.
    status old_status = status::waiting;
    auto& st = _detached_state->st;
    if (st.compare_exchange_strong(old_status, status::running)) {
        preempt_enable();
        return;
    }
    preempt_enable();
    if (old_status == status::terminated) {
        // We raced with thread::unsafe_stop() and lost
        cpu::schedule();
    }
    while (st.load() == status::waking || st.load() == status::sending_lock) {
        cpu::schedule();
    }
    assert(st.load() == status::running);
}

void thread::complete()
{
    run_exit_notifiers();

    auto value = detach_state::attached;
    _detach_state.compare_exchange_strong(value, detach_state::attached_complete);
    if (value == detach_state::detached) {
        _s_reaper->add_zombie(this);
    }
    // If this thread gets preempted after changing status it will never be
    // scheduled again to set terminating_thread. So must disable preemption.
    preempt_disable();
    _detached_state->st.store(status::terminating);
    // We want to run destroy() here, but can't because it cause the stack we're
    // running on to be deleted. Instead, set a _cpu field telling the next
    // thread running on this cpu to do the unref() for us.
    if (_detached_state->_cpu->terminating_thread) {
        assert(_detached_state->_cpu->terminating_thread != this);
        _detached_state->_cpu->terminating_thread->destroy();
    }
    _detached_state->_cpu->terminating_thread = this;
    // The thread is now in the "terminating" state, so on call to schedule()
    // it will never get to run again.
    while (true) {
        cpu::schedule();
    }
}

/*
 * Exit a thread.  Doesn't unwind any C++ ressources, and should
 * only be used to implement higher level threading abstractions.
 */
void thread::exit()
{
    thread* t = current();

    t->complete();
}

void timer_base::client::suspend_timers()
{
    if (_timers_need_reload) {
        return;
    }
    _timers_need_reload = true;
    cpu::current()->timers.suspend(_active_timers);
}

void timer_base::client::resume_timers()
{
    if (!_timers_need_reload) {
        return;
    }
    _timers_need_reload = false;
    cpu::current()->timers.resume(_active_timers);
}

void thread::join()
{
    auto& st = _detached_state->st;
    if (st.load() == status::unstarted) {
        // To allow destruction of a thread object before start().
        return;
    }
    sched::thread *old_joiner = nullptr;
    if (!_joiner.compare_exchange_strong(old_joiner, current())) {
        // The thread is concurrently completing and took _joiner in destroy().
        // At this point we know that destroy() will no longer use 'this', so
        // it's fine to return and for our caller to delete the thread.
        return;
    }
    wait_until([&] { return st.load() == status::terminated; });
}

void thread::detach()
{
    _attr._detached = true;
    auto value = detach_state::attached;
    _detach_state.compare_exchange_strong(value, detach_state::detached);
    if (value == detach_state::attached_complete) {
        // Complete was called prior to our call to detach. If we
        // don't add ourselves to the reaper now, nobody will.
        _s_reaper->add_zombie(this);
    }
}

thread::stack_info thread::get_stack_info()
{
    return _attr._stack;
}

void thread::set_cleanup(std::function<void ()> cleanup)
{
    assert(_detached_state->st == status::unstarted);
    _cleanup = cleanup;
}

void thread::timer_fired()
{
    wake();
}

unsigned int thread::id() const
{
    return _id;
}

void thread::set_name(std::string name)
{
    _attr.name(name);
}

std::string thread::name() const
{
    return _attr._name.data();
}

void* thread::setup_tls(ulong module, const void* tls_template,
        size_t init_size, size_t uninit_size)
{
    _tls.resize(std::max(module + 1, _tls.size()));
    _tls[module]  = new char[init_size + uninit_size];
    auto p = _tls[module];
    memcpy(p, tls_template, init_size);
    memset(p + init_size, 0, uninit_size);
    return p;
}

void thread::sleep_impl(timer &t)
{
    wait_until([&] { return t.expired(); });
}

void thread_handle::wake()
{
    WITH_LOCK(rcu_read_lock) {
        thread::detached_state* ds = _t.read();
        if (ds) {
            thread::wake_impl(ds);
        }
    }
}

timer_list::callback_dispatch::callback_dispatch()
{
    clock_event->set_callback(this);
}

void timer_list::fired()
{
    auto now = osv::clock::uptime::now();
 again:
    _last = osv::clock::uptime::time_point::max();
    _list.expire(now);
    timer_base* timer;
    while ((timer = _list.pop_expired())) {
        assert(timer->_state == timer_base::state::armed);
        timer->expire();
    }
    if (!_list.empty()) {
        // We could have simply called rearm() here, but this would lead to
        // recursion if the next timer has already expired in the time that
        // passed above. Better iterate in that case, instead.
        now = osv::clock::uptime::now();
        auto t = _list.get_next_timeout();
        if (t <= now) {
            goto again;
        } else {
            _last = t;
            clock_event->set(t - now);
        }
    }
}

void timer_list::rearm()
{
    auto t = _list.get_next_timeout();
    if (t < _last) {
        _last = t;
        clock_event->set(t - osv::clock::uptime::now());
    }
}

// call with irq disabled
void timer_list::suspend(timer_base::client_list_t& timers)
{
    for (auto& t : timers) {
        assert(t._state == timer::state::armed);
        _list.remove(t);
    }
}

// call with irq disabled
void timer_list::resume(timer_base::client_list_t& timers)
{
    bool do_rearm = false;
    for (auto& t : timers) {
        assert(t._state == timer::state::armed);
        do_rearm |= _list.insert(t);
    }
    if (do_rearm) {
        rearm();
    }
}

void timer_list::callback_dispatch::fired()
{
    cpu::current()->timers.fired();
}

timer_list::callback_dispatch timer_list::_dispatch;

timer_base::timer_base(timer_base::client& t)
    : _t(t)
{
}

timer_base::~timer_base()
{
    cancel();
}

void timer_base::expire()
{
    trace_timer_fired(this);
    _state = state::expired;
    _t._active_timers.erase(_t._active_timers.iterator_to(*this));
    _t.timer_fired();
}

void timer_base::set(osv::clock::uptime::time_point time)
{
    trace_timer_set(this, time.time_since_epoch().count());
    irq_save_lock_type irq_lock;
    WITH_LOCK(irq_lock) {
        _state = state::armed;
        _time = time;

        auto& timers = cpu::current()->timers;
        _t._active_timers.push_back(*this);
        if (timers._list.insert(*this)) {
            timers.rearm();
        }
    }
};

void timer_base::cancel()
{
    if (_state == state::free) {
        return;
    }
    trace_timer_cancel(this);
    irq_save_lock_type irq_lock;
    WITH_LOCK(irq_lock) {
        if (_state == state::armed) {
            _t._active_timers.erase(_t._active_timers.iterator_to(*this));
            cpu::current()->timers._list.remove(*this);
        }
        _state = state::free;
    }
    // even if we remove the first timer, allow it to expire rather than
    // reprogramming the timer
}

void timer_base::reset(osv::clock::uptime::time_point time)
{
    trace_timer_reset(this, time.time_since_epoch().count());

    auto& timers = cpu::current()->timers;

    irq_save_lock_type irq_lock;
    WITH_LOCK(irq_lock) {
        if (_state == state::armed) {
            timers._list.remove(*this);
        } else {
            _t._active_timers.push_back(*this);
            _state = state::armed;
        }

        _time = time;

        if (timers._list.insert(*this)) {
            timers.rearm();
        }
    }
};

bool timer_base::expired() const
{
    return _state == state::expired;
}

bool operator<(const timer_base& t1, const timer_base& t2)
{
    if (t1._time < t2._time) {
        return true;
    } else if (t1._time == t2._time) {
        return &t1 < &t2;
    } else {
        return false;
    }
}

thread::reaper::reaper()
    : _mtx{}, _zombies{}, _thread(thread::make([=] { reap(); }))
{
    _thread->start();
}

void thread::reaper::reap()
{
    while (true) {
        WITH_LOCK(_mtx) {
            wait_until(_mtx, [=] { return !_zombies.empty(); });
            while (!_zombies.empty()) {
                auto z = _zombies.front();
                _zombies.pop_front();
                z->join();
                z->_cleanup();
            }
        }
    }
}

void thread::reaper::add_zombie(thread* z)
{
    assert(z->_attr._detached);
    WITH_LOCK(_mtx) {
        _zombies.push_back(z);
        _thread->wake();
    }
}

thread::reaper *thread::_s_reaper;

void init_detached_threads_reaper()
{
    thread::_s_reaper = new thread::reaper;
}

void start_early_threads()
{
    // We're called from the idle thread, which must not sleep, hence this
    // strange try_lock() loop instead of just a lock().
    while (!thread_map_mutex.try_lock()) {
        cpu::schedule();
    }
    SCOPE_ADOPT_LOCK(thread_map_mutex);
    for (auto th : thread_map) {
        thread *t = th.second;
        if (t == sched::thread::current()) {
            continue;
        }
        t->remote_thread_local_var(s_current) = t;
        thread::status expected = thread::status::prestarted;
        if (t->_detached_state->st.compare_exchange_strong(expected,
                thread::status::unstarted, std::memory_order_relaxed)) {
            t->start();
        }
    }
}

void init(std::function<void ()> cont)
{
    thread::attr attr;
    attr.stack(4096*10).pin(smp_initial_find_current_cpu());
    attr.name("init");
    thread t{cont, attr, true};
    t.switch_to_first();
}

void init_tls(elf::tls_data tls_data)
{
    tls = tls_data;
}

size_t kernel_tls_size()
{
    return tls.size;
}

void with_all_threads(std::function<void(thread &)> f) {
    WITH_LOCK(thread_map_mutex) {
        for (auto th : thread_map) {
            f(*th.second);
        }
    }
}

void with_thread_by_id(unsigned id, std::function<void(thread *)> f) {
    WITH_LOCK(thread_map_mutex) {
        f(thread::find_by_id(id));
    }
}


}

irq_lock_type irq_lock;
