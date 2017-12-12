/*
 * Copyright (C) 2013 Cloudius Systems, Ltd.
 *
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */

#ifndef ARCH_SWITCH_HH_
#define ARCH_SWITCH_HH_

#include "msr.hh"
#include <osv/barrier.hh>
#include <string.h>

extern "C" {
void thread_main(void);
void thread_main_c(sched::thread* t);
}

namespace sched {

void set_fsbase_msr(u64 v)
{
    processor::wrmsr(msr::IA32_FS_BASE, v);
}

void set_fsbase_fsgsbase(u64 v)
{
    processor::wrfsbase(v);
}

extern "C"
void (*resolve_set_fsbase(void))(u64 v)
{
    // can't use processor::features, because it is not initialized
    // early enough.
    if (processor::features().fsgsbase) {
        return set_fsbase_fsgsbase;
    } else {
        return set_fsbase_msr;
    }
}

void set_fsbase(u64 v) __attribute__((ifunc("resolve_set_fsbase")));

void thread::switch_to(bool complete_stage_migration)
{
    thread* old = current();
    // writing to fs_base invalidates memory accesses, so surround with
    // barriers
    barrier();
    set_fsbase(reinterpret_cast<u64>(_tcb));
    barrier();
    auto c = _detached_state->_cpu;
    old->_state.exception_stack = c->arch.get_exception_stack();
    c->arch.set_interrupt_stack(&_arch);
    c->arch.set_exception_stack(_state.exception_stack);
    auto fpucw = processor::fnstcw();
    auto mxcsr = processor::stmxcsr();

    /* When completing stage migration, the old thread must have status::stagemig.
       When not completing stage migration, we shouldn't do anything. */
    static_assert(sizeof(old->_detached_state->st) == 4,
            "switch_to performs the equivalent of "
            "a.compare_exchange_strong(cmp_old_status, status::running) "
            "but without touching the stack");
    static_assert(sizeof(status::stagemig_comp) == 4);
    asm volatile
        (
         "mov %%rbp, %c[rbp](%[ost]) \n\t"
         "movq $1f, %c[rip](%[ost]) \n\t"
         "mov %%rsp, %c[rsp](%[ost]) \n\t"
         "cmp $0, %%rax\n\t"
         "je switch \n\t"
         "lock xchg %%edx, (%%rsi)\n\t" // use 32bit regs because sizeof(_detached_state->st) == 4
         "switch: \n\t"
         "mov %c[rsp](%[nst]), %%rsp \n\t"
         "mov %c[rbp](%[nst]), %%rbp \n\t"
         "jmpq *%c[rip](%[nst]) \n\t"
         "1: \n\t"
         // NOTE: register allocation is done manually using constraints (had errors on gcc 7.2.1)
         :
         : "S"(&old->_detached_state->st), // rsi/esi
           [ost]"b"(&old->_state), // rbx/ebx
           [nst]"c"(&this->_state), // rcx/ecx
           [rsp]"i"(offsetof(thread_state, rsp)),
           [rbp]"i"(offsetof(thread_state, rbp)),
           [rip]"i"(offsetof(thread_state, rip)),
           "a"(complete_stage_migration ? 1 : 0),
           "d"(status::stagemig_comp) //rdx/edx
         : // used above: rax, rbx, rcx, rdx, rsi
           "rdi", "r8", "r9",
           "r10", "r11", "r12", "r13", "r14", "r15",
           "memory", // we switch the stack
           "cc" // cmpxchg clobbers flags register
               );
    processor::fldcw(fpucw);
    processor::ldmxcsr(mxcsr);
}

void thread::switch_to_first()
{
    barrier();
    processor::wrmsr(msr::IA32_FS_BASE, reinterpret_cast<u64>(_tcb));
    barrier();
    s_current = this;
    current_cpu = _detached_state->_cpu;
    remote_thread_local_var(percpu_base) = _detached_state->_cpu->percpu_base;
    _detached_state->_cpu->arch.set_interrupt_stack(&_arch);
    _detached_state->_cpu->arch.set_exception_stack(&_arch);
    asm volatile
        ("mov %c[rsp](%0), %%rsp \n\t"
         "mov %c[rbp](%0), %%rbp \n\t"
         "jmp *%c[rip](%0)"
         :
         : "c"(&this->_state),
           [rsp]"i"(offsetof(thread_state, rsp)),
           [rbp]"i"(offsetof(thread_state, rbp)),
           [rip]"i"(offsetof(thread_state, rip))
         : "rbx", "rdx", "rsi", "rdi", "r8", "r9",
           "r10", "r11", "r12", "r13", "r14", "r15", "memory");
}

void thread::init_stack()
{
    auto& stack = _attr._stack;
    if (!stack.size) {
        stack.size = 65536;
    }
    if (!stack.begin) {
        stack.begin = malloc(stack.size);
        stack.deleter = stack.default_deleter;
    }
    void** stacktop = reinterpret_cast<void**>(stack.begin + stack.size);
    _state.rbp = this;
    _state.rip = reinterpret_cast<void*>(thread_main);
    _state.rsp = stacktop;
    _state.exception_stack = _arch.exception_stack + sizeof(_arch.exception_stack);
}

void thread::setup_tcb()
{
    assert(tls.size);

    void* user_tls_data;
    size_t user_tls_size = 0;
    if (_app_runtime) {
        auto obj = _app_runtime->app.lib();
        assert(obj);
        user_tls_size = obj->initial_tls_size();
        user_tls_data = obj->initial_tls();
    }

    // In arch/x64/loader.ld, the TLS template segment is aligned to 64
    // bytes, and that's what the objects placed in it assume. So make
    // sure our copy is allocated with the same 64-byte alignment, and
    // verify that object::init_static_tls() ensured that user_tls_size
    // also doesn't break this alignment .
    assert(align_check(tls.size, (size_t)64));
    assert(align_check(user_tls_size, (size_t)64));
    void* p = aligned_alloc(64, sched::tls.size + user_tls_size + sizeof(*_tcb));
    if (user_tls_size) {
        memcpy(p, user_tls_data, user_tls_size);
    }
    memcpy(p + user_tls_size, sched::tls.start, sched::tls.filesize);
    memset(p + user_tls_size + sched::tls.filesize, 0,
           sched::tls.size - sched::tls.filesize);
    _tcb = static_cast<thread_control_block*>(p + tls.size + user_tls_size);
    _tcb->self = _tcb;
    _tcb->tls_base = p + user_tls_size;
}

void thread::free_tcb()
{
    if (_app_runtime) {
        auto obj = _app_runtime->app.lib();
        free(_tcb->tls_base - obj->initial_tls_size());
    } else {
        free(_tcb->tls_base);
    }
}

void thread_main_c(thread* t)
{
    arch::irq_enable();
#ifdef CONF_preempt
    preempt_enable();
#endif
    // make sure thread starts with clean fpu state instead of
    // inheriting one from a previous running thread
    processor::init_fpu();
    t->main();
    t->complete();
}

}

#endif /* ARCH_SWITCH_HH_ */
