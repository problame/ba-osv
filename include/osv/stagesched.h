/*
 * Copyright (C) 2017 Christian Schwarz
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */
#ifndef _STAGESCHED_H
#define _STAGESCHED_H

#ifdef __cplusplus

#include <string>
#include <stack>

#include <osv/mutex.h>

namespace sched {

    struct cpu;

    /**
     * Abstraction for stage-aware scheduling
     *
     * Data structure that tracks the stage queued containing threads.
     *
     * Threads enqueue themselves into a particular stage queue,
     * CPUs dequeue blindly from this data structure.
     *
     * This isolates queuing implementation and scheduling policy.
     */
    class stage {
        public:

            /**
             * Defines a new stage.
             * If no more stages can be defined (max_stages), returns NULL.
             */
            static stage* define(const std::string name);

            /**
             * Switch the current thread to this stage.
             * The thread must already be dequeued from the cpu's runqueue.
             */
            void enqueue();

            /** FIXME: should be private
             * Dequeue runnable threads into the calling cpu's runqueue.
             *
             * The policy which thread in which stage is dequeued is subject
             * to implementation of this data structure.
             */
            static void dequeue();

        private:
            stage() {}
            stage(stage const&) {}
            void operator=(stage const&) {}

            cpu *enqueue_policy();
            friend class thread;

        public:
            static const int max_stages = 32;
        private: /* global state */
            static stage stages[max_stages];
            static mutex stages_mtx;
            static int stages_next;

        private: /* per-instance state */
            int _id;
            std::string _name;
    };

    /**
     * Sometimes, one piece of code P called from multipler callers
     * should always execute in a stage S.
     * Thus, it makes sense that P contains the code to migrate to S
     * to avoid duplication.
     * But when P is about to return to its caller,
     * it does not know which stage the caller was in.
     *
     * A stack of stages is the natural abstraction for this,
     * which is what stage_stack is all about.
     **/
    class stage_stack {
        public:
            stage_stack(stage* default_stage, bool disabled_ = false)
                : disabled(disabled_) {
                    begin(default_stage);
            }

            /**
             * Begin executing code in a given stage
             **/
            void begin(sched::stage* stage) {
                if (disabled) return;
                stack.push(stage);
                stage->enqueue();
            }

            /**
             * Finish executing the current stage and return to previous
             * stage or the default stage specified on initalization.
             **/
            void finish() {
                if (disabled) return;
                if (stack.size() > 1) {
                    stack.pop();
                }
                stack.top()->enqueue();
            }

            /**
             * Switch to a given stage, replacing the current one
             **/
            void switch_to(sched::stage* stage) {
                if (disabled) return;
                stack.emplace(stage);
                stage->enqueue();
            }

        private:
            std::stack<sched::stage*> stack;
            bool disabled;

        public:

            /**
             * A guard is the equivalent of a std::lock_guard for stage_stack.
             * It uses the RAII paradigm to push the next stage onto the
             * stage_stack on initialization and pops from that stack at
             * the en* of scope.
             **/
            class guard {
                public:
                    guard(stage_stack& stack, stage *next) : s(stack) {
                        s.begin(next);
                    }
                    ~guard() {
                        s.finish();
                    }
                private:
                    stage_stack& s;
            };

    };

}

#endif /* __cplusplus */
#endif /*_STAGESCHED_H */
