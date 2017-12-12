/*
 * Copyright (C) 2017 Christian Schwarz
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */
#ifndef _STAGESCHED_H
#define _STAGESCHED_H

#ifdef __cplusplus

#include <string>

namespace sched {

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

            stage(stage const&) = delete;
            void operator=(stage const&)  = delete;

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

            /**
             * Dequeue runnable threads into the calling cpu's runqueue.
             *
             * The policy which thread in which stage is dequeued is subject
             * to implementation of this data structure.
             */
            static void dequeue();

        private:
            stage() {}

        public:
            constexpr static int max_stages = 32;

        private: /* per-instance state */
            int _id;
            std::string _name;
    };

}
#endif

#endif /*_STAGESCHED_H */
