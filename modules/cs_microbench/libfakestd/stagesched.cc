/*
 * Copyright (C) 2017 Christian Schwarz
 * This work is open source software, licensed under the terms of the
 * BSD license as described in the LICENSE file in the top-level directory.
 */

#include <osv/stagesched.h>
#include <mutex>
#include <iostream>
#include <thread>

using namespace sched;

stage* stage::define(const std::string name) {

    static std::mutex _stages_mtx;
    static stage stages[stage::max_stages];
    static int stages_next;

    std::lock_guard<std::mutex> guard(_stages_mtx);

    if (stages_next == stage::max_stages)
        return nullptr;

    auto& next = stages[stages_next];
    next._id = stages_next;
    stages_next++;
    next._name = name;

    return &next;
}

void stage::enqueue() {}
