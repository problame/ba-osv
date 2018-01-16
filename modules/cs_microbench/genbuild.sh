#!/bin/bash

cd $(dirname $0)

BUILDH="$(cat build.h)"

CUR="#define SCM_VERSION \"$(git describe --tags --dirty)\""

if [ "$BUILDH" != "$CUR" ]; then
    echo "$CUR" > build.h
fi
