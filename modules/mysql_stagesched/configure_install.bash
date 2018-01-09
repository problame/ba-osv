#!/bin/bash
#
# Deploy my.cnf and initialize the MySQL data directory.
#
# Initialization can either happen via the MySQL-provided script
#   or a pre-initialized data directory from a tarball
#   (useful for OLTP benchmark)

set -ex

if [ "$#" -lt "2" ]; then
	exit 1
fi

BASEDIR="$1"
ROOTDIR="$2"
PRESEED="$3"
if [ ! -z "$PRESEED" ]; then
    PRESEED="$(readlink -f $PRESEED)"
fi

cd "$ROOTDIR"
cp "$BASEDIR/my.cnf" etc/

cd usr

if [ ! -z "$PRESEED" ]; then

    echo "Extracting preseed to data directory: $PRESEED"
    mkdir -p data
    tar -C data -xf "$PRESEED"

else

    ./scripts/mysql_install_db --basedir=./ --datadir=data

fi

