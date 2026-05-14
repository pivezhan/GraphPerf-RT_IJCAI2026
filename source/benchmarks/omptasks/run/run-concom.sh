#!/bin/bash
# defaults: concom is size-based (no input files)
DEF_INPUTS=100000   # N; L and M use app defaults (20 and 100000)

# don't modify from here
BASE_DIR=$(dirname $0)/..
source $BASE_DIR/run/run.common

parse_args "$@"
set_values
exec_all_sizes
