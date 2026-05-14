# PolyBench/C 4.2.1 — OpenMP omnibus Makefile (BOTS-style)
# Build all benchmarks into ./bin with dataset variants.
# Usage:
#   make all DATASET=LARGE_DATASET          # build all with chosen dataset
#   make atax DATASET=MEDIUM_DATASET        # build single benchmark
#   make list                               # list known benchmarks
# Options:
#   CC=gcc|clang    OMP_FLAG=-fopenmp
#   CFLAGS="-O3 ..." DEFS="..." LOCAL_HEADERS=0|1|2
#   DATASET in {MINI_DATASET,SMALL_DATASET,MEDIUM_DATASET,LARGE_DATASET,EXTRALARGE_DATASET}

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

# ---- Toolchain ----
CC        ?= gcc
OMP_FLAG  ?= -fopenmp
CSTD      ?= -std=gnu99
OPT       ?= -O3
WARN      ?= -Wall -Wextra
# Quiet common PolyBench warnings and enable posix_memalign decl:
CFLAGS    ?= $(OPT) $(WARN) $(CSTD) $(OMP_FLAG) -D_POSIX_C_SOURCE=200112L -Wno-unknown-pragmas -Wno-unused-variable -Wno-misleading-indentation
LDFLAGS   ?= $(OMP_FLAG)
LDLIBS    ?= -lm
DEFS      ?= -DPOLYBENCH_TIME

# Dataset macro (compile-time)
DATASET   ?= STANDARD_DATASET
# Valid: MINI_DATASET, SMALL_DATASET, MEDIUM_DATASET, LARGE_DATASET, EXTRALARGE_DATASET

# Copy or symlink polybench.h into each kernel dir (0=off, 1=symlink, 2=copy)
LOCAL_HEADERS ?= 0

# ---- Paths ----
ROOT := $(CURDIR)
UTIL := $(ROOT)/utilities
BIN  := $(ROOT)/bin
$(shell mkdir -p "$(BIN)")

# ---- Bench list (name:path) ----
# BLAS
BLIST += gemm:linear-algebra/blas/gemm
BLIST += gemver:linear-algebra/blas/gemver
BLIST += gesummv:linear-algebra/blas/gesummv
BLIST += symm:linear-algebra/blas/symm
BLIST += syr2k:linear-algebra/blas/syr2k
BLIST += syrk:linear-algebra/blas/syrk
BLIST += trmm:linear-algebra/blas/trmm
# Kernels
BLIST += 2mm:linear-algebra/kernels/2mm
BLIST += 3mm:linear-algebra/kernels/3mm
BLIST += atax:linear-algebra/kernels/atax
BLIST += bicg:linear-algebra/kernels/bicg
BLIST += doitgen:linear-algebra/kernels/doitgen
BLIST += mvt:linear-algebra/kernels/mvt
# Solvers
BLIST += cholesky:linear-algebra/solvers/cholesky
BLIST += durbin:linear-algebra/solvers/durbin
BLIST += gramschmidt:linear-algebra/solvers/gramschmidt
BLIST += lu:linear-algebra/solvers/lu
BLIST += ludcmp:linear-algebra/solvers/ludcmp
BLIST += trisolv:linear-algebra/solvers/trisolv
# Datamining
BLIST += correlation:datamining/correlation
BLIST += covariance:datamining/covariance
# Medley
BLIST += deriche:medley/deriche
BLIST += floyd-warshall:medley/floyd-warshall
BLIST += nussinov:medley/nussinov
# Stencils
BLIST += adi:stencils/adi
BLIST += fdtd-2d:stencils/fdtd-2d
BLIST += heat-3d:stencils/heat-3d
BLIST += jacobi-1d:stencils/jacobi-1d
BLIST += jacobi-2d:stencils/jacobi-2d
BLIST += seidel-2d:stencils/seidel-2d

# ---- Derived lists ----
BENCH_NAMES := $(foreach pair,$(BLIST),$(firstword $(subst :, ,$(pair))))
BENCH_PATHS := $(foreach pair,$(BLIST),$(lastword  $(subst :, ,$(pair))))

# name->dir map
define bench_dir
$(lastword $(subst :, ,$(filter $(1):%,$(BLIST))))
endef

# Dataset suffix lowercased (NO extra "_dataset" here; the value already contains "_DATASET")
dataset_lc = $(shell echo "$(DATASET)" | tr '[:upper:]' '[:lower:]')

# ---- Public targets ----
.PHONY: all list clean veryclean $(BENCH_NAMES)

all: $(addprefix build-,$(BENCH_NAMES))

list:
	@echo "Known benchmarks:"
	@printf "  %s\n" $(BENCH_NAMES)
	@echo
	@echo "DATASET=$(DATASET)"
	@echo "LOCAL_HEADERS=$(LOCAL_HEADERS) (0=off, 1=symlink, 2=copy)"

# single-benchmark convenience: `make atax DATASET=LARGE_DATASET`
$(BENCH_NAMES):
	@$(MAKE) build-$@ DATASET=$(DATASET) LOCAL_HEADERS=$(LOCAL_HEADERS)

# ---- Pattern: build one bench ----
.PHONY: build-% headers-%
build-%: headers-%
	@name="$*"; dir="$(call bench_dir,$*)"; \
	src_kernel="$$dir/$$name.c"; \
	[[ -f "$$src_kernel" ]] || { echo "Missing $$src_kernel"; exit 1; } ; \
	out="$(BIN)/$*_$(dataset_lc)"; \
	echo "==> CC $$out  (DATASET=$(DATASET))"; \
	$(CC) $(CFLAGS) $(DEFS) -D$(DATASET) \
	    -I "$(UTIL)" -I "$$dir" \
	    "$(UTIL)/polybench.c" "$$src_kernel" \
	    -o "$$out" $(LDFLAGS) $(LDLIBS)

# ---- Optional: stage polybench.h locally per dir ----
headers-%:
	@name="$*"; dir="$(call bench_dir,$*)"; \
	if [[ "$(LOCAL_HEADERS)" -eq 1 ]]; then \
	  if [[ ! -e "$$dir/polybench.h" ]]; then \
	    rel=$$(python - <<'PY'\nimport os,sys\nprint(os.path.relpath('utilities/polybench.h', sys.argv[1]))\nPY \
	      "$$dir"); \
	    ln -s "$$rel" "$$dir/polybench.h"; \
	    echo "symlinked $$dir/polybench.h -> $$rel"; \
	  fi; \
	elif [[ "$(LOCAL_HEADERS)" -eq 2 ]]; then \
	  cp -f "$(UTIL)/polybench.h" "$$dir/polybench.h"; \
	  echo "copied $$dir/polybench.h"; \
	fi

clean:
	@find . -name '*.o' -o -name '*.obj' -o -name '*.out' -o -name 'polybench.h' -type l -print -delete || true

veryclean: clean
	@rm -rf "$(BIN)"
	@mkdir -p "$(BIN)"

