# SPDX-License-Identifier: Apache-2.0
# deepseek-v4-in-c

CC      ?= cc
CFLAGS  ?= -O3 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla \
           -Wno-unused-parameter -march=native -fopenmp -pthread -ffp-contract=off
# -ffp-contract=off is not an optimisation choice. The scalar, OpenMP and AVX2
# paths must produce bit-identical results, so that a performance change can
# never quietly become an accuracy change. Fusing a multiply and an add changes
# the rounding, so it stays off.
INCS    := -Iinclude -Iinclude/dsv4 -Ithird_party -Isrc/core -Isrc/io \
           -Isrc/cache -Isrc/model -Isrc/tokenizer
LDFLAGS ?= -lm -fopenmp -pthread
LDLIBS_EXTRA = $(CUDA_LD)

BUILD := build
BIN   := bin

.PHONY: all test cfg-fixtures st-fixtures ref-fixtures tok-fixtures fixtures clean help

# EVERY binary depends on EVERY engine source and header.
#
# The per-target rules below list only the sources each one compiles, which is
# accurate but not sufficient: editing src/cache/dsv4_cache.c did NOT rebuild
# bin/test_cache, so a gate silently kept testing the previous build and
# reported a failure that had already been instrumented away. Twice.
#
# Rebuilding twenty small binaries on any header change costs a few seconds and
# removes a whole class of phantom results, so it is not a trade worth tuning.
ENGINE := $(wildcard src/core/*.c src/core/*.h src/io/*.c src/io/*.h \
                     src/cache/*.c src/cache/*.h src/model/*.c src/model/*.h \
                     src/tokenizer/*.c src/tokenizer/*.h \
                     include/dsv4/*.h third_party/*.h)
ALLBIN := $(BIN)/test_cuda $(BIN)/test_pro $(BIN)/dsv4 $(BIN)/test_cfg $(BIN)/test_st $(BIN)/test_bind $(BIN)/test_quant $(BIN)/test_matmul $(BIN)/test_ops $(BIN)/test_hc $(BIN)/test_rope $(BIN)/test_attn $(BIN)/test_compress $(BIN)/test_indexer $(BIN)/test_layer $(BIN)/test_cache $(BIN)/test_bindmem $(BIN)/test_oracle $(BIN)/test_layer_oracle $(BIN)/test_trunk $(BIN)/test_tok $(BIN)/test_model_oracle

all: $(ALLBIN)

$(ALLBIN): $(ENGINE)


$(BIN)/test_cfg: tests/unit/test_cfg.c include/dsv4/dsv4.h include/dsv4/dsv4_cfg.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) $< -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_st: tests/unit/test_st.c src/io/dsv4_st.c src/io/dsv4_st.h \
                src/io/dsv4_portable_io.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_st.c src/io/dsv4_st.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

# Regenerate fixtures. Both are derived from the released checkpoint, so they are
# only regenerated when it changes; the outputs are committed.
cfg-fixtures:
	python3 tools/make_cfg_fixtures.py
st-fixtures:
	python3 tools/make_st_fixture.py tests/fixtures/st
ref-fixtures:
	~/venv-cuda/bin/python tools/emit_fixtures.py tests/fixtures/ref
	~/venv-cuda/bin/python tools/emit_layer_fixture.py tests/fixtures/ref
fixtures: cfg-fixtures st-fixtures

$(BIN)/test_bind: tests/unit/test_bind.c src/model/dsv4_bind.c src/model/dsv4_bind.h                   src/io/dsv4_st.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_bind.c src/model/dsv4_bind.c 	      src/io/dsv4_st.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_quant: tests/unit/test_quant.c src/core/dsv4_quant.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_quant.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_matmul: tests/unit/test_matmul.c src/core/dsv4_matmul.c                     src/core/dsv4_quant.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_matmul.c src/core/dsv4_matmul.c 	      -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_ops: tests/unit/test_ops.c src/core/dsv4_ops.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_ops.c src/core/dsv4_ops.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_hc: tests/unit/test_hc.c src/core/dsv4_hc.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_hc.c src/core/dsv4_hc.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_rope: tests/unit/test_rope.c src/core/dsv4_rope.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_rope.c src/core/dsv4_rope.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_attn: tests/unit/test_attn.c src/core/dsv4_attn.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_attn.c src/core/dsv4_attn.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_compress: tests/unit/test_compress.c src/core/dsv4_compress.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_compress.c src/core/dsv4_compress.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_indexer: tests/unit/test_indexer.c src/core/dsv4_indexer.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_indexer.c src/core/dsv4_indexer.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

DSV4_CORE := src/core/dsv4_ops.c src/core/dsv4_matmul.c src/core/dsv4_hc.c              src/core/dsv4_rope.c src/core/dsv4_attn.c              src/core/dsv4_compress.c src/core/dsv4_indexer.c

$(BIN)/test_layer: tests/unit/test_layer.c src/model/dsv4_layer.c $(DSV4_CORE)                    src/model/dsv4_layer.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_layer.c src/model/dsv4_layer.c 	      $(DSV4_CORE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_cache: tests/unit/test_cache.c src/cache/dsv4_cache.c src/io/dsv4_st.c                    src/cache/dsv4_cache.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) -Isrc/cache tests/unit/test_cache.c 	      src/cache/dsv4_cache.c src/io/dsv4_st.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_bindmem: tests/unit/test_bindmem.c src/model/dsv4_bind.c src/io/dsv4_st.c                      src/model/dsv4_bind.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_bindmem.c src/model/dsv4_bind.c 	      src/io/dsv4_st.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_oracle: tests/unit/test_oracle.c $(DSV4_CORE)                     include/dsv4/dsv4.h third_party/json.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_oracle.c $(DSV4_CORE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_layer_oracle: tests/unit/test_layer_oracle.c src/model/dsv4_layer.c                           $(DSV4_CORE) src/model/dsv4_layer.h third_party/json.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_layer_oracle.c src/model/dsv4_layer.c 	      $(DSV4_CORE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_trunk: tests/unit/test_trunk.c src/io/dsv4_trunk.c src/io/dsv4_st.c                    src/model/dsv4_bind.c src/io/dsv4_trunk.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_trunk.c src/io/dsv4_trunk.c 	      src/io/dsv4_st.c src/model/dsv4_bind.c -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_tok: tests/unit/test_tok.c src/tokenizer/dsv4_tok.c                  src/tokenizer/dsv4_tok.h third_party/dsv4_unicode.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_tok.c src/tokenizer/dsv4_tok.c 	      -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

tok-fixtures:
	python3 tools/pack_tokenizer.py $(DSV4_MODEL) tests/fixtures/tok/tokenizer.bin
	~/venv-cuda/bin/python tools/emit_tok_fixture.py $(DSV4_MODEL) tests/fixtures/tok

# ---------------------------------------------------------------- CUDA ------
# Optional and auto-detected. A build without nvcc links the stub, where
# dsv4_cuda_available() returns 0 -- which every caller must already handle for
# a machine that has no device, so a CPU-only build is complete, not degraded.
#
# sm_120 is the RTX 5060's Blackwell architecture and needs CUDA 12.8+. The
# real cc is passed through so the host half of the .cu sees the same compiler
# the rest of the engine is built with.
NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
  DSV4_CUDA_SRC := src/cuda/dsv4_cuda_stub.c
  DSV4_CUDA_OBJ :=
  CUDA_LD :=
else
  DSV4_CUDA_SRC :=
  DSV4_CUDA_OBJ := $(BUILD)/dsv4_cuda.o
  CUDA_ARCH ?= sm_120
  CUDA_LD := -L$(dir $(shell readlink -f $(NVCC)))../lib64 -lcudart
  CFLAGS += -DDSV4_HAVE_CUDA
endif

$(BUILD)/dsv4_cuda.o: src/cuda/dsv4_cuda.cu include/dsv4/dsv4_cuda.h
	@mkdir -p $(BUILD)
	$(NVCC) -O3 -arch=$(CUDA_ARCH) $(INCS) -c $< -o $@

DSV4_ENGINE := src/io/dsv4_st.c src/io/dsv4_trunk.c src/cache/dsv4_cache.c                src/model/dsv4_bind.c src/model/dsv4_layer.c                src/tokenizer/dsv4_tok.c $(DSV4_CORE) $(DSV4_CUDA_SRC) $(DSV4_CUDA_OBJ)

$(BIN)/dsv4: src/cli/dsv4_run.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) src/cli/dsv4_run.c $(DSV4_ENGINE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_cuda: tests/unit/test_cuda.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_cuda.c $(DSV4_ENGINE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_pro: tests/unit/test_pro.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_pro.c $(DSV4_ENGINE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

$(BIN)/test_model_oracle: tests/unit/test_model_oracle.c $(DSV4_ENGINE)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_model_oracle.c $(DSV4_ENGINE) -o $@ $(LDFLAGS) $(LDLIBS_EXTRA)

tiny-fixture:
	~/venv-cuda/bin/python tools/make_tiny_checkpoint.py $(HOME)/dsv4-tiny
	python3 tools/pack_trunk.py $(HOME)/dsv4-tiny $(HOME)/dsv4-tiny-trunk

test: $(BIN)/test_cuda $(BIN)/test_pro $(BIN)/test_cfg $(BIN)/test_st $(BIN)/test_bind $(BIN)/test_quant $(BIN)/test_matmul $(BIN)/test_ops $(BIN)/test_hc $(BIN)/test_rope $(BIN)/test_attn $(BIN)/test_compress $(BIN)/test_indexer $(BIN)/test_layer $(BIN)/test_cache $(BIN)/test_bindmem $(BIN)/test_oracle $(BIN)/test_layer_oracle $(BIN)/test_trunk $(BIN)/test_tok $(BIN)/test_model_oracle
	@./$(BIN)/test_cfg
	@echo
	@./$(BIN)/test_st
	@echo
	@./$(BIN)/test_bind
	@echo
	@./$(BIN)/test_quant
	@echo
	@./$(BIN)/test_matmul
	@echo
	@./$(BIN)/test_ops
	@echo
	@./$(BIN)/test_hc
	@echo
	@./$(BIN)/test_rope
	@echo
	@./$(BIN)/test_attn
	@echo
	@./$(BIN)/test_compress
	@echo
	@./$(BIN)/test_indexer
	@echo
	@./$(BIN)/test_layer
	@echo
	@./$(BIN)/test_cache
	@echo
	@./$(BIN)/test_bindmem
	@echo
	@./$(BIN)/test_oracle
	@echo
	@./$(BIN)/test_layer_oracle
	@echo
	@./$(BIN)/test_trunk
	@echo
	@./$(BIN)/test_tok
	@echo
	@DSV4_TINY=$(HOME)/dsv4-tiny DSV4_TINY_TRUNK=$(HOME)/dsv4-tiny-trunk ./$(BIN)/test_model_oracle
	@echo
	@./$(BIN)/test_pro
	@echo
	@./$(BIN)/test_cuda

clean:
	rm -rf $(BUILD) $(BIN)

help:
	@echo "  make test          the config gate"
	@echo "  make cfg-fixtures  regenerate bad-config fixtures from the released config"
	@echo "  make clean"
