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

BUILD := build
BIN   := bin

.PHONY: all test cfg-fixtures st-fixtures fixtures clean help

all: $(BIN)/test_cfg $(BIN)/test_st $(BIN)/test_bind $(BIN)/test_quant $(BIN)/test_matmul $(BIN)/test_ops $(BIN)/test_hc $(BIN)/test_rope $(BIN)/test_attn $(BIN)/test_compress

$(BIN)/test_cfg: tests/unit/test_cfg.c include/dsv4/dsv4.h include/dsv4/dsv4_cfg.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) $< -o $@ $(LDFLAGS)

$(BIN)/test_st: tests/unit/test_st.c src/io/dsv4_st.c src/io/dsv4_st.h \
                src/io/dsv4_portable_io.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_st.c src/io/dsv4_st.c -o $@ $(LDFLAGS)

# Regenerate fixtures. Both are derived from the released checkpoint, so they are
# only regenerated when it changes; the outputs are committed.
cfg-fixtures:
	python3 tools/make_cfg_fixtures.py
st-fixtures:
	python3 tools/make_st_fixture.py tests/fixtures/st
fixtures: cfg-fixtures st-fixtures

$(BIN)/test_bind: tests/unit/test_bind.c src/model/dsv4_bind.c src/model/dsv4_bind.h                   src/io/dsv4_st.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_bind.c src/model/dsv4_bind.c 	      src/io/dsv4_st.c -o $@ $(LDFLAGS)

$(BIN)/test_quant: tests/unit/test_quant.c src/core/dsv4_quant.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_quant.c -o $@ $(LDFLAGS)

$(BIN)/test_matmul: tests/unit/test_matmul.c src/core/dsv4_matmul.c                     src/core/dsv4_quant.h include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_matmul.c src/core/dsv4_matmul.c 	      -o $@ $(LDFLAGS)

$(BIN)/test_ops: tests/unit/test_ops.c src/core/dsv4_ops.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_ops.c src/core/dsv4_ops.c -o $@ $(LDFLAGS)

$(BIN)/test_hc: tests/unit/test_hc.c src/core/dsv4_hc.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_hc.c src/core/dsv4_hc.c -o $@ $(LDFLAGS)

$(BIN)/test_rope: tests/unit/test_rope.c src/core/dsv4_rope.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_rope.c src/core/dsv4_rope.c -o $@ $(LDFLAGS)

$(BIN)/test_attn: tests/unit/test_attn.c src/core/dsv4_attn.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_attn.c src/core/dsv4_attn.c -o $@ $(LDFLAGS)

$(BIN)/test_compress: tests/unit/test_compress.c src/core/dsv4_compress.c include/dsv4/dsv4.h
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCS) tests/unit/test_compress.c src/core/dsv4_compress.c -o $@ $(LDFLAGS)

test: $(BIN)/test_cfg $(BIN)/test_st $(BIN)/test_bind $(BIN)/test_quant $(BIN)/test_matmul $(BIN)/test_ops $(BIN)/test_hc $(BIN)/test_rope $(BIN)/test_attn $(BIN)/test_compress
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

clean:
	rm -rf $(BUILD) $(BIN)

help:
	@echo "  make test          the config gate"
	@echo "  make cfg-fixtures  regenerate bad-config fixtures from the released config"
	@echo "  make clean"
