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

all: $(BIN)/test_cfg $(BIN)/test_st

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

test: $(BIN)/test_cfg $(BIN)/test_st
	@./$(BIN)/test_cfg
	@echo
	@./$(BIN)/test_st

clean:
	rm -rf $(BUILD) $(BIN)

help:
	@echo "  make test          the config gate"
	@echo "  make cfg-fixtures  regenerate bad-config fixtures from the released config"
	@echo "  make clean"
