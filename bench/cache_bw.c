/* SPDX-License-Identifier: Apache-2.0 */
/* bench/cache_bw.c - expert read throughput: cold/warm, and against queue depth.
 *
 * Reads DISTINCT experts on every repetition so the page cache cannot make the
 * later runs look fast for the wrong reason: a benchmark that re-reads the same
 * 100 experts measures RAM, not the disk, and the real engine's 148 GB working
 * set never fits in RAM.
 *
 * MODE 2, THE QUEUE-DEPTH SWEEP  (cache_bw <model> qd [reads-per-point])
 *   src/cache/dsv4_cache.h records 2.0 GB/s in situ against 4.4 GB/s for the
 *   same drive under a benchmark, at a queue depth of at most six -- because
 *   dsv4_cache_get_many issues one read per top-k expert and top-k is 6. The
 *   obvious reading is "we are queue-depth-bound; build io_uring". This sweep
 *   exists to test that BEFORE anyone writes a submission ring, because the
 *   suggestion cost nothing to make and io_uring costs 3-5 days.
 *
 *   It issues expert-sized (12.75 MB) O_DIRECT reads over distinct regions of
 *   the real checkpoint at concurrency 1..32 and reports GB/s at each point.
 *
 *     - If throughput keeps climbing past 6, the queue is the limit, io_uring
 *       is the fix, and striping is a second step that only pays once one drive
 *       is saturated.
 *     - If it plateaus at or below 6, the queue is NOT the limit. io_uring buys
 *       nothing, striping is the only lever, and that is a negative result
 *       worth having for the price of one afternoon.
 *
 *   Note what this does and does not settle. It measures the DRIVE at depth.
 *   If the sweep plateaus early and the engine still reads at 2.0 GB/s, then
 *   the gap is somewhere between the two -- widened alignment windows, the
 *   bounce-buffer copy, or the serial decide/publish passes around the parallel
 *   read -- and that is a different investigation, not more queue.
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4_cache.h"
#include "dsv4_cfg.h"
#include "dsv4_st.h"

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* ------------------------------------------------------- queue-depth sweep --
 * One read == one expert-sized aligned window, exactly what the engine issues.
 * Regions never repeat across the whole sweep, so no point can be helped by a
 * cache the deeper points did not get. */
typedef struct { int shard; int64_t off, nbytes; } Region;

/* One region == the coalesced w1..w3 run of one real expert, which is what the
 * cache's load path reads as a single widened window. Regions are built from
 * NAMED experts rather than from "every tensor over N bytes": an individual
 * expert weight is ~4.2 MB, so a size filter set at the 12.75 MB expert size
 * finds only the trunk's big matrices and measures the wrong thing. That is not
 * hypothetical -- it is what the first version of this sweep did. */
static int collect_regions(const DSV4St *st, const DSV4Cfg *cfg,
                           Region *r, int cap)
{
    int n = 0;
    for (int L = 0; L < cfg->n_layers && n < cap; L++)
        for (int e = 0; e < cfg->n_experts && n < cap; e++) {
            char nm[224];
            int64_t lo = -1, hi = -1, sh = -1;
            static const char *wn[3] = { "w1", "w2", "w3" };
            int ok = 1;
            for (int i = 0; i < 3 && ok; i++) {
                snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.%s.weight",
                         L, e, wn[i]);
                const DSV4Tensor *t = dsv4_st_find(st, nm);
                if (!t) { ok = 0; break; }
                if (sh < 0) sh = t->shard;
                else if (t->shard != sh) { ok = 0; break; }  /* never split */
                if (lo < 0 || t->off < lo) lo = t->off;
                if (t->off + t->nbytes > hi) hi = t->off + t->nbytes;
            }
            if (!ok || lo < 0 || hi <= lo) continue;
            r[n].shard  = (int)sh;
            r[n].off    = lo;
            r[n].nbytes = hi - lo;
            n++;
        }
    return n;
}

static int sweep(const DSV4St *st, const DSV4Cfg *cfg, int per_point)
{
    static const int QD[] = { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 };
    const int NQD = (int)(sizeof QD / sizeof QD[0]);

    /* Threads here are BLOCKED IN pread, not computing, so the useful
     * concurrency is not bounded by cores -- and clamping to 20 would stop
     * the sweep exactly where the question gets interesting. Turn dynamic
     * adjustment off so a num_threads(32) request is actually honoured. */
    omp_set_dynamic(0);
    const int maxth = QD[NQD - 1];
    const int cap   = NQD * per_point + 16;

    Region *reg = (Region *)malloc((size_t)cap * sizeof *reg);
    if (!reg) return 1;
    const int nreg = collect_regions(st, cfg, reg, cap);
    if (nreg < NQD * per_point) {
        printf("  only %d expert regions found; need %d. Lower "
               "reads-per-point and rerun.\n", nreg, NQD * per_point);
        free(reg);
        return 1;
    }
    int64_t expert_bytes = 0;
    for (int i = 0; i < nreg; i++)
        if (reg[i].nbytes > expert_bytes) expert_bytes = reg[i].nbytes;

    /* One aligned staging buffer per thread, sized for the widest window. */
    const int64_t bufcap = expert_bytes + 2 * 4096;
    unsigned char **buf = (unsigned char **)calloc((size_t)maxth, sizeof *buf);
    if (!buf) { free(reg); return 1; }
    for (int i = 0; i < maxth; i++)
        if (posix_memalign((void **)&buf[i], 4096, (size_t)bufcap) != 0) {
            printf("  could not allocate %d x %.1f MB of staging\n",
                   maxth, (double)bufcap / 1048576.0);
            free(buf); free(reg);
            return 1;
        }

    printf("\nqueue-depth sweep: %d reads of %.2f MB per point, O_DIRECT,\n"
           "distinct regions throughout (%d available)\n\n",
           per_point, (double)expert_bytes / 1048576.0, nreg);
    printf("  %3s %10s %10s %12s %10s\n",
           "QD", "GB/s", "ms/read", "MB in", "vs QD1");

    int    next = 0;
    double base = 0.0;

    for (int q = 0; q < NQD; q++) {
        const int qd = QD[q];
        int64_t total = 0;

        const double t0 = now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(qd) reduction(+:total)
        for (int i = 0; i < per_point; i++) {
            const Region *g = &reg[next + i];
            int b = omp_get_thread_num();
            if (b >= maxth) b = 0;
            int64_t poff = 0;
            const int64_t got = dsv4_st_read_aligned(st, g->shard, g->off,
                                                     g->nbytes, buf[b], bufcap,
                                                     &poff);
            if (got > 0) total += got;
        }
        const double dt = now() - t0;
        next += per_point;

        const double gbs = (double)total / 1073741824.0 / dt;
        if (q == 0) base = gbs;
        printf("  %3d %10.2f %10.2f %12.0f %9.2fx\n",
               qd, gbs, dt / per_point * 1e3,
               (double)total / 1048576.0, base > 0.0 ? gbs / base : 0.0);
    }

    printf("\nRead the shape, not the peak. A curve that is flat from QD1 says\n"
           "the drive is already saturated by one request in flight, and no\n"
           "amount of io_uring will change that -- the only lever left is more\n"
           "drives. A curve still climbing at 32 says the opposite.\n");

    for (int i = 0; i < maxth; i++) free(buf[i]);
    free(buf);
    free(reg);
    return 0;
}

int main(int argc, char **argv)
{
    const char *m = argc > 1 ? argv[1] : getenv("DSV4_MODEL");
    if (!m) {
        printf("usage: cache_bw <model_dir> [reps] [experts]\n");
        printf("       cache_bw <model_dir> qd [reads-per-point]\n");
        return 2;
    }

    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    DSV4St st;
    char path[512];
    snprintf(path, sizeof path, "%s/config.json", m);
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, path)) return 1;
    if (dsv4_st_open(&st, m) != 0) return 1;

    if (argc > 2 && !strcmp(argv[2], "qd")) {
        /* Deliberately no DSV4Cache here. The sweep measures the drive
         * through the same O_DIRECT entry point the cache uses, and
         * building a cache only to read one field would emit two
         * warnings about a budget the sweep never uses. */
        const int rc = sweep(&st, &c, argc > 3 ? atoi(argv[3]) : 48);
        dsv4_st_close(&st);
        return rc;
    }

    int reps = argc > 2 ? atoi(argv[2]) : 3;
    int per  = argc > 3 ? atoi(argv[3]) : 60;
    int e0 = 0;
    for (int r = 0; r < reps; r++) {
        DSV4Cache k;
        if (dsv4_cache_init(&k, &st, &c, 13369344LL * 2) != 0) return 1;
        double t0 = now();
        int n = 0;
        for (int L = 0; L < 10; L++)
            for (int e = e0; e < e0 + per / 10; e++) {
                if (!dsv4_cache_get(&k, L, e)) return 1;
                n++;
            }
        double dt = now() - t0;
        double gb = (double)k.bytes_read / 1073741824.0;
        printf("  rep %d (experts %d..%d): %.2f GB in %.2f s = %.2f GB/s, "
               "%.1f ms/expert  [%s]\n", r, e0, e0 + per / 10 - 1, gb, dt,
               gb / dt, dt / n * 1e3, k.direct ? "O_DIRECT" : "buffered");
        e0 += per / 10;                     /* fresh experts next rep */
        dsv4_cache_free(&k);
    }
    dsv4_st_close(&st);
    return 0;
}
