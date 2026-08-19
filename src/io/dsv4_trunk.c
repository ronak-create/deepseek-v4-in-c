/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk.c - see dsv4_trunk.h. */

/* O_DIRECT is a GNU extension, so the feature macro has to precede EVERY libc
 * header. Defining it after an include silently leaves O_DIRECT undeclared and
 * the fallback path becomes the only path -- a 15x slowdown that compiles. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "dsv4_portable_io.h"   /* then: sets _DARWIN_C_SOURCE before libc */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "dsv4_trunk.h"
#include "dsv4_st.h"
#include "json.h"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* The manifest stores dtypes as the safetensors spelling, so the mapping is the
 * reader's, not a second table that could drift from it. */
static int dtype_from_name(const char *s)
{
    if (!strcmp(s, "BF16"))    return DSV4_DT_BF16;
    if (!strcmp(s, "F32"))     return DSV4_DT_F32;
    if (!strcmp(s, "F16"))     return DSV4_DT_F16;
    if (!strcmp(s, "U8"))      return DSV4_DT_U8;
    if (!strcmp(s, "I8"))      return DSV4_DT_I8;
    if (!strcmp(s, "I64"))     return DSV4_DT_I64;
    if (!strcmp(s, "F8_E4M3")) return DSV4_DT_F8_E4M3;
    if (!strcmp(s, "F8_E8M0")) return DSV4_DT_F8_E8M0;
    return DSV4_DT_UNKNOWN;
}

static char *slurp(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) { fclose(f); return NULL; }
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0; fclose(f);
    if (len) *len = n;
    return s;
}

static int parse_manifest(DSV4Trunk *tr, const char *dir)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.json", dir);
    tr->json_text = slurp(p, NULL);
    if (!tr->json_text) return -1;

    jval *root = json_parse(tr->json_text, &tr->json_arena);
    if (!root) { fprintf(stderr, "%s: not valid JSON\n", p); return -1; }

    jval *nl = json_get(root, "n_layers");
    jval *ls = json_get(root, "layers");
    if (!nl || !ls || ls->t != J_ARR) {
        fprintf(stderr, "%s: missing n_layers or layers\n", p);
        return -1;
    }
    tr->n_layers = (int)nl->num;
    tr->lay = (DSV4TrunkLayer *)calloc((size_t)tr->n_layers, sizeof(DSV4TrunkLayer));
    if (!tr->lay) return -1;

    for (int L = 0; L < tr->n_layers && L < ls->len; L++) {
        jval *o = ls->kids[L];
        if (o->t != J_OBJ) continue;                 /* null = layer absent */
        DSV4TrunkLayer *lay = &tr->lay[L];
        lay->file_off = (int64_t)json_get(o, "file_off")->num;
        lay->nbytes   = (int64_t)json_get(o, "nbytes")->num;
        jval *ts = json_get(o, "tensors");
        lay->nt = ts->len;
        lay->t = (DSV4TrunkTensor *)calloc((size_t)lay->nt, sizeof(DSV4TrunkTensor));
        for (int i = 0; i < lay->nt; i++) {
            jval *e = ts->kids[i];
            lay->t[i].name   = json_get(e, "name")->str;   /* borrowed from arena */
            lay->t[i].off    = (int64_t)json_get(e, "off")->num;
            lay->t[i].nbytes = (int64_t)json_get(e, "nbytes")->num;
            lay->t[i].dtype  = dtype_from_name(json_get(e, "dtype")->str);
            if (lay->t[i].dtype == DSV4_DT_UNKNOWN) {
                fprintf(stderr, "dsv4_trunk: layer %d tensor %s has dtype '%s', "
                                "which this build does not implement\n",
                        L, lay->t[i].name, json_get(e, "dtype")->str);
                return -1;
            }
        }
    }
    return 0;
}

int dsv4_trunk_open(DSV4Trunk *tr, const char *dir, const DSV4Cfg *c,
                    int64_t budget_bytes)
{
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
    if (parse_manifest(tr, dir) != 0) { dsv4_trunk_close(tr); return -1; }

    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    tr->fd = open(p, O_RDONLY | O_DIRECT);
    tr->direct = 1;
    if (tr->fd < 0) {
        /* O_DIRECT is unavailable on some filesystems -- notably /mnt/c under
         * WSL, where it works but at 295 MB/s against 4.4 GB/s on ext4. Falling
         * back keeps the engine CORRECT there; it does not make it fast. */
        tr->fd = open(p, O_RDONLY);
        tr->direct = 0;
        if (tr->fd < 0) { perror(p); dsv4_trunk_close(tr); return -1; }
        fprintf(stderr, "dsv4_trunk: O_DIRECT unavailable on %s, using buffered "
                        "reads (expect a large slowdown)\n", p);
    }
    dsv4_set_direct(tr->fd);

    /* Widest layer decides the slot size, because ring slots are uniform. */
    int64_t widest = 0;
    for (int L = 0; L < tr->n_layers; L++)
        if (tr->lay[L].nbytes > widest) widest = tr->lay[L].nbytes;
    /* Room for an O_DIRECT read widened out to the enclosing aligned window. */
    tr->slot_bytes = (widest + 2 * DSV4_TRUNK_ALIGN + DSV4_TRUNK_ALIGN - 1)
                   / DSV4_TRUNK_ALIGN * DSV4_TRUNK_ALIGN;

    tr->widen_cap = dsv4_bind_widen_bytes(c);
    if (posix_memalign((void **)&tr->widen, DSV4_TRUNK_ALIGN,
                       tr->widen_cap ? tr->widen_cap : 1) != 0) {
        dsv4_trunk_close(tr); return -1;
    }

    /* Spend the budget on pinned layers first, keeping at least one ring slot.
     * A pinned layer gets an EXACT allocation: sizing every pin like the widest
     * layer would waste the difference between a 165 MB indexed layer and a
     * 131 MB dense one on every single pin. */
    int64_t left = budget_bytes - tr->slot_bytes - (int64_t)tr->widen_cap;
    int npin = 0;
    while (npin < tr->n_layers && tr->lay[npin].nbytes > 0
           && left >= tr->lay[npin].nbytes) {
        left -= tr->lay[npin].nbytes;
        npin++;
    }
    tr->npin = npin;
    tr->nslot = 1 + (int)(left / tr->slot_bytes);
    if (tr->nslot < 1) tr->nslot = 1;
    /* Never more slots than there are unpinned layers to put in them.
     *
     * Without this a generous budget asks for thousands of slots -- a probe with
     * a 1 TB budget computed ~6000 x 165 MB and the allocation simply failed,
     * so an over-provisioned machine could not open the trunk at all. Once every
     * layer is pinned one slot is enough, and it is still allocated because
     * dsv4_trunk_bind uses the ring for any layer outside the pinned prefix. */
    {
        const int tail = tr->n_layers - tr->npin;
        const int cap = tail > 0 ? tail : 1;
        if (tr->nslot > cap) tr->nslot = cap;
    }

    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(void *));
    tr->layer_of = (int *)malloc((size_t)tr->nslot * sizeof(int));
    tr->slot_of  = (int32_t *)malloc((size_t)tr->n_layers * sizeof(int32_t));
    if (!tr->pin || !tr->layer_of || !tr->slot_of) { dsv4_trunk_close(tr); return -1; }
    for (int i = 0; i < tr->nslot; i++) tr->layer_of[i] = -1;
    for (int i = 0; i < tr->n_layers; i++) tr->slot_of[i] = -1;

    if (posix_memalign((void **)&tr->ring, DSV4_TRUNK_ALIGN,
                       (size_t)tr->nslot * (size_t)tr->slot_bytes) != 0) {
        dsv4_trunk_close(tr); return -1;
    }
    return 0;
}

void dsv4_trunk_close(DSV4Trunk *tr)
{
    if (!tr) return;
    if (tr->fd >= 0) close(tr->fd);
    for (int i = 0; i < tr->npin; i++) if (tr->pin) free(tr->pin[i]);
    free(tr->pin); free(tr->ring); free(tr->widen);
    free(tr->layer_of); free(tr->slot_of);
    if (tr->lay) for (int L = 0; L < tr->n_layers; L++) free(tr->lay[L].t);
    free(tr->lay); free(tr->json_arena); free(tr->json_text);
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;
}

/* Read one layer's run into `dst`, which must hold slot_bytes.
 *
 * O_DIRECT requires the offset, the length AND the buffer to be aligned. The
 * packer aligns every run's START, so only the length needs widening here. */
static int read_run(DSV4Trunk *tr, int L, unsigned char *dst)
{
    const int64_t off = tr->lay[L].file_off;
    int64_t want = tr->lay[L].nbytes;
    if (tr->direct)
        want = (want + DSV4_TRUNK_ALIGN - 1) / DSV4_TRUNK_ALIGN * DSV4_TRUNK_ALIGN;

    const double t0 = now_s();
    int64_t got = 0;
    while (got < want) {
        const ssize_t r = pread(tr->fd, dst + got, (size_t)(want - got),
                                (off_t)(off + got));
        if (r <= 0) {
            /* A short read at the very end is expected when the file's last run
             * is not a whole number of alignment units; the packer pads for
             * exactly this, so anything else is a real failure. */
            if (r == 0 && got >= tr->lay[L].nbytes) break;
            fprintf(stderr, "dsv4_trunk: short read on layer %d at %lld\n",
                    L, (long long)(off + got));
            return -1;
        }
        got += r;
    }
    tr->load_seconds += now_s() - t0;
    tr->bytes_read += (uint64_t)got;
    return 0;
}

/* Resolver handed to dsv4_bind_layer_mem: linear over one layer's ~40 tensors,
 * which is faster than building a hash for a list this short and is called once
 * per layer per token rather than per lookup. */
typedef struct { const DSV4TrunkLayer *lay; } FindCtx;

static int trunk_find(void *ctx, const char *name,
                      int64_t *off, int64_t *nbytes, int *dtype)
{
    const DSV4TrunkLayer *lay = ((FindCtx *)ctx)->lay;
    for (int i = 0; i < lay->nt; i++)
        if (!strcmp(lay->t[i].name, name)) {
            *off = lay->t[i].off; *nbytes = lay->t[i].nbytes;
            *dtype = lay->t[i].dtype;
            return 0;
        }
    return -1;
}

int dsv4_trunk_bind(DSV4Trunk *tr, const DSV4Cfg *c, int L, DSV4LayerBind *b)
{
    if (L < 0 || L >= tr->n_layers || tr->lay[L].nt == 0) {
        fprintf(stderr, "dsv4_trunk: layer %d is not in the trunk\n", L);
        return -1;
    }

    unsigned char *run;
    if (L < tr->npin) {
        if (!tr->pin[L]) {
            if (posix_memalign((void **)&tr->pin[L], DSV4_TRUNK_ALIGN,
                               (size_t)tr->lay[L].nbytes + 2 * DSV4_TRUNK_ALIGN) != 0)
                return -1;
            if (read_run(tr, L, tr->pin[L]) != 0) return -1;
            tr->misses++;
        } else {
            tr->hits++;
        }
        run = tr->pin[L];
    } else if (tr->slot_of[L] >= 0) {
        run = tr->ring + (size_t)tr->slot_of[L] * (size_t)tr->slot_bytes;
        tr->hits++;
    } else {
        /* Round-robin, not LRU. See the header: a cyclic scan defeats LRU
         * entirely, and round-robin over the unpinned tail is what makes the
         * hit rate exactly npin/n_layers rather than zero. */
        const int slot = tr->next_slot;
        tr->next_slot = (tr->next_slot + 1) % tr->nslot;
        if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;
        run = tr->ring + (size_t)slot * (size_t)tr->slot_bytes;
        if (read_run(tr, L, run) != 0) return -1;
        tr->layer_of[slot] = L;
        tr->slot_of[L] = slot;
        tr->misses++;
    }

    FindCtx fc = { &tr->lay[L] };
    DSV4MemSrc src = { trunk_find, &fc };
    return dsv4_bind_layer_mem(c, L, b, run, &src, tr->widen, tr->widen_cap, NULL);
}

void dsv4_trunk_report(const DSV4Trunk *tr, const char *label)
{
    const uint64_t n = tr->hits + tr->misses;
    printf("trunk %s: %d layers, %d pinned + %d ring slot(s) of %.2f MB\n",
           label, tr->n_layers, tr->npin, tr->nslot,
           (double)tr->slot_bytes / 1048576.0);
    printf("  %llu binds, %llu hits (%.1f%%), %llu reads\n",
           (unsigned long long)n, (unsigned long long)tr->hits,
           n ? 100.0 * (double)tr->hits / (double)n : 0.0,
           (unsigned long long)tr->misses);
    printf("  %.2f GB read in %.2f s = %.2f GB/s%s\n",
           (double)tr->bytes_read / 1073741824.0, tr->load_seconds,
           tr->load_seconds > 0
               ? (double)tr->bytes_read / tr->load_seconds / 1e9 : 0.0,
           tr->direct ? " (O_DIRECT)" : " (buffered)");
}
