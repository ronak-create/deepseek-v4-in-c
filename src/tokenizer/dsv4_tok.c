/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok.c - see dsv4_tok.h for the pre-tokenizer's shape and its traps. */
#include "dsv4_tok.h"

/* ------------------------------------------------------------------ load --- */

int dsv4_tok_load(DSV4Tok *t, const char *path)
{
    memset(t, 0, sizeof *t);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *raw = (char *)malloc((size_t)n);
    if (!raw || fread(raw, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(raw); return -1;
    }
    fclose(f);
    t->raw = raw;

    if (memcmp(raw, DSV4_TOK_MAGIC, 7) != 0) {
        fprintf(stderr, "%s: not a packed DeepSeek-V4 tokenizer\n", path);
        return -1;
    }
    const char *p = raw + 8;
    uint32_t ver;
    memcpy(&ver, p, 4);          p += 4;
    memcpy(&t->n_vocab, p, 4);   p += 4;
    memcpy(&t->n_merge, p, 4);   p += 4;
    memcpy(&t->n_added, p, 4);   p += 4;
    memcpy(&t->blob_bytes, p, 8); p += 8;
    if (ver != 1) { fprintf(stderr, "%s: version %u, expected 1\n", path, ver);
                    return -1; }

    t->voff  = (uint32_t *)p; p += (size_t)t->n_vocab * 4;
    t->vlen  = (uint16_t *)p; p += (size_t)t->n_vocab * 2;
    t->ma    = (uint32_t *)p; p += (size_t)t->n_merge * 4;
    t->mb    = (uint32_t *)p; p += (size_t)t->n_merge * 4;
    t->mo    = (uint32_t *)p; p += (size_t)t->n_merge * 4;
    t->added = (uint32_t *)p; p += (size_t)t->n_added * 4;
    t->blob  = (char *)p;

    /* Bucket merges by their LEFT id. Without this the merge loop scans all
     * 127,741 pairs for every adjacent pair at every step, which is quadratic
     * in a way that shows up immediately on real text. */
    t->pair_head = (int32_t *)malloc((size_t)t->n_vocab * sizeof(int32_t));
    t->pair_next = (int32_t *)malloc((size_t)t->n_merge * sizeof(int32_t));
    if (!t->pair_head || !t->pair_next) return -1;
    for (uint32_t i = 0; i < t->n_vocab; i++) t->pair_head[i] = -1;
    /* Walk backwards so each bucket ends up in ascending rank order, and the
     * first match found is therefore the best one. */
    for (int32_t i = (int32_t)t->n_merge - 1; i >= 0; i--) {
        t->pair_next[i] = t->pair_head[t->ma[i]];
        t->pair_head[t->ma[i]] = i;
    }
    return 0;
}

void dsv4_tok_free(DSV4Tok *t)
{
    if (!t) return;
    free(t->pair_head); free(t->pair_next); free(t->raw);
    memset(t, 0, sizeof *t);
}

/* ------------------------------------------------------------ lookup ------ */

/* Linear-probed hash over the vocab blob, built lazily on first use. */
static int32_t *g_bucket; static uint32_t g_nbucket; static const DSV4Tok *g_owner;

static uint32_t hash_bytes(const char *s, int n)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static void build_index(const DSV4Tok *t)
{
    g_nbucket = 1; while (g_nbucket < t->n_vocab * 2) g_nbucket <<= 1;
    g_bucket = (int32_t *)malloc((size_t)g_nbucket * sizeof(int32_t));
    for (uint32_t i = 0; i < g_nbucket; i++) g_bucket[i] = -1;
    for (uint32_t i = 0; i < t->n_vocab; i++) {
        if (t->vlen[i] == 0) continue;
        uint32_t h = hash_bytes(t->blob + t->voff[i], t->vlen[i]) & (g_nbucket - 1);
        while (g_bucket[h] >= 0) h = (h + 1) & (g_nbucket - 1);
        g_bucket[h] = (int32_t)i;
    }
    g_owner = t;
}

int32_t dsv4_tok_find(const DSV4Tok *t, const char *s, int n)
{
    if (g_owner != t) build_index(t);
    uint32_t h = hash_bytes(s, n) & (g_nbucket - 1);
    while (g_bucket[h] >= 0) {
        const int32_t id = g_bucket[h];
        if (t->vlen[id] == n && memcmp(t->blob + t->voff[id], s, (size_t)n) == 0)
            return id;
        h = (h + 1) & (g_nbucket - 1);
    }
    return -1;
}

/* ------------------------------------------------- stage 2 alternation ---- */

/* Length in CODEPOINTS consumed at `i`, and the byte length via *blen.
 * Alternatives are tried in order and the first that matches wins -- see the
 * header: reordering them silently moves every leading space. */
static int match_stage2(const char *s, int n, int i, int *blen)
{
    int j = i, cp_i;
    uint32_t c;

    /* a)  [ASCII punct][A-Za-z]+ */
    cp_i = j; c = dsv4_utf8_next(s, n, &cp_i);
    if (dsv4_is_ascii_punct(c)) {
        int k = cp_i, cnt = 0, last = cp_i;
        while (k < n) {
            int t2 = k; uint32_t d = dsv4_utf8_next(s, n, &t2);
            if (!dsv4_is_alpha(d)) break;
            cnt++; k = t2; last = t2;
        }
        if (cnt > 0) { *blen = last - i; return 1; }
    }

    /* b)  [^\r\n\p{L}\p{P}\p{S}]? [\p{L}\p{M}]+ */
    {
        int k = i;
        int t2 = k; uint32_t d = dsv4_utf8_next(s, n, &t2);
        int opt_end = k;
        if (k < n && d != '\r' && d != '\n' && !dsv4_is_L(d)
            && !dsv4_is_P(d) && !dsv4_is_S(d)) {
            opt_end = t2;                     /* the optional leading char */
        }
        int m = opt_end, cnt = 0, last = opt_end;
        while (m < n) {
            int t3 = m; uint32_t e = dsv4_utf8_next(s, n, &t3);
            if (!dsv4_is_L(e) && !dsv4_is_M(e)) break;
            cnt++; m = t3; last = t3;
        }
        if (cnt > 0) { *blen = last - i; return 1; }
    }

    /* c)   ?[\p{P}\p{S}]+[\r\n]* */
    {
        int k = i;
        int t2 = k; uint32_t d = dsv4_utf8_next(s, n, &t2);
        int after_sp = k;
        if (d == ' ') { after_sp = t2; }
        int m = after_sp, cnt = 0, last = after_sp;
        while (m < n) {
            int t3 = m; uint32_t e = dsv4_utf8_next(s, n, &t3);
            if (!dsv4_is_P(e) && !dsv4_is_S(e)) break;
            cnt++; m = t3; last = t3;
        }
        if (cnt > 0) {
            while (last < n) { int t3 = last; uint32_t e = dsv4_utf8_next(s, n, &t3);
                               if (e != '\r' && e != '\n') break; last = t3; }
            *blen = last - i; return 1;
        }
    }

    /* d)  \s*[\r\n]+ */
    {
        int k = i, last_ws = i;
        while (k < n) {
            int t2 = k; uint32_t e = dsv4_utf8_next(s, n, &t2);
            if (!dsv4_is_space(e)) break;
            k = t2;
        }
        /* back off to the first \r\n inside that whitespace run */
        int m = i, nl_start = -1;
        while (m < k) { int t2 = m; uint32_t e = dsv4_utf8_next(s, n, &t2);
                        if (e == '\r' || e == '\n') { nl_start = m; break; }
                        m = t2; }
        if (nl_start >= 0) {
            int e2 = nl_start;
            while (e2 < n) { int t2 = e2; uint32_t e = dsv4_utf8_next(s, n, &t2);
                             if (e != '\r' && e != '\n') break; e2 = t2; }
            *blen = e2 - i; return 1;
        }
        (void)last_ws;
    }

    /* e) \s+(?!\S)   and   f) \s+  -- both consume the run; the lookahead only
     * distinguishes them when non-space follows, and in that case e) gives back
     * its last character so f) can take it with what comes next. */
    {
        int k = i, cnt = 0, last = i, prev = i;
        while (k < n) {
            int t2 = k; uint32_t e = dsv4_utf8_next(s, n, &t2);
            if (!dsv4_is_space(e)) break;
            prev = k; cnt++; k = t2; last = t2;
        }
        if (cnt > 0) {
            if (last < n && cnt > 1) { *blen = prev - i; return 1; } /* (?!\S) */
            *blen = last - i; return 1;
        }
    }

    *blen = 0;
    return 0;
}

/* ---------------------------------------------------------- pre-tokenize --- */

typedef struct { int off, len; } Piece;

/* Stage 0: isolate runs of 1..3 digits. Stage 1: isolate CJK runs. */
static int split_isolated(const char *s, const Piece *in, int nin,
                          Piece *out, int max, int mode)
{
    int nout = 0;
    for (int p = 0; p < nin; p++) {
        const int base = in[p].off, end = base + in[p].len;
        int i = base, run_start = base;
        while (i < end) {
            int t2 = i; uint32_t c = dsv4_utf8_next(s, end, &t2);
            const int hit = (mode == 0) ? dsv4_is_N(c) : dsv4_is_cjk(c);
            if (!hit) { i = t2; continue; }

            if (i > run_start && nout < max)
                out[nout++] = (Piece){ run_start, i - run_start };

            int k = i, taken = 0;
            while (k < end) {
                int t3 = k; uint32_t d = dsv4_utf8_next(s, end, &t3);
                const int h2 = (mode == 0) ? dsv4_is_N(d) : dsv4_is_cjk(d);
                if (!h2) break;
                /* \p{N}{1,3} is BOUNDED at three; the CJK class is not. */
                if (mode == 0 && taken == 3) break;
                taken++; k = t3;
            }
            if (nout < max) out[nout++] = (Piece){ i, k - i };
            i = k; run_start = k;
        }
        if (end > run_start && nout < max)
            out[nout++] = (Piece){ run_start, end - run_start };
    }
    return nout;
}

static int split_stage2(const char *s, const Piece *in, int nin,
                        Piece *out, int max)
{
    int nout = 0;
    for (int p = 0; p < nin; p++) {
        const int base = in[p].off, end = base + in[p].len;
        int i = base, gap = base;      /* gap..i is unmatched text so far */
        while (i < end) {
            int blen = 0;
            if (match_stage2(s, end, i, &blen) && blen > 0) {
                /* Flush the unmatched run as ONE piece before the match.
                 *
                 * This is the whole point. A Split with Isolated behaviour
                 * emits its matches as pieces and LEAVES everything else
                 * intact; it does not cut the gaps up. An earlier version
                 * emitted one codepoint per unmatched character, which meant a
                 * digit run reached the BPE already shattered: "12" arrived as
                 * "1" then "2" and could never merge into 736. Twelve of the
                 * fourteen parity cases still passed, because only digits fall
                 * through every alternative of stage 2. */
                if (i > gap && nout < max) out[nout++] = (Piece){ gap, i - gap };
                if (nout < max) out[nout++] = (Piece){ i, blen };
                i += blen;
                gap = i;
            } else {
                int t2 = i;
                dsv4_utf8_next(s, end, &t2);
                i = t2;                 /* extend the unmatched run */
            }
        }
        if (end > gap && nout < max) out[nout++] = (Piece){ gap, end - gap };
    }
    return nout;
}

/* --------------------------------------------------------------- encode ---- */

#define MAXSYM 4096

int dsv4_tok_encode(const DSV4Tok *t, const char *text, int n,
                    int32_t *ids, int max_ids)
{
    if (n <= 0) return 0;

    static Piece a[65536], b[65536], c[65536];
    Piece whole = { 0, n };
    int na = split_isolated(text, &whole, 1, a, 65536, 0);
    int nb = split_isolated(text, a, na, b, 65536, 1);
    int nc = split_stage2(text, b, nb, c, 65536);

    int nids = 0;
    for (int p = 0; p < nc; p++) {
        /* byte-level: every byte becomes a printable codepoint */
        static char enc[MAXSYM * 4];
        static int  sym_off[MAXSYM], sym_len[MAXSYM];
        static int32_t sym[MAXSYM];
        int ns = 0, w = 0;
        for (int i = 0; i < c[p].len && ns < MAXSYM; i++) {
            const unsigned char by = (unsigned char)text[c[p].off + i];
            sym_off[ns] = w;
            w += dsv4_cp_to_utf8(dsv4_byte_to_cp(by), enc + w);
            sym_len[ns] = w - sym_off[ns];
            sym[ns] = dsv4_tok_find(t, enc + sym_off[ns], sym_len[ns]);
            ns++;
        }

        /* Merge the best-ranked adjacent pair until none applies. Buckets are
         * in ascending rank order, so the first hit in a bucket is that pair's
         * rank and the scan stops there. */
        for (;;) {
            int best = -1, best_at = -1; uint32_t best_out = 0;
            for (int i = 0; i + 1 < ns; i++) {
                if (sym[i] < 0 || sym[i + 1] < 0) continue;
                for (int32_t m = t->pair_head[sym[i]]; m >= 0; m = t->pair_next[m]) {
                    if (t->mb[m] == (uint32_t)sym[i + 1]) {
                        if (best < 0 || m < best) { best = m; best_at = i;
                                                    best_out = t->mo[m]; }
                        break;
                    }
                }
            }
            if (best < 0) break;
            sym[best_at] = (int32_t)best_out;
            for (int i = best_at + 1; i + 1 < ns; i++) sym[i] = sym[i + 1];
            ns--;
        }

        for (int i = 0; i < ns && nids < max_ids; i++)
            if (sym[i] >= 0) ids[nids++] = sym[i];
    }
    return nids;
}
