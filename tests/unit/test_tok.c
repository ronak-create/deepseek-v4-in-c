/* SPDX-License-Identifier: Apache-2.0 */
/* test_tok.c - exact id-stream parity against HuggingFace tokenizers.
 *
 * PARITY IS EXACT OR IT IS NOTHING. There is no tolerance for a tokenizer: one
 * different id is a different prompt, and the model's output diverges from that
 * token onward. So every case must match id for id, and a case that differs
 * prints BOTH streams so the disagreement can be read directly rather than
 * inferred from a count.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "dsv4_tok.h"
#include "json.h"

int     dsv4_tok_load(DSV4Tok *, const char *);
void    dsv4_tok_free(DSV4Tok *);
int     dsv4_tok_encode(const DSV4Tok *, const char *, int, int32_t *, int);

static int fails = 0, cases_ok = 0, cases_run = 0;

static char *slurp(const char *p)
{
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0; fclose(f); return s;
}

static void dump(const char *what, const int32_t *v, int n)
{
    printf("          %-6s [", what);
    for (int i = 0; i < n && i < 24; i++) printf("%s%d", i ? ", " : "", v[i]);
    if (n > 24) printf(", ... %d more", n - 24);
    printf("]\n");
}

int main(void)
{
    struct stat st;
    if (stat("tests/fixtures/tok/tokenizer.bin", &st) != 0 ||
        stat("tests/fixtures/tok/tok_cases.json", &st) != 0) {
        printf("DeepSeek-V4 tokenizer gate\n");
        printf("  SKIP  no packed tokenizer. Build it with:\n");
        printf("          python3 tools/pack_tokenizer.py <model_dir> "
               "tests/fixtures/tok/tokenizer.bin\n");
        printf("          python3 tools/emit_tok_fixture.py <model_dir> "
               "tests/fixtures/tok\n");
        printf("\nTOKENIZER GATE SKIPPED\n");
        return 0;
    }

    printf("DeepSeek-V4 tokenizer gate\n");

    DSV4Tok t;
    if (dsv4_tok_load(&t, "tests/fixtures/tok/tokenizer.bin") != 0) {
        printf("  FAIL  could not load the packed tokenizer\n"); return 1;
    }
    printf("  loaded %u ids, %u merges, %u added\n",
           t.n_vocab, t.n_merge, t.n_added);

    char *txt = slurp("tests/fixtures/tok/tok_cases.json");
    char *arena = NULL;
    jval *G = json_parse(txt, &arena);
    if (!G) { printf("  FAIL  fixture is not valid JSON\n"); return 1; }

    jval *vs = json_get(G, "vocab_size");
    if (vs && vs->t == J_NUM && (uint32_t)vs->num != t.n_vocab) {
        printf("  FAIL  packed vocab is %u, the reference reports %d\n",
               t.n_vocab, (int)vs->num);
        fails++;
    }

    jval *cases = json_get(G, "cases");
    printf("\n");
    for (int i = 0; i < cases->len; i++) {
        const char *name = cases->keys[i];
        jval *c = cases->kids[i];
        jval *jt = json_get(c, "text");
        jval *ji = json_get(c, "ids");
        if (!jt || !ji) continue;

        const int n = (int)strlen(jt->str);
        static int32_t got[4096];
        const int ngot = dsv4_tok_encode(&t, jt->str, n, got, 4096);

        cases_run++;
        int ok = (ngot == ji->len);
        if (ok)
            for (int k = 0; k < ngot; k++)
                if (got[k] != (int32_t)ji->kids[k]->num) { ok = 0; break; }

        if (ok) {
            cases_ok++;
            printf("  ok    %-13s %d ids\n", name, ngot);
        } else {
            static int32_t want[4096];
            for (int k = 0; k < ji->len && k < 4096; k++)
                want[k] = (int32_t)ji->kids[k]->num;
            printf("  FAIL  %-13s C gave %d ids, reference %d\n",
                   name, ngot, ji->len);
            dump("C", got, ngot);
            dump("ref", want, ji->len);
            fails++;
        }
    }

    dsv4_tok_free(&t);
    printf("\n  %d of %d cases match exactly\n", cases_ok, cases_run);
    if (fails) { printf("TOKENIZER GATE FAILED: %d case(s)\n", fails); return 1; }
    printf("TOKENIZER GATE PASSED\n");
    return 0;
}
