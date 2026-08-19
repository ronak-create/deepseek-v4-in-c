/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok.h - DeepSeek-V4's byte-level BPE, from the packed tokenizer.
 *
 * THE PRE-TOKENIZER IS THE HARD PART, NOT THE MERGES
 *   The merge loop is mechanical: repeatedly join the adjacent pair with the
 *   lowest rank. Everything that can go subtly wrong lives in how the input is
 *   cut up first, because a piece boundary in the wrong place changes the ids
 *   without producing anything that looks like an error.
 *
 *   tokenizer.json's pre_tokenizer is a Sequence of four stages, applied in
 *   order, each operating on the pieces the previous one produced:
 *
 *     0.  Split \p{N}{1,3}                     Isolated
 *     1.  Split [CJK|hiragana|katakana]+       Isolated
 *     2.  Split <the big alternation>          Isolated
 *     3.  ByteLevel(add_prefix_space=false, use_regex=false)
 *
 *   "Isolated" means the match becomes its own piece and the text around it
 *   stays. Splits do NOT consume: a stage that finds nothing leaves its input
 *   untouched.
 *
 * THREE THINGS THAT ARE EASY TO GET WRONG
 *
 *   1. \p{N}{1,3} IS BOUNDED. It isolates runs of one to THREE digits, so
 *      "1234" becomes "123" + "4", not one piece and not four. Treating it as
 *      \p{N}+ is the obvious reading and changes every long number.
 *
 *   2. THE CJK CLASS IS SPECIFIC. U+4E00..U+9FA5 plus hiragana plus katakana,
 *      NOT "anything non-Latin". Cyrillic and Arabic go through stage 2 and are
 *      handled by the letter-run alternative instead.
 *
 *   3. STAGE 2'S ALTERNATIVES ARE ORDERED AND THE FIRST MATCH WINS.
 *        a) [ASCII punct][A-Za-z]+          "#define" is ONE piece
 *        b) [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+   an optional leading
 *                                           non-letter, then a letter run --
 *                                           this is what puts the SPACE on
 *                                           " the" rather than leaving it
 *        c)  ?[\p{P}\p{S}]+[\r\n]*          punctuation runs
 *        d) \s*[\r\n]+                      newline runs
 *        e) \s+(?!\S)                       trailing whitespace
 *        f) \s+                             any other whitespace
 *      Reordering b) and c) alone silently moves every leading space.
 *
 * BYTE LEVEL, NOT UTF-8
 *   Each input BYTE maps to a printable codepoint before matching against the
 *   vocab, so the token for " the" is spelled "Gthe" with U+0120. The packed
 *   vocab stores exactly those spellings, which is why nothing here decodes
 *   UTF-8 for lookup.
 */
#ifndef DSV4_TOK_H
#define DSV4_TOK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_unicode.h"

#define DSV4_TOK_MAGIC "DSV4TOK"

typedef struct {
    uint32_t *voff;      /* [n_vocab] offset into blob                 */
    uint16_t *vlen;      /* [n_vocab] byte length                      */
    uint32_t *ma, *mb;   /* [n_merge] the pair, as vocab ids           */
    uint32_t *mo;        /* [n_merge] what it merges into              */
    uint32_t *added;     /* [n_added] special/added ids                */
    char     *blob;
    uint32_t  n_vocab, n_merge, n_added;
    uint64_t  blob_bytes;

    /* rank[a] is a sorted list of (b, out, rank) for pairs starting with a.
     * Built once at load: the merge loop then finds a pair's rank without
     * scanning 127,741 entries per step. */
    int32_t  *pair_head;   /* [n_vocab] index into pair_* or -1 */
    int32_t  *pair_next;   /* [n_merge] */

    void     *raw;         /* one allocation backing everything */
} DSV4Tok;

/* ---- byte-level alphabet -------------------------------------------------
 * GPT-2's map: printable ASCII and most Latin-1 stay put, everything else
 * moves to U+0100.. so that no token ever contains a raw control byte. */
static inline uint32_t dsv4_byte_to_cp(unsigned char b)
{
    if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE))
        return b;
    /* the 68 remaining bytes, in ascending order, map to 256, 257, ... */
    uint32_t n = 0;
    for (unsigned c = 0; c < b; c++)
        if (!((c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE)))
            n++;
    return 256 + n;
}

static inline int dsv4_cp_to_utf8(uint32_t cp, char *out)
{
    if (cp < 0x80)   { out[0] = (char)cp; return 1; }
    if (cp < 0x800)  { out[0] = (char)(0xC0 | (cp >> 6));
                       out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

/* ---- UTF-8 decoding of the INPUT ---------------------------------------- */
static inline uint32_t dsv4_utf8_next(const char *s, int n, int *i)
{
    const unsigned char *p = (const unsigned char *)s + *i;
    const unsigned char c = p[0];
    if (c < 0x80)             { *i += 1; return c; }
    if ((c & 0xE0) == 0xC0 && *i + 1 < n)
                              { *i += 2; return ((uint32_t)(c & 0x1F) << 6)
                                              | (p[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0 && *i + 2 < n)
                              { *i += 3; return ((uint32_t)(c & 0x0F) << 12)
                                              | ((uint32_t)(p[1] & 0x3F) << 6)
                                              | (p[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0 && *i + 3 < n)
                              { *i += 4; return ((uint32_t)(c & 0x07) << 18)
                                              | ((uint32_t)(p[1] & 0x3F) << 12)
                                              | ((uint32_t)(p[2] & 0x3F) << 6)
                                              | (p[3] & 0x3F); }
    *i += 1;                                   /* malformed: consume one byte */
    return 0xFFFD;
}

static inline int dsv4_is_ascii_punct(uint32_t c)
{
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@')
        || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}
static inline int dsv4_is_alpha(uint32_t c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
static inline int dsv4_is_space(uint32_t c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r'
        || c == '\v' || c == '\f' || c == 0x85 || c == 0xA0
        || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029
        || c == 0x202F || c == 0x205F || c == 0x3000;
}
/* U+4E00..U+9FA5 ideographs, U+3040..U+309F hiragana, U+30A0..U+30FF katakana */
static inline int dsv4_is_cjk(uint32_t c)
{
    return (c >= 0x4E00 && c <= 0x9FA5)
        || (c >= 0x3040 && c <= 0x309F)
        || (c >= 0x30A0 && c <= 0x30FF);
}

#endif /* DSV4_TOK_H */
