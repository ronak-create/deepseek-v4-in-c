/* dsv4_st.h - safetensors reader for the real DeepSeek-V4 checkpoint.
 *
 * Ported from kimi-k3-in-c/src/io/k3_st.c (Apache-2.0). The logic is that file's
 * and is unchanged: the container format is identical and none of it encoded
 * anything about K3's architecture. This was established empirically before the
 * port -- the UNMODIFIED K3 reader was pointed at DeepSeek-V4-Flash shard 1 and
 * indexed embed.weight, resolved it through the hash index, computed
 * numel = 529,530,880 (= 129280 x 4096), read at offset 96, and decoded a
 * plausible embedding row (finite, mean 0.002, RMS 0.171). Only the dtype table
 * needed extending.
 *
 * FORMAT, verified against DeepSeek-V4-Flash model-00001/4/5-of-00046:
 *   [8 bytes little-endian N] [N bytes of JSON header] [tensor data]
 *   Each header entry: {"dtype": ..., "shape": [...], "data_offsets": [start, end]}
 *   data_offsets are relative to the END of the header, so the absolute file offset
 *   is 8 + N + start.
 *   The data region is fully contiguous with no gaps between tensors.
 *
 * THREE STORAGE FORMATS, NOT TWO  (read from the real headers, not the model card)
 *   BF16     embed.weight, every norm, ffn.gate.weight, all compressor.*
 *            projections. No scale tensor.
 *   F8_E4M3  attention (wq_a/wq_b/wkv/wo_a/wo_b), the shared expert, and
 *            indexer.wq_b. Paired with an F8_E8M0 .scale over 128x128 blocks.
 *   I8       the routed experts w1/w2/w3. This is FP4 PACKED TWO VALUES PER
 *            BYTE -- safetensors has no FP4 dtype -- so the declared shape is
 *            already in BYTES, not elements: w1.weight is [2048, 2048] for a
 *            [2048, 4096] matrix. Its F8_E8M0 .scale is [2048, 128], i.e. ONE
 *            SCALE PER 32 VALUES, an MXFP4-style 1x32 block.
 *
 *   The 128x128 in config.json's quantization_config.weight_block_size applies
 *   to the F8_E4M3 tensors ONLY. Applying it to the experts misreads every
 *   scale, and does so quietly, because the byte counts still add up.
 *
 * WHY A HASH INDEX IS NOT OPTIONAL
 *   Flash carries 43 layers x 256 experts x 3 matrices x 2 tensors = 66,048 expert
 *   tensors alone; Pro carries 140,544. A linear scan per lookup costs seconds per
 *   token. Every expert load does six lookups, and a decode step touches six
 *   experts across every layer, so lookups must be O(1).
 *
 * WHY pread AND NOT mmap
 *   Pages read into a buffer the engine owns never become file-backed mappings
 *   counted against the process, so peak RSS tracks what is actually resident
 *   rather than the whole checkpoint. mmap would make the RSS figure meaningless.
 */
#ifndef DSV4_ST_H
#define DSV4_ST_H

#include <stddef.h>
#include <stdint.h>

/* Storage dtypes, exactly as they are spelled in the safetensors header.
 *
 * DSV4_DT_I8 IS NOT AN INTEGER TENSOR. It is how DeepSeek ships packed FP4:
 * two 4-bit values per byte, because safetensors has no FP4 dtype. Consequently
 * dsv4_st_numel() on such a tensor counts BYTES, and the element count is twice
 * that. Every caller must know which it wants; the reader deliberately does not
 * guess, and never unpacks.
 *
 * DSV4_DT_F8_E8M0 is an exponent-only scale (no mantissa, no sign): the stored
 * byte b denotes 2^(b-127). It appears only as the .scale partner of an
 * F8_E4M3 or I8 weight.
 *
 * K3's DSV4_DT_I8R (per-row int8 draft format) is deliberately absent. It served
 * K3's speculative draft model, which has no counterpart here, and carrying a
 * dtype no kernel implements is the same mistake as defaulting a config field.
 */
typedef enum { DSV4_DT_UNKNOWN = 0, DSV4_DT_U8, DSV4_DT_BF16, DSV4_DT_F16,
               DSV4_DT_F32, DSV4_DT_F8_E4M3, DSV4_DT_F8_E8M0, DSV4_DT_I8,
               DSV4_DT_I64 } DSV4Dtype;

/* 1 when the dtype packs two values into each stored byte. */
static inline int dsv4_dt_is_packed4(DSV4Dtype d) { return d == DSV4_DT_I8; }

typedef struct {
    char     *name;
    int       shard;          /* index into DSV4St.fd[]                      */
    DSV4Dtype   dtype;
    int       ndim;
    int64_t   shape[4];
    int64_t   off;            /* ABSOLUTE byte offset within its shard file */
    int64_t   nbytes;
} DSV4Tensor;

/* O_DIRECT alignment. Offset, length and buffer must all be multiples of this. */
#define DSV4_ST_ALIGN 4096

typedef struct {
    int       *fd;            /* one open descriptor per shard             */
    int       *dfd;           /* the same shards opened O_DIRECT, or -1    */
    char     **path;
    int        nshard;

    DSV4Tensor  *t;             /* every tensor, in discovery order          */
    int        nt;

    int32_t   *bucket;        /* open-addressed hash, -1 empty             */
    int        nbucket;

    char      *strpool;       /* all names, one allocation                 */
    size_t     strcap, strlen_;
} DSV4St;

/* Open every *.safetensors in dir and index every tensor. Returns 0 on success. */
int  dsv4_st_open(DSV4St *s, const char *dir);
void dsv4_st_close(DSV4St *s);

/* O(1) lookup. Returns NULL when absent, which callers must treat as fatal: a
 * silently missing weight reads as zeros and the model still runs, plausibly wrong. */
const DSV4Tensor *dsv4_st_find(const DSV4St *s, const char *name);

/* Raw bytes, exactly as stored. buf must hold t->nbytes. Returns bytes read. */
int64_t dsv4_st_read(const DSV4St *s, const DSV4Tensor *t, void *buf);

/* Read [off, off+nbytes) from a shard with O_DIRECT, bypassing the page cache.
 *
 * WHY THIS IS NOT JUST dsv4_st_read WITH A FLAG
 *   O_DIRECT demands that the file offset, the length and the buffer address all be
 *   multiples of DSV4_ST_ALIGN. An expert run starts wherever the checkpoint put it, so
 *   the read is WIDENED outward to the enclosing aligned window and the caller is told
 *   where the payload actually begins inside buf. buf must therefore hold
 *   nbytes + 2*DSV4_ST_ALIGN and be page aligned (posix_memalign).
 *
 *   Worth it because a streamed expert is read once and evicted: the page cache can only
 *   copy it twice and push out something useful. Under a 32 GB cgroup cap the buffered
 *   path measured 1,247 MB/s on a disk that does 6,400.
 *
 * Returns payload bytes available, and sets *payload_off. Falls back to a buffered read
 * at offset 0 (setting *payload_off = 0) when O_DIRECT is not available. */
int64_t dsv4_st_read_aligned(const DSV4St *s, int shard, int64_t off, int64_t nbytes,
                           void *buf, int64_t bufcap, int64_t *payload_off);

/* Read and widen to float32. Handles F32 (memcpy), BF16 (shift left 16), F16, and
 * U8 (raw byte value, for callers that want the quantised codes as numbers).
 * out must hold t->nbytes/elem_size floats. */
int64_t dsv4_st_read_f32(const DSV4St *s, const DSV4Tensor *t, float *out);

/* Elements in a tensor, and bytes per element for its dtype. */
int64_t dsv4_st_numel(const DSV4Tensor *t);
int     dsv4_st_elemsize(DSV4Dtype d);

/* bf16 -> f32 is a pure bit shift: bf16 IS the top 16 bits of an f32. No table, no
 * rounding, and unlike f16 there is no exponent rebias. */
static inline float dsv4_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

#endif /* DSV4_ST_H */
