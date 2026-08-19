/* SPDX-License-Identifier: Apache-2.0 */
/* bench/cache_bw.c - expert cache read throughput, cold and warm.
 *
 * Reads DISTINCT experts on every repetition so the page cache cannot make the
 * later runs look fast for the wrong reason: a benchmark that re-reads the same
 * 100 experts measures RAM, not the disk, and the real engine's 148 GB working
 * set never fits in RAM. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "dsv4_cache.h"
#include "dsv4_cfg.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
                        return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  const char *m = argc>1?argv[1]:"/home/ronak/models/dsv4-flash";
  int reps = argc>2?atoi(argv[2]):3, per = argc>3?atoi(argv[3]):60;
  DSV4Cfg c; int cr[DSV4_MAX_LAYERS]; DSV4St st;
  if(!dsv4_cfg_load_file(&c,cr,DSV4_MAX_LAYERS,
     ({static char p[512];snprintf(p,sizeof p,"%s/config.json",m);p;}))) return 1;
  if(dsv4_st_open(&st,m)!=0) return 1;
  int e0 = 0;
  for(int r=0;r<reps;r++){
    DSV4Cache k;
    if(dsv4_cache_init(&k,&st,&c,13369344LL*2)!=0) return 1;
    double t0=now(); int n=0;
    for(int L=0;L<10;L++)
      for(int e=e0;e<e0+per/10;e++){ if(!dsv4_cache_get(&k,L,e)) return 1; n++; }
    double dt=now()-t0;
    double gb=(double)k.bytes_read/1073741824.0;
    printf("  rep %d (experts %d..%d): %.2f GB in %.2f s = %.2f GB/s, "
           "%.1f ms/expert  [%s]\n", r, e0, e0+per/10-1, gb, dt, gb/dt,
           dt/n*1e3, k.direct ? "O_DIRECT" : "buffered");
    e0 += per/10;                       /* fresh experts next rep */
    dsv4_cache_free(&k);
  }
  return 0;
}
