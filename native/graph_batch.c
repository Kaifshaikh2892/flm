/* Fixed-width sparse matrix/dense matrix product for offline graph batches.
 * All edges in CSR order, IEEE float32, no fast-math or fused multiply-add.
 * A lane is an independent conversation. No cross-lane reduction occurs.
 */
#include <stdint.h>
void flm_csr8(int32_t n, const int32_t *restrict ptr, const int32_t *restrict ix,
              const float *restrict weight, const float *restrict x, float *restrict y) {
    for (int32_t row=0; row<n; ++row) {
        float s[8]={0};
        for (int32_t p=ptr[row]; p<ptr[row+1]; ++p) {
            const float w=weight[p];
            const float *v=x+(int64_t)ix[p]*8;
            for (int j=0; j<8; ++j) s[j]+=w*v[j];
        }
        for (int j=0;j<8;++j) y[(int64_t)row*8+j]=s[j];
    }
}
