#pragma GCC target("avx") // Hint to the compiler to allow AVX instructions

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <malloc.h> // For _aligned_malloc on Windows/MSVC
#include <omp.h>    // Added for Phase 5 OpenMP support

#define CLEAR_STDIN() do { int ch; while ((ch = getchar()) != '\n' && ch != EOF); } while(0)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Define the vector type for GNU-C Vector Extensions (aligned to 32 bytes by default)
typedef double v4df __attribute__ ((vector_size (32)));

void fill(double* x, int n) {
    int i;
    for (i = 0, n = n * n; i < n; i++, x++)
        *x = ((double)(1 + rand() % 12345)) / ((double)(1 + rand() % 6789));
}

// Phase 0: Reference
void matrix_mult_index(int n, double* a, double* b, double* c) {
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            c[i * n + j] = 0;
            for (k = 0; k < n; k++)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
        }
}

// Phase 1: Pointers and Registers
void matrix_mult_ptr_reg(int n, double* a, double* b, double* c) {
    register double cij;
    register double *at, *bt;
    register int i, j, k;
    for (i = 0; i < n; i++, a += n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for (k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt += n)
                cij += *at * *bt;
            *c = cij;
        }
}

void matrix_mult_ptr_no_reg(int n, double* a, double* b, double* c) {
    double cij;
    double *at, *bt;
    int i, j, k;
    for (i = 0; i < n; i++, a += n)
        for (j = 0; j < n; j++, c++) {
            cij = 0;
            for (k = 0, at = a, bt = &b[j]; k < n; k++, at++, bt += n)
                cij += *at * *bt;
            *c = cij;
        }
}

// Phase 2 - Method 1: Transpose + Pointers + Registers
void matrix_mult_transpose(int n, double* a, double* b, double* c) {
    register int i, j, k;

    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) {
        printf("Memory Allocation Error for transposed matrix.\n");
        return;
    }

    register double *p_b = b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++, p_b++) {
            b_t[j * n + i] = *p_b;
        }
    }

    register double sum;
    register double *pa, *pb_t;
    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j++, c++) {
            sum = 0.0;
            pb_t = b_t + (j * n);
            for (k = 0, pa = a; k < n; k++, pa++, pb_t++) {
                sum += (*pa) * (*pb_t);
            }
            *c = sum;
        }
    }

    _aligned_free(b_t);
}

// Phase 2 - Method 2: Block/Tiling + Pointers + Registers
void matrix_mult_block(int n, int block_size, double* a, double* b, double* c) {
    register int i, j, k, i_inner, j_inner, k_inner;
    register double a_val;
    register double *pc, *pb;

    register double *pc_init = c;
    for (i = 0; i < n * n; i++, pc_init++) {
        *pc_init = 0.0;
    }

    for (i = 0; i < n; i += block_size) {
        for (k = 0; k < n; k += block_size) {
            for (j = 0; j < n; j += block_size) {
                register int i_limit = MIN(i + block_size, n);
                register int k_limit = MIN(k + block_size, n);
                register int j_limit = MIN(j + block_size, n);

                for (i_inner = i; i_inner < i_limit; i_inner++) {
                    for (k_inner = k; k_inner < k_limit; k_inner++) {
                        a_val = a[i_inner * n + k_inner];
                        pc = c + (i_inner * n + j);
                        pb = b + (k_inner * n + j);
                        for (j_inner = j; j_inner < j_limit; j_inner++, pc++, pb++) {
                            *pc += a_val * (*pb);
                        }
                    }
                }
            }
        }
    }
}

// Phase 3: Loop Unrolling Factor 2
void matrix_mult_unrolled_2(int n, double* a, double* b, double* c) {
    register int i, j, k;

    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) return;

    register double *p_b = b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++, p_b++) b_t[j * n + i] = *p_b;
    }

    register double *pa, *pb_t;
    int limit = n - (n % 2);

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j++, c++) {
            register double sum1 = 0.0;
            register double sum2 = 0.0;

            pb_t = b_t + (j * n);

            for (k = 0, pa = a; k < limit; k += 2, pa += 2, pb_t += 2) {
                sum1 += pa[0] * pb_t[0];
                sum2 += pa[1] * pb_t[1];
            }

            register double total_sum = sum1 + sum2;

            for (; k < n; k++, pa++, pb_t++) {
                total_sum += (*pa) * (*pb_t);
            }

            *c = total_sum;
        }
    }
    _aligned_free(b_t);
}

// Phase 3: Loop Unrolling Factor 4
void matrix_mult_unrolled_4(int n, double* a, double* b, double* c) {
    register int i, j, k;

    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) return;

    register double *p_b = b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++, p_b++) b_t[j * n + i] = *p_b;
    }

    register double *pa, *pb_t;
    int limit = n - (n % 4);

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j++, c++) {
            register double sum1 = 0.0;
            register double sum2 = 0.0;
            register double sum3 = 0.0;
            register double sum4 = 0.0;

            pb_t = b_t + (j * n);

            for (k = 0, pa = a; k < limit; k += 4, pa += 4, pb_t += 4) {
                sum1 += pa[0] * pb_t[0];
                sum2 += pa[1] * pb_t[1];
                sum3 += pa[2] * pb_t[2];
                sum4 += pa[3] * pb_t[3];
            }

            register double total_sum = sum1 + sum2 + sum3 + sum4;

            for (; k < n; k++, pa++, pb_t++) {
                total_sum += (*pa) * (*pb_t);
            }

            *c = total_sum;
        }
    }
    _aligned_free(b_t);
}

// Phase 3: Loop Unrolling Factor 8
void matrix_mult_unrolled_8(int n, double* a, double* b, double* c) {
    register int i, j, k;

    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) return;

    register double *p_b = b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++, p_b++) b_t[j * n + i] = *p_b;
    }

    register double *pa, *pb_t;
    int limit = n - (n % 8);

    for (i = 0; i < n; i++, a += n) {
        for (j = 0; j < n; j++, c++) {
            register double sum1 = 0.0; register double sum2 = 0.0;
            register double sum3 = 0.0; register double sum4 = 0.0;
            register double sum5 = 0.0; register double sum6 = 0.0;
            register double sum7 = 0.0; register double sum8 = 0.0;

            pb_t = b_t + (j * n);

            for (k = 0, pa = a; k < limit; k += 8, pa += 8, pb_t += 8) {
                sum1 += pa[0] * pb_t[0]; sum2 += pa[1] * pb_t[1];
                sum3 += pa[2] * pb_t[2]; sum4 += pa[3] * pb_t[3];
                sum5 += pa[4] * pb_t[4]; sum6 += pa[5] * pb_t[5];
                sum7 += pa[6] * pb_t[6]; sum8 += pa[7] * pb_t[7];
            }

            register double total_sum = sum1 + sum2 + sum3 + sum4 +
                                        sum5 + sum6 + sum7 + sum8;

            for (; k < n; k++, pa++, pb_t++) {
                total_sum += (*pa) * (*pb_t);
            }

            *c = total_sum;
        }
    }
    _aligned_free(b_t);
}

// Phase 4: Cache Friendly (1D Strip Mining on Transposed Matrix)
void matrix_mult_cache_friendly(int n, double* a, double* b, double* c) {
    register double cij;
    register double *at, *bt;
    register int i, j, k, jj;
    register int BS = 64; // Block Size

    // Create a transposed copy of b to avoid mutating original
    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) return;

    register double *p_b = b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++, p_b++) b_t[j * n + i] = *p_b;
    }

    // Cache-friendly multiplication loops
    for (jj = 0; jj < n; jj += BS) {
        for (i = 0; i < n; i++) {
            for (j = jj; j < jj + BS && j < n; j++) {
                cij = 0.0;
                at = a + i * n;
                bt = b_t + j * n;

                for (k = 0; k < n; k++, at++, bt++) {
                    cij += (*at) * (*bt);
                }
                c[i * n + j] = cij;
            }
        }
    }
    _aligned_free(b_t);
}

// Phase 4: Cache Friendly + GNU-C Vector Extensions
void matrix_mult_cache_friendly_vec(int n, double* a, double* b, double* c) {
    register double cij;
    register double *at, *bt;
    register int i, j, k, jj;
    register int BS = 64; // Block Size

    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) return;

    register double *p_b = b;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++, p_b++) b_t[j * n + i] = *p_b;
    }

    int vec_limit = n - (n % 16);

    for (jj = 0; jj < n; jj += BS) {
        for (i = 0; i < n; i++) {
            for (j = jj; j < jj + BS && j < n; j++) {
                at = a + i * n;
                bt = b_t + j * n;

                v4df sum0 = {0.0, 0.0, 0.0, 0.0};
                v4df sum1 = {0.0, 0.0, 0.0, 0.0};
                v4df sum2 = {0.0, 0.0, 0.0, 0.0};
                v4df sum3 = {0.0, 0.0, 0.0, 0.0};

                v4df *vec_at = (v4df*)at;
                v4df *vec_bt = (v4df*)bt;

                for (k = 0; k < vec_limit; k += 16) {
                    sum0 += vec_at[0] * vec_bt[0];
                    sum1 += vec_at[1] * vec_bt[1];
                    sum2 += vec_at[2] * vec_bt[2];
                    sum3 += vec_at[3] * vec_bt[3];

                    vec_at += 4;
                    vec_bt += 4;
                }

                v4df final_vec = sum0 + sum1 + sum2 + sum3;
                cij = final_vec[0] + final_vec[1] + final_vec[2] + final_vec[3];

                at = (double*)vec_at;
                bt = (double*)vec_bt;
                for (; k < n; k++, at++, bt++) {
                    cij += (*at) * (*bt);
                }

                c[i * n + j] = cij;
            }
        }
    }
    _aligned_free(b_t);
}

// =========================================================================
// Phase 5: OpenMP Multithreading + Vector Extensions + Cache Friendly
// =========================================================================
void matrix_mult_omp_vec(int n, double* a, double* b, double* c) {
    int i, j;
    int BS = 64; // Block Size

    // Allocate transpose buffer
    double* b_t = (double*)_aligned_malloc(n * n * sizeof(double), 64);
    if (b_t == NULL) return;

    // 1. Parallelize the transpose
    #pragma omp parallel for private(j) schedule(static)
    for (i = 0; i < n; i++) {
        double *p_b = b + i * n;
        for (j = 0; j < n; j++) {
            b_t[j * n + i] = p_b[j];
        }
    }

    int vec_limit = n - (n % 16);

    // 2. Parallelize the multiplication
    // We open the parallel region ONCE to avoid thread fork/join overhead on every block
    #pragma omp parallel
    {
        // Thread-private variables
        register int jj, j_inner, k;
        register double *at, *bt;
        register double cij;

        // Keep the cache-friendly block loop strictly on the OUTSIDE
        for (jj = 0; jj < n; jj += BS) {

            // Distribute the rows of 'A' among the threads for the current block.
            // Static scheduling ensures maximum cache contiguity and zero OpenMP overhead.
            #pragma omp for schedule(static)
            for (i = 0; i < n; i++) {

                for (j_inner = jj; j_inner < jj + BS && j_inner < n; j_inner++) {
                    at = a + i * n;
                    bt = b_t + j_inner * n;

                    v4df sum0 = {0.0, 0.0, 0.0, 0.0};
                    v4df sum1 = {0.0, 0.0, 0.0, 0.0};
                    v4df sum2 = {0.0, 0.0, 0.0, 0.0};
                    v4df sum3 = {0.0, 0.0, 0.0, 0.0};

                    register v4df *vec_at = (v4df*)at;
                    register v4df *vec_bt = (v4df*)bt;

                    for (k = 0; k < vec_limit; k += 16) {
                        sum0 += vec_at[0] * vec_bt[0];
                        sum1 += vec_at[1] * vec_bt[1];
                        sum2 += vec_at[2] * vec_bt[2];
                        sum3 += vec_at[3] * vec_bt[3];

                        vec_at += 4;
                        vec_bt += 4;
                    }

                    v4df final_vec = sum0 + sum1 + sum2 + sum3;
                    cij = final_vec[0] + final_vec[1] + final_vec[2] + final_vec[3];

                    // Tail loop for leftovers
                    at = (double*)vec_at;
                    bt = (double*)vec_bt;
                    for (; k < n; k++, at++, bt++) {
                        cij += (*at) * (*bt);
                    }

                    c[i * n + j_inner] = cij;
                }
            }
        }
    } // End of parallel region

    _aligned_free(b_t);
}
// =========================================================================

int main()
{
    clock_t t0, t1;
    double t_start, t_end; // For OpenMP higher-resolution timing
    int n, ref;

    do {
        printf("Input size of matrix, n = ");
        if(scanf("%d", &n) != 1) break;

        ref = 0;

        double* A = (double*)_aligned_malloc(n * n * sizeof(double), 64);
        double* B = (double*)_aligned_malloc(n * n * sizeof(double), 64);
        double* C1 = (double*)_aligned_malloc(n * n * sizeof(double), 64);
        double* C2 = (double*)_aligned_malloc(n * n * sizeof(double), 64);

        if (A == NULL || B == NULL || C1 == NULL || C2 == NULL) {
            printf("Memory Allocation Error\n\n");
            return(-1);
        }

        unsigned int seed = time(NULL);
        printf("\nseed = %u\n", seed);

        srand(seed);
        fill(A, n);
        fill(B, n);

        // Phase 0
        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_index (y/n)? ");
        if (getchar() == 'y') {
            ref = 1;
            t0 = clock();
            matrix_mult_index(n, A, B, C1);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_index = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        // Phase 1
        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_ptr_reg (y/n)? ");
        if (getchar() == 'y') {
            ref++;
            t0 = clock();
            matrix_mult_ptr_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_reg = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_ptr_no_reg (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_ptr_no_reg(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_ptr_no_reg = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        // Phase 2
        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_transpose (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_transpose(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_transpose = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_block (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            int block_size;
            printf("\n\tInput size of block = ");
            scanf("%d", &block_size);
            t0 = clock();
            matrix_mult_block(n, block_size, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_block = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        // Phase 3
        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_unrolled_2 (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_unrolled_2(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_unrolled_2 = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_unrolled_4 (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_unrolled_4(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_unrolled_4 = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_unrolled_8 (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_unrolled_8(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_unrolled_8 = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        // Phase 4
        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_cache_friendly (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_cache_friendly(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_cache_friendly = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_cache_friendly_vec (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;
            t0 = clock();
            matrix_mult_cache_friendly_vec(n, A, B, ref == 1 ? C1 : C2);
            t1 = clock();
            printf("\n\t\t\tExecution time of matrix_mult_cache_friendly_vec = %0.2f s", (float)(t1 - t0) / CLOCKS_PER_SEC);
        }

        // Phase 5
        CLEAR_STDIN();
        printf("\n\nDo you want to run matrix_mult_omp_vec (y/n)? ");
        if (getchar() == 'y') {
            if (++ref > 2) ref = 2;

            // clock() measures total CPU time across all threads.
            // We use omp_get_wtime() here for real-world wall clock time
            t_start = omp_get_wtime();
            matrix_mult_omp_vec(n, A, B, ref == 1 ? C1 : C2);
            t_end = omp_get_wtime();

            printf("\n\t\t\tExecution time of matrix_mult_omp_vec = %0.2f s", t_end - t_start);
        }

        printf("\n\n\nEnd Of Execution\n\n");

        if (ref == 2) {
            int i;
            double* c1, * c2;
            printf("\n\nStart of Compare: ");
            for (i = 0, c1 = C1, c2 = C2, n = n * n; i < n; i++, c1++, c2++) {
                if (fabs((*c1 - *c2) / *c1) > 1E-10)
                    break;
                if (i % (n / 20) == 0)
                    printf(".");
            }

            if (i != n)
                printf(" Ooops, Error Found @ %d: %0.3f vs %0.3f\n\n", i, *c1, *c2);
            else
                printf(" OK, OK, Matrixes are equivalent.\n\n");
        }
        else
            printf("\n\nNo Compare due to No Reference or No Data.\n\n");

        _aligned_free(A);
        _aligned_free(B);
        _aligned_free(C1);
        _aligned_free(C2);

        CLEAR_STDIN();
        printf("\n\nDo you want to continue (y/n)? ");

    } while (getchar() == 'y');

    return 0;
}
