#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef ANE_STATUS_DEFINED
#define ANE_STATUS_DEFINED 1
typedef enum {
    ANE_OK = 0,
    ANE_ERR_CONFIG,
    ANE_ERR_OOM,
    ANE_ERR_NUMERIC,
    ANE_ERR_INTERNAL
} ANE_Status;
#endif

static inline bool attention_mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (!out) return true;
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static inline ANE_Status attention_validate_config(const Model *m, size_t S,
                                                    int n_heads, int head_dim) {
    if (!m || S == 0 || n_heads <= 0 || head_dim <= 0 || m->cfg.dim <= 0)
        return ANE_ERR_CONFIG;

    const size_t heads = (size_t)n_heads;
    const size_t dimension = (size_t)head_dim;
    if (heads > SIZE_MAX / dimension)
        return ANE_ERR_CONFIG;

    const size_t expected_dim = heads * dimension;
    if (expected_dim != (size_t)m->cfg.dim)
        return ANE_ERR_CONFIG;

    size_t elements = 0;
    if (attention_mul_overflow_size(S, expected_dim, &elements) || elements == 0)
        return ANE_ERR_OOM;

    return ANE_OK;
}

static inline bool attention_finite_buffer(const float *buffer, size_t count) {
    if (!buffer) return false;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(buffer[i])) return false;
    return true;
}
