// forward.h — Forward pass: ANE baked-weight conv for linears, CPU for element-wise
#pragma once
#include "model.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

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

static bool attention_forward_mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (!out) return true;
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

// ANE conv eval: input [S, in_dim] row-major → transpose to [in_dim, S] channels-first
// ANE computes conv(W, x) with baked W → output [out_dim, S]
// Transpose back to [S, out_dim] row-major
static bool ane_conv_eval(ANEKernel *kernel, const float *x, float *y,
                          int S, int in_dim, int out_dim) {
    float *x_t = (float*)malloc(S * in_dim * sizeof(float));
    for (int t = 0; t < S; t++)
        for (int i = 0; i < in_dim; i++)
            x_t[i*S + t] = x[t*in_dim + i];

    ane_write_input(kernel, 0, x_t, S * in_dim * sizeof(float));
    bool ok = ane_eval(kernel);
    if (!ok) {
        free(x_t);
        return false;
    }

    float *y_t = (float*)malloc(S * out_dim * sizeof(float));
    ane_read_output(kernel, 0, y_t, S * out_dim * sizeof(float));

    for (int t = 0; t < S; t++)
        for (int i = 0; i < out_dim; i++)
            y[t*out_dim + i] = y_t[i*S + t];

    free(x_t); free(y_t);
    return true;
}

// CPU matmul fallback: y = W @ x, W[out_dim, in_dim], x[S, in_dim] → y[S, out_dim]
static void cpu_matmul(const float *W, const float *x, float *y, int S, int in_dim, int out_dim) {
    for (int t = 0; t < S; t++)
        for (int i = 0; i < out_dim; i++) {
            float sum = 0;
            for (int j = 0; j < in_dim; j++)
                sum += W[i*in_dim + j] * x[t*in_dim + j];
            y[t*out_dim + i] = sum;
        }
}

static void cpu_rmsnorm(float *out, const float *x, const float *w, int S, int D) {
    for (int t = 0; t < S; t++) {
        float ss = 0;
        for (int i = 0; i < D; i++) ss += x[t*D+i] * x[t*D+i];
        ss = 1.0f / sqrtf(ss / D + 1e-5f);
        for (int i = 0; i < D; i++) out[t*D+i] = x[t*D+i] * ss * w[i];
    }
}

static void cpu_rope(float *q, float *k, int S, int n_heads, int head_dim) {
    for (int t = 0; t < S; t++)
        for (int h = 0; h < n_heads; h++)
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
                float val = t * freq;
                float cos_v = cosf(val), sin_v = sinf(val);
                int off = t * n_heads * head_dim + h * head_dim + i;
                float q0 = q[off], q1 = q[off+1];
                q[off]   = q0 * cos_v - q1 * sin_v;
                q[off+1] = q0 * sin_v + q1 * cos_v;
                float k0 = k[off], k1 = k[off+1];
                k[off]   = k0 * cos_v - k1 * sin_v;
                k[off+1] = k0 * sin_v + k1 * cos_v;
            }
}

// Hardened causal CPU attention. Temporary workspace is O(S), not O(S^2).
static ANE_Status cpu_attention_forward(const Model *m, float *out,
                                        const float *q, const float *k, const float *v,
                                        size_t S, int n_heads, int head_dim) {
    if (!m || !out || !q || !k || !v || S == 0 || n_heads <= 0 ||
        head_dim <= 0 || m->cfg.dim <= 0)
        return ANE_ERR_CONFIG;

    const size_t stride = (size_t)m->cfg.dim;
    const size_t expected_dim = (size_t)n_heads * (size_t)head_dim;
    if (expected_dim != stride)
        return ANE_ERR_CONFIG;

    size_t workspace_bytes = 0;
    if (attention_forward_mul_overflow_size(S, sizeof(float), &workspace_bytes))
        return ANE_ERR_OOM;

    float *scores = (float *)malloc(workspace_bytes);
    if (!scores)
        return ANE_ERR_OOM;

    ANE_Status status = ANE_OK;
    const float scale = 1.0f / sqrtf((float)head_dim);
    if (!isfinite(scale)) {
        status = ANE_ERR_NUMERIC;
        goto cleanup;
    }

    for (size_t h = 0; h < (size_t)n_heads; ++h) {
        const size_t head_offset = h * (size_t)head_dim;
        for (size_t t = 0; t < S; ++t) {
            const float *q_head = q + t * stride + head_offset;
            float *out_head = out + t * stride + head_offset;
            float max_score = -INFINITY;

            for (size_t s = 0; s <= t; ++s) {
                const float *k_head = k + s * stride + head_offset;
                float dot = 0.0f;
                for (size_t i = 0; i < (size_t)head_dim; ++i)
                    dot += q_head[i] * k_head[i];
                scores[s] = dot * scale;
                if (!isfinite(scores[s])) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
                if (scores[s] > max_score) max_score = scores[s];
            }

            float sum = 0.0f;
            for (size_t s = 0; s <= t; ++s) {
                scores[s] = expf(scores[s] - max_score);
                if (!isfinite(scores[s])) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
                sum += scores[s];
            }
            if (!isfinite(sum) || sum <= FLT_MIN) {
                status = ANE_ERR_NUMERIC;
                goto cleanup;
            }

            for (size_t i = 0; i < (size_t)head_dim; ++i)
                out_head[i] = 0.0f;

            for (size_t s = 0; s <= t; ++s) {
                const float probability = scores[s] / sum;
                const float *v_head = v + s * stride + head_offset;
                if (!isfinite(probability)) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
                for (size_t i = 0; i < (size_t)head_dim; ++i)
                    out_head[i] += probability * v_head[i];
            }

            for (size_t i = 0; i < (size_t)head_dim; ++i) {
                if (!isfinite(out_head[i])) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
            }
        }
    }

cleanup:
    free(scores);
    return status;
}

// Compatibility wrapper for existing callers that cannot propagate status.
// A failure is represented as NaN in the output so model_forward can abort.
static void cpu_attention(float *out, const float *q, const float *k, const float *v,
                          int S, int n_heads, int head_dim) {
    Model config = {0};
    config.cfg.dim = n_heads > 0 && head_dim > 0 ? n_heads * head_dim : 0;
    if (cpu_attention_forward(&config, out, q, k, v, (size_t)(S > 0 ? S : 0),
                              n_heads, head_dim) != ANE_OK && out) {
        out[0] = NAN;
    }
}

static inline float silu_f(float x) { return x / (1.0f + expf(-x)); }

// Forward pass — returns loss. Saves activations for backward.
static float model_forward(Model *m, const int *tokens, bool use_ane) {
    int S = m->seq_len, d = m->cfg.dim, hd = m->cfg.hidden_dim;
    int nh = m->cfg.n_heads, hdim = HEAD_DIM, vs = m->cfg.vocab_size;

    float *x = (float*)malloc(S * d * sizeof(float));
    for (int t = 0; t < S; t++)
        memcpy(x + t*d, m->token_embedding + tokens[t]*d, d * sizeof(float));

    for (int l = 0; l < N_LAYERS; l++) {
        memcpy(m->act_x[l], x, S * d * sizeof(float));

        cpu_rmsnorm(m->act_xnorm[l], x, m->rms_att_w[l], S, d);

        if (use_ane) {
            ane_conv_eval(m->kern_q[l], m->act_xnorm[l], m->act_q[l], S, d, d);
            ane_conv_eval(m->kern_k[l], m->act_xnorm[l], m->act_k[l], S, d, d);
            ane_conv_eval(m->kern_v[l], m->act_xnorm[l], m->act_v[l], S, d, d);
        } else {
            cpu_matmul(m->wq[l], m->act_xnorm[l], m->act_q[l], S, d, d);
            cpu_matmul(m->wk[l], m->act_xnorm[l], m->act_k[l], S, d, d);
            cpu_matmul(m->wv[l], m->act_xnorm[l], m->act_v[l], S, d, d);
        }

        cpu_rope(m->act_q[l], m->act_k[l], S, nh, hdim);
        Model attention_config = *m;
        ANE_Status attention_status = cpu_attention_forward(
            &attention_config, m->act_attn_out[l], m->act_q[l], m->act_k[l],
            m->act_v[l], (size_t)S, nh, hdim);
        if (attention_status != ANE_OK) {
            free(x);
            return NAN;
        }

        float *o_out = (float*)malloc(S * d * sizeof(float));
        if (use_ane) {
            ane_conv_eval(m->kern_o[l], m->act_attn_out[l], o_out, S, d, d);
        } else {
            cpu_matmul(m->wo[l], m->act_attn_out[l], o_out, S, d, d);
        }
        for (int i = 0; i < S * d; i++) x[i] += o_out[i];
        free(o_out);

        cpu_rmsnorm(m->act_ffn_in[l], x, m->rms_ffn_w[l], S, d);

        if (use_ane) {
            ane_conv_eval(m->kern_w1[l], m->act_ffn_in[l], m->act_h1[l], S, d, hd);
            ane_conv_eval(m->kern_w3[l], m->act_ffn_in[l], m->act_h3[l], S, d, hd);
        } else {
            cpu_matmul(m->w1[l], m->act_ffn_in[l], m->act_h1[l], S, d, hd);
            cpu_matmul(m->w3[l], m->act_ffn_in[l], m->act_h3[l], S, d, hd);
        }

        for (int t = 0; t < S; t++)
            for (int i = 0; i < hd; i++)
                m->act_silu[l][t*hd+i] = silu_f(m->act_h1[l][t*hd+i]) * m->act_h3[l][t*hd+i];

        float *ffn_out = (float*)malloc(S * d * sizeof(float));
        if (use_ane) {
            ane_conv_eval(m->kern_w2[l], m->act_silu[l], ffn_out, S, hd, d);
        } else {
            cpu_matmul(m->w2[l], m->act_silu[l], ffn_out, S, hd, d);
        }
        for (int i = 0; i < S * d; i++) x[i] += ffn_out[i];
        free(ffn_out);
    }

    memcpy(m->act_pre_final, x, S * d * sizeof(float));
    cpu_rmsnorm(m->act_final, x, m->rms_final_w, S, d);

    if (use_ane && m->kern_cls) {
        ane_conv_eval(m->kern_cls, m->act_final, m->logits, S, d, vs);
    } else {
        cpu_matmul(m->wcls, m->act_final, m->logits, S, d, vs);
    }

    free(x);

    float loss = 0;
    for (int t = 0; t < S - 1; t++) {
        float mx = -1e9f;
        for (int i = 0; i < vs; i++) if (m->logits[t*vs+i] > mx) mx = m->logits[t*vs+i];
        float sm = 0;
        for (int i = 0; i < vs; i++) sm += expf(m->logits[t*vs+i] - mx);
        float log_prob = m->logits[t*vs + tokens[t+1]] - mx - logf(sm);
        loss -= log_prob;
    }
    return loss / (S - 1);
}
