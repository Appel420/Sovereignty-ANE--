// backward.h — Backward pass using CPU matmul (correct gradients) + ANE optional
#pragma once
#include "model.h"
#include "forward.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Structured status returned by hardened CPU attention routines.
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

static bool attention_mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (!out) return true;
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static ANE_Status attention_validate_config(const Model *m, size_t S,
                                             int n_heads, int head_dim) {
    if (!m || S == 0 || n_heads <= 0 || head_dim <= 0 || m->cfg.dim <= 0)
        return ANE_ERR_CONFIG;

    const size_t expected_dim = (size_t)n_heads * (size_t)head_dim;
    if (expected_dim != (size_t)m->cfg.dim)
        return ANE_ERR_CONFIG;

    return ANE_OK;
}

static bool attention_finite_buffer(const float *buffer, size_t count) {
    if (!buffer) return false;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(buffer[i])) return false;
    return true;
}

// dW += dy @ x^T — dy: [S, out_dim], x: [S, in_dim], dW: [out_dim, in_dim]
static void cpu_accum_dW(float *dW, const float *dy, const float *x, int S, int out_dim, int in_dim) {
    for (int t = 0; t < S; t++)
        for (int i = 0; i < out_dim; i++)
            for (int j = 0; j < in_dim; j++)
                dW[i*in_dim+j] += dy[t*out_dim+i] * x[t*in_dim+j];
}

// dx = W^T @ dy — W: [out_dim, in_dim], dy: [S, out_dim] → dx: [S, in_dim]
static void cpu_matmul_backward_dx(const float *W, const float *dy, float *dx,
                                    int S, int out_dim, int in_dim) {
    for (int t = 0; t < S; t++)
        for (int j = 0; j < in_dim; j++) {
            float sum = 0;
            for (int i = 0; i < out_dim; i++)
                sum += W[i*in_dim+j] * dy[t*out_dim+i];
            dx[t*in_dim+j] = sum;
        }
}

static void cpu_rmsnorm_backward(float *dx, const float *dy, const float *x, const float *w,
                                  int S, int D) {
    for (int t = 0; t < S; t++) {
        float ss = 0;
        for (int i = 0; i < D; i++) ss += x[t*D+i] * x[t*D+i];
        float rms = sqrtf(ss / D + 1e-5f);
        float inv_rms = 1.0f / rms;
        float dot = 0;
        for (int i = 0; i < D; i++)
            dot += dy[t*D+i] * w[i] * x[t*D+i];
        dot /= (D * rms * rms);
        for (int i = 0; i < D; i++)
            dx[t*D+i] = dy[t*D+i] * w[i] * inv_rms - x[t*D+i] * dot;
    }
}

static inline float silu_backward(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

// Production-hardened causal attention backward.
// The model configuration is validated before any tensor access. Temporary
// workspace is O(S), allocations are checked, and every failure reaches cleanup.
static ANE_Status cpu_attention_backward(
    const Model *m,
    float *dq, float *dk, float *dv,
    const float *d_out, const float *q, const float *k, const float *v,
    size_t S, int n_heads, int head_dim) {
    ANE_Status status = attention_validate_config(m, S, n_heads, head_dim);
    if (status != ANE_OK) return status;
    if (!dq || !dk || !dv || !d_out || !q || !k || !v)
        return ANE_ERR_CONFIG;

    const size_t stride = (size_t)m->cfg.dim;
    size_t tensor_elements = 0;
    size_t tensor_bytes = 0;
    size_t workspace_bytes = 0;
    if (attention_mul_overflow_size(S, stride, &tensor_elements) ||
        attention_mul_overflow_size(tensor_elements, sizeof(float), &tensor_bytes) ||
        attention_mul_overflow_size(S, sizeof(float), &workspace_bytes))
        return ANE_ERR_OOM;

    float *scores = NULL;
    float *dscores = NULL;

    scores = (float *)malloc(workspace_bytes);
    if (!scores) {
        status = ANE_ERR_OOM;
        goto cleanup;
    }
    dscores = (float *)malloc(workspace_bytes);
    if (!dscores) {
        status = ANE_ERR_OOM;
        goto cleanup;
    }

    memset(dq, 0, tensor_bytes);
    memset(dk, 0, tensor_bytes);
    memset(dv, 0, tensor_bytes);

    const float scale = 1.0f / sqrtf((float)head_dim);
    if (!isfinite(scale)) {
        status = ANE_ERR_NUMERIC;
        goto cleanup;
    }

    for (size_t h = 0; h < (size_t)n_heads; ++h) {
        const size_t head_offset = h * (size_t)head_dim;
        for (size_t t = 0; t < S; ++t) {
            const float *q_head = q + t * stride + head_offset;
            const float *do_head = d_out + t * stride + head_offset;
            float *dq_head = dq + t * stride + head_offset;
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
            for (size_t s = 0; s <= t; ++s) {
                scores[s] /= sum;
                if (!isfinite(scores[s])) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
            }

            float ds_sum = 0.0f;
            for (size_t s = 0; s <= t; ++s) {
                const float *v_head = v + s * stride + head_offset;
                float value_grad = 0.0f;
                for (size_t i = 0; i < (size_t)head_dim; ++i)
                    value_grad += do_head[i] * v_head[i];
                if (!isfinite(value_grad)) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
                dscores[s] = value_grad;
                ds_sum += scores[s] * value_grad;
            }
            if (!isfinite(ds_sum)) {
                status = ANE_ERR_NUMERIC;
                goto cleanup;
            }

            for (size_t s = 0; s <= t; ++s) {
                const float *k_head = k + s * stride + head_offset;
                float *dk_head = dk + s * stride + head_offset;
                const float ds = scores[s] * (dscores[s] - ds_sum) * scale;
                if (!isfinite(ds)) {
                    status = ANE_ERR_NUMERIC;
                    goto cleanup;
                }
                for (size_t i = 0; i < (size_t)head_dim; ++i) {
                    dq_head[i] += ds * k_head[i];
                    dk_head[i] += ds * q_head[i];
                }

                float *dv_head = dv + s * stride + head_offset;
                for (size_t i = 0; i < (size_t)head_dim; ++i)
                    dv_head[i] += scores[s] * do_head[i];
            }
        }
    }

    if (!attention_finite_buffer(dq, tensor_elements) ||
        !attention_finite_buffer(dk, tensor_elements) ||
        !attention_finite_buffer(dv, tensor_elements)) {
        status = ANE_ERR_NUMERIC;
        goto cleanup;
    }

    status = ANE_OK;

cleanup:
    free(scores);
    free(dscores);
    return status;
}

static void cpu_rope_backward(float *dq, float *dk, int S, int n_heads, int head_dim) {
    for (int t = 0; t < S; t++)
        for (int h = 0; h < n_heads; h++)
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(10000.0f, (float)i / head_dim);
                float val = t * freq;
                float cos_v = cosf(val), sin_v = sinf(val);
                int off = t * n_heads * head_dim + h * head_dim + i;
                float dq0 = dq[off], dq1 = dq[off+1];
                dq[off]   = dq0 * cos_v + dq1 * sin_v;
                dq[off+1] = -dq0 * sin_v + dq1 * cos_v;
                float dk0 = dk[off], dk1 = dk[off+1];
                dk[off]   = dk0 * cos_v + dk1 * sin_v;
                dk[off+1] = -dk0 * sin_v + dk1 * cos_v;
            }
}

static void model_clip_gradients(Model *m, float max_norm) {
    int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size;
    double total_norm_sq = 0;
    #define ACCUM_NORM(grad, size) do { \
        for (size_t _i = 0; _i < (size_t)(size); _i++) total_norm_sq += (double)(grad)[_i] * (grad)[_i]; \
    } while(0)
    for (int l = 0; l < N_LAYERS; l++) {
        ACCUM_NORM(m->grad_wq[l], d*d); ACCUM_NORM(m->grad_wk[l], d*d);
        ACCUM_NORM(m->grad_wv[l], d*d); ACCUM_NORM(m->grad_wo[l], d*d);
        ACCUM_NORM(m->grad_w1[l], hd*d); ACCUM_NORM(m->grad_w2[l], d*hd);
        ACCUM_NORM(m->grad_w3[l], hd*d);
    }
    ACCUM_NORM(m->grad_wcls, vs*d); ACCUM_NORM(m->grad_emb, vs*d);
    #undef ACCUM_NORM
    float total_norm = sqrtf((float)total_norm_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        #define SCALE_GRAD(grad, size) do { \
            for (size_t _i = 0; _i < (size_t)(size); _i++) (grad)[_i] *= scale; \
        } while(0)
        for (int l = 0; l < N_LAYERS; l++) {
            SCALE_GRAD(m->grad_wq[l], d*d); SCALE_GRAD(m->grad_wk[l], d*d);
            SCALE_GRAD(m->grad_wv[l], d*d); SCALE_GRAD(m->grad_wo[l], d*d);
            SCALE_GRAD(m->grad_w1[l], hd*d); SCALE_GRAD(m->grad_w2[l], d*hd);
            SCALE_GRAD(m->grad_w3[l], hd*d);
        }
        SCALE_GRAD(m->grad_wcls, vs*d); SCALE_GRAD(m->grad_emb, vs*d);
        #undef SCALE_GRAD
    }
}

static void model_backward(Model *m, const int *tokens) {
    int S = m->seq_len, d = m->cfg.dim, hd = m->cfg.hidden_dim;
    int nh = m->cfg.n_heads, hdim = HEAD_DIM, vs = m->cfg.vocab_size;

    for (int l = 0; l < N_LAYERS; l++) {
        memset(m->grad_wq[l], 0, d*d*sizeof(float));
        memset(m->grad_wk[l], 0, d*d*sizeof(float));
        memset(m->grad_wv[l], 0, d*d*sizeof(float));
        memset(m->grad_wo[l], 0, d*d*sizeof(float));
        memset(m->grad_w1[l], 0, hd*d*sizeof(float));
        memset(m->grad_w2[l], 0, d*hd*sizeof(float));
        memset(m->grad_w3[l], 0, hd*d*sizeof(float));
    }
    memset(m->grad_wcls, 0, (size_t)vs*d*sizeof(float));
    memset(m->grad_emb, 0, (size_t)vs*d*sizeof(float));

    float *dlogits = (float*)calloc(S * vs, sizeof(float));
    for (int t = 0; t < S - 1; t++) {
        float mx = -1e9f;
        for (int i = 0; i < vs; i++) if (m->logits[t*vs+i] > mx) mx = m->logits[t*vs+i];
        float sm = 0;
        for (int i = 0; i < vs; i++) sm += expf(m->logits[t*vs+i] - mx);
        for (int i = 0; i < vs; i++)
            dlogits[t*vs+i] = expf(m->logits[t*vs+i] - mx) / sm;
        dlogits[t*vs + tokens[t+1]] -= 1.0f;
        for (int i = 0; i < vs; i++)
            dlogits[t*vs+i] /= (S - 1);
    }

    cpu_accum_dW(m->grad_wcls, dlogits, m->act_final, S, vs, d);
    float *dx = (float*)calloc(S * d, sizeof(float));
    cpu_matmul_backward_dx(m->wcls, dlogits, dx, S, vs, d);
    free(dlogits);

    float *dx_norm = (float*)malloc(S * d * sizeof(float));
    cpu_rmsnorm_backward(dx_norm, dx, m->act_pre_final, m->rms_final_w, S, d);
    memcpy(dx, dx_norm, S * d * sizeof(float));
    free(dx_norm);

    for (int l = N_LAYERS - 1; l >= 0; l--) {
        float *d_silu = (float*)calloc(S * hd, sizeof(float));
        cpu_matmul_backward_dx(m->w2[l], dx, d_silu, S, d, hd);
        cpu_accum_dW(m->grad_w2[l], dx, m->act_silu[l], S, d, hd);

        float *d_h1 = (float*)malloc(S * hd * sizeof(float));
        float *d_h3 = (float*)malloc(S * hd * sizeof(float));
        for (int t = 0; t < S; t++)
            for (int i = 0; i < hd; i++) {
                d_h1[t*hd+i] = d_silu[t*hd+i] * m->act_h3[l][t*hd+i] * silu_backward(m->act_h1[l][t*hd+i]);
                d_h3[t*hd+i] = d_silu[t*hd+i] * silu_f(m->act_h1[l][t*hd+i]);
            }
        free(d_silu);

        cpu_accum_dW(m->grad_w1[l], d_h1, m->act_ffn_in[l], S, hd, d);
        cpu_accum_dW(m->grad_w3[l], d_h3, m->act_ffn_in[l], S, hd, d);

        float *dx_ffn_in = (float*)calloc(S * d, sizeof(float));
        float *dx_w1 = (float*)malloc(S * d * sizeof(float));
        float *dx_w3 = (float*)malloc(S * d * sizeof(float));
        cpu_matmul_backward_dx(m->w1[l], d_h1, dx_w1, S, hd, d);
        cpu_matmul_backward_dx(m->w3[l], d_h3, dx_w3, S, hd, d);
        for (int i = 0; i < S * d; i++) dx_ffn_in[i] = dx_w1[i] + dx_w3[i];
        free(d_h1); free(d_h3); free(dx_w1); free(dx_w3);

        float *dx_ffn_norm = (float*)malloc(S * d * sizeof(float));
        cpu_rmsnorm_backward(dx_ffn_norm, dx_ffn_in, m->act_x[l], m->rms_ffn_w[l], S, d);
        for (int i = 0; i < S * d; i++) dx[i] += dx_ffn_norm[i];
        free(dx_ffn_in); free(dx_ffn_norm);

        float *d_attn_out = (float*)calloc(S * d, sizeof(float));
        cpu_matmul_backward_dx(m->wo[l], dx, d_attn_out, S, d, d);
        cpu_accum_dW(m->grad_wo[l], dx, m->act_attn_out[l], S, d, d);

        float *dq = (float*)calloc(S * d, sizeof(float));
        float *dk = (float*)calloc(S * d, sizeof(float));
        float *dv = (float*)calloc(S * d, sizeof(float));
        ANE_Status attention_status = cpu_attention_backward(
            m, dq, dk, dv, d_attn_out, m->act_q[l], m->act_k[l], m->act_v[l],
            (size_t)S, nh, hdim);
        free(d_attn_out);
        if (attention_status != ANE_OK) {
            fprintf(stderr, "attention backward failed with status %d at layer %d\n",
                    (int)attention_status, l);
            free(dq); free(dk); free(dv);
            free(dx);
            return;
        }

        cpu_rope_backward(dq, dk, S, nh, hdim);

        cpu_accum_dW(m->grad_wq[l], dq, m->act_xnorm[l], S, d, d);
        cpu_accum_dW(m->grad_wk[l], dk, m->act_xnorm[l], S, d, d);
        cpu_accum_dW(m->grad_wv[l], dv, m->act_xnorm[l], S, d, d);

        float *dx_qkv = (float*)calloc(S * d, sizeof(float));
        float *tmp = (float*)malloc(S * d * sizeof(float));
        cpu_matmul_backward_dx(m->wq[l], dq, tmp, S, d, d);
        for (int i = 0; i < S*d; i++) dx_qkv[i] += tmp[i];
        cpu_matmul_backward_dx(m->wk[l], dk, tmp, S, d, d);
        for (int i = 0; i < S*d; i++) dx_qkv[i] += tmp[i];
        cpu_matmul_backward_dx(m->wv[l], dv, tmp, S, d, d);
        for (int i = 0; i < S*d; i++) dx_qkv[i] += tmp[i];
        free(tmp); free(dq); free(dk); free(dv);

        float *dx_att_norm = (float*)malloc(S * d * sizeof(float));
        cpu_rmsnorm_backward(dx_att_norm, dx_qkv, m->act_x[l], m->rms_att_w[l], S, d);
        for (int i = 0; i < S * d; i++) dx[i] += dx_att_norm[i];
        free(dx_qkv); free(dx_att_norm);
    }

    for (int t = 0; t < S; t++)
        for (int i = 0; i < d; i++)
            m->grad_emb[tokens[t]*d + i] += dx[t*d + i];

    free(dx);
}

static void model_adam_step(Model *m, float lr, float beta1, float beta2, float eps) {
    m->adam_step++;
    float bc1 = 1.0f - powf(beta1, m->adam_step);
    float bc2 = 1.0f - powf(beta2, m->adam_step);
    size_t idx = 0;

    #define ADAM_UPDATE(param, grad, size) do { \
        for (size_t _i = 0; _i < (size_t)(size); _i++) { \
            float g = (grad)[_i]; \
            m->adam_m[idx] = beta1 * m->adam_m[idx] + (1-beta1) * g; \
            m->adam_v[idx] = beta2 * m->adam_v[idx] + (1-beta2) * g * g; \
            float m_hat = m->adam_m[idx] / bc1; \
            float v_hat = m->adam_v[idx] / bc2; \
            (param)[_i] -= lr * m_hat / (sqrtf(v_hat) + eps); \
            idx++; \
        } \
    } while(0)

    int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size;
    for (int l = 0; l < N_LAYERS; l++) {
        ADAM_UPDATE(m->wq[l], m->grad_wq[l], d*d);
        ADAM_UPDATE(m->wk[l], m->grad_wk[l], d*d);
        ADAM_UPDATE(m->wv[l], m->grad_wv[l], d*d);
        ADAM_UPDATE(m->wo[l], m->grad_wo[l], d*d);
        ADAM_UPDATE(m->w1[l], m->grad_w1[l], hd*d);
        ADAM_UPDATE(m->w2[l], m->grad_w2[l], d*hd);
        ADAM_UPDATE(m->w3[l], m->grad_w3[l], hd*d);
    }
    ADAM_UPDATE(m->wcls, m->grad_wcls, vs*d);
    ADAM_UPDATE(m->token_embedding, m->grad_emb, vs*d);
    #undef ADAM_UPDATE
}
