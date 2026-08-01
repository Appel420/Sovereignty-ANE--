#import <Foundation/Foundation.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "backward.h"

static void test_configuration_validation(void) {
    Model model = {0};
    model.cfg.dim = 4;

    assert(attention_validate_config(&model, 1, 2, 2) == ANE_OK);
    assert(attention_validate_config(&model, 0, 2, 2) == ANE_ERR_CONFIG);
    assert(attention_validate_config(&model, 1, 0, 2) == ANE_ERR_CONFIG);
    assert(attention_validate_config(&model, 1, 2, 0) == ANE_ERR_CONFIG);
    assert(attention_validate_config(&model, 1, 1, 2) == ANE_ERR_CONFIG);
    assert(attention_validate_config(NULL, 1, 2, 2) == ANE_ERR_CONFIG);
}

static void test_overflow_detection(void) {
    size_t result = 0;
    assert(!attention_mul_overflow_size(4, sizeof(float), &result));
    assert(result == 4 * sizeof(float));
    assert(attention_mul_overflow_size(SIZE_MAX, 2, &result));
    assert(attention_mul_overflow_size(1, SIZE_MAX, &result));
    assert(attention_mul_overflow_size(1, 1, NULL));
}

static void test_valid_and_numeric_paths(void) {
    Model model = {0};
    model.cfg.dim = 2;

    float q[2] = {0.0f, 0.0f};
    float k[2] = {0.0f, 0.0f};
    float v[2] = {1.0f, 2.0f};
    float d_out[2] = {0.0f, 0.0f};
    float dq[2] = {99.0f, 99.0f};
    float dk[2] = {99.0f, 99.0f};
    float dv[2] = {99.0f, 99.0f};

    assert(cpu_attention_backward(&model, dq, dk, dv, d_out, q, k, v,
                                  1, 1, 2) == ANE_OK);
    assert(dq[0] == 0.0f && dq[1] == 0.0f);
    assert(dk[0] == 0.0f && dk[1] == 0.0f);
    assert(dv[0] == 0.0f && dv[1] == 0.0f);

    q[0] = NAN;
    assert(cpu_attention_backward(&model, dq, dk, dv, d_out, q, k, v,
                                  1, 1, 2) == ANE_ERR_NUMERIC);
}

int main(void) {
    @autoreleasepool {
        test_configuration_validation();
        test_overflow_detection();
        test_valid_and_numeric_paths();
    }
    return 0;
}
