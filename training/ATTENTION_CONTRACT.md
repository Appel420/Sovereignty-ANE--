# Attention Contract Integration

This file records the concrete hardening boundary applied to the existing CPU causal-attention path.

Implemented in the training headers:

- Configuration validation before attention tensor traversal.
- Canonical row stride derived from `Model.cfg.dim`.
- Checked dimension and workspace multiplication.
- O(S) score workspace for forward and backward paths.
- Explicit allocation failure handling and cleanup.
- Stable max-subtracted softmax.
- Non-finite score, probability, output, and gradient rejection.
- Causal masking preserved through the `s <= t` traversal.
- Structured `ANE_Status` results.

This document does not claim ANE parity, sanitizer success, finite-difference results, signed artifacts, or production readiness. Those require executable device-side validation.
