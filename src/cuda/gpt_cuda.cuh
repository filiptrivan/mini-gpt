#ifndef GPT_CUDA_CUH
#define GPT_CUDA_CUH

/*
 * gpt_cuda.cuh — GPU forward/backward for the whole GPT model.
 *
 * These are the device-resident counterparts of gpt_forward / gpt_backward in
 * model/gpt.h. They take the SAME GPTModel and the same (tokens, targets, B, T)
 * and produce the same results — model->loss and model->a.* for the forward,
 * model->grads for the backward — but they run every layer op as a CUDA kernel
 * with the tensors kept resident on the GPU between kernels (see gpt_cuda.cu).
 *
 * They are the integration point validated on Colab: a test runs the CPU and
 * the CUDA versions on two identically-seeded models and asserts the loss and
 * gradients agree within tolerance.
 *
 * extern "C": gpt_cuda.cu is compiled by nvcc (C++), but these must be callable
 * from C (the training loop). Same self-describing-linkage pattern as gpt.h /
 * cuda_layers.cuh.
 */

#include "model/gpt.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * gpt_forward_cuda — GPU forward pass. Same contract as gpt_forward (gpt.h):
 * runs the model on (tokens, targets, B, T), fills model->a.* (copied back from
 * the device) and, if targets != NULL, model->loss. Allocates the activation
 * blocks on the first call via gpt_ensure_acts, exactly like the CPU forward.
 */
void gpt_forward_cuda(GPTModel *model, const int *tokens, const int *targets,
                      int B, int T);

/*
 * gpt_backward_cuda — GPU backward pass. Same contract as gpt_backward (gpt.h):
 * a matching gpt_forward_cuda must have run first (its activations, left in
 * model->acts, are re-uploaded here). Computes the gradient of the loss w.r.t.
 * every weight and writes it into model->grads. The device gradient blocks are
 * zeroed internally, so there is no separate gpt_zero_grad step on the GPU.
 */
void gpt_backward_cuda(GPTModel *model, const int *tokens, const int *targets,
                       int B, int T);

#ifdef __cplusplus
}
#endif

#endif /* GPT_CUDA_CUH */
