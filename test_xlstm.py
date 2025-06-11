import torch

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4, chunk_size=128
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    num_blocks=7,
    embedding_dim=128,
    slstm_at=[1],

)

xlstm_stack = xLSTMBlockStack(cfg)

x = torch.randn(4, 256, 128).to("cuda")
xlstm_stack = xlstm_stack.to("cuda")
y1 = xlstm_stack(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y1.shape}")
print(f"Shapes match: {y1.shape == x.shape}")
print("xLSTM CUDA test passed! ✓")

# Test statefulness
print("\n--- Stateful forward pass test ---")
y2 = xlstm_stack(x, return_last_state=False)
y3, state = xlstm_stack(x, return_last_state=True)

assert torch.allclose(y1, y2)
assert torch.allclose(y1, y3)
print("Outputs with and without return_last_state match.")

# Check that all blocks returned a state and the content is correct
print(f"Returned state keys: {sorted(state.keys())}")
assert len(state.keys()) == cfg.num_blocks

for i in range(cfg.num_blocks):
    block_key = f"block_{i}"
    assert block_key in state
    block_state = state[block_key]

    if i in cfg.slstm_at:
        assert "slstm_state" in block_state
        assert "conv_state" in block_state
        print(f"State returned for sLSTM block {i} as expected.")
    else:
        assert "mlstm_state" in block_state
        assert "conv_state" in block_state
        assert block_state["mlstm_state"] is not None
        print(f"State returned for mLSTM block {i} as expected.")

# Pass the state back in, expect a different output
y4, new_state = xlstm_stack(x, state=state, return_last_state=True)
assert not torch.allclose(y1, y4)
print("Passing state results in different output as expected.")

print("xLSTM stateful forward test passed! ✓")


# Test chunkwise forward pass
print("\n--- Chunkwise forward pass test ---")
# Process the whole sequence at once
y_full, state_full = xlstm_stack(x, return_last_state=True)

# Process in two chunks
seq_len = x.shape[1]
x1 = x[:, :seq_len//2, :]
x2 = x[:, seq_len//2:, :]

y1_chunk, state1 = xlstm_stack(x1, return_last_state=True)
y2_chunk, state2 = xlstm_stack(x2, state=state1, return_last_state=True)

# Concatenate results from chunked processing
y_chunkwise = torch.cat([y1_chunk, y2_chunk], dim=1)

# Compare the outputs
assert torch.allclose(y_full, y_chunkwise, atol=1e-2, rtol=1e-2)
print("Chunkwise forward pass output matches full forward pass output.")

# Compare the final states
for block_key in state_full.keys():
    for state_key in state_full[block_key].keys():
        for i in range(len(state_full[block_key][state_key])):
            assert torch.allclose(state_full[block_key][state_key][i], state2[block_key][state_key][i], atol=1e-2, rtol=1e-2)
print("Chunkwise forward pass final state matches full forward pass final state.")


print("xLSTM chunkwise forward test passed! ✓")