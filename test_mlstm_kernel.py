import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "external" / "mlstm_kernels"))

import torch
# directly import mLSTMexp TFLA kernel
from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk

# run the kernel
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
B = 2
S = 512
DHQK = 128
DHHV = 256
NH = 4

torch.manual_seed(1)
matQ = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
matK = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
matV = torch.randn((B, NH, S, DHHV), dtype=DTYPE, device=DEVICE)
vecI = torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
vecF = 3.0 + torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)

# First, run on the whole sequence to get a reference output
matH_ref, (C_ref, n_ref, m_ref) = mlstm_chunkwise__xl_chunk(
    q=matQ, k=matK, v=matV, i=vecI, f=vecF, return_last_states=True, chunk_size=256
)

# Now, run in two parts and pass the state
S_half = S // 2
matQ1, matQ2 = torch.split(matQ, S_half, dim=2)
matK1, matK2 = torch.split(matK, S_half, dim=2)
matV1, matV2 = torch.split(matV, S_half, dim=2)
vecI1, vecI2 = torch.split(vecI, S_half, dim=2)
vecF1, vecF2 = torch.split(vecF, S_half, dim=2)

matH1, (C1, n1, m1) = mlstm_chunkwise__xl_chunk(
    q=matQ1, k=matK1, v=matV1, i=vecI1, f=vecF1, return_last_states=True, chunk_size=256
)

matH2, (C2, n2, m2) = mlstm_chunkwise__xl_chunk(
    q=matQ2, k=matK2, v=matV2, i=vecI2, f=vecF2,
    c_initial=C1, n_initial=n1, m_initial=m1,
    return_last_states=True, chunk_size=256
)

# Concatenate the outputs from the two parts
matH_stateful = torch.cat([matH1, matH2], dim=2)

# Compare the results
print("Checking correctness of hidden state passing...")
assert torch.allclose(matH_ref, matH_stateful, atol=1e-2, rtol=1e-2)
assert torch.allclose(C_ref, C2, atol=1e-2, rtol=1e-2)
assert torch.allclose(n_ref, n2, atol=1e-2, rtol=1e-2)
assert torch.allclose(m_ref, m2, atol=1e-2, rtol=1e-2)
print("Hidden state passing test successful!")


matH1 = mlstm_chunkwise__xl_chunk(
    q=matQ, k=matK, v=matV, i=vecI, f=vecF, return_last_states=False, chunk_size=256
)
print("Stateless execution successful!")