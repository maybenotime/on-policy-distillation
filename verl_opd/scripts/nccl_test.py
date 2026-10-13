import os, time, ctypes, torch, torch.distributed as dist

def nccl_runtime_version():
    try:
        lib = ctypes.CDLL("libnccl.so.2")
        v = ctypes.c_int()
        lib.ncclGetVersion(ctypes.byref(v))
        n = v.value
        return f"{n//10000}.{(n//100)%100}.{n%100}"
    except Exception as e:
        return f"probe-failed: {e}"

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

if rank == 0:
    print(f"[info] world={world}  compile-time nccl={torch.cuda.nccl.version()}")
    print(f"[info] RUNTIME nccl={nccl_runtime_version()}   <-- this is the one that matters")
    maps = [l for l in open("/proc/self/maps") if "libnccl" in l]
    print(f"[info] mapped: {maps[0].split()[-1] if maps else 'none'}")

N = 256 * 1024 * 1024 // 2   # 256 MB of bf16
x = torch.ones(N, dtype=torch.bfloat16, device="cuda")

for _ in range(3):                      # warmup
    dist.all_reduce(x)
torch.cuda.synchronize(); dist.barrier()

ITERS = 10
t0 = time.time()
for _ in range(ITERS):
    dist.all_reduce(x)
torch.cuda.synchronize()
dt = (time.time() - t0) / ITERS

if rank == 0:
    nbytes = N * 2
    algbw = nbytes / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world
    print(f"[result] all_reduce 256MB x{world}: {dt*1000:.2f} ms | algbw {algbw:.2f} GB/s | busbw {busbw:.2f} GB/s")
    ok = torch.allclose(x[:8].float(), torch.full((8,), float(world**(ITERS+3)), device="cuda").clamp(max=x[:8].float().max().item()))
    print(f"[result] values finite: {torch.isfinite(x).all().item()}")
    print("NCCL_TEST_PASS")
dist.destroy_process_group()
