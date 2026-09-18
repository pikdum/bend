# AMD HIP backend

This fork adds a native HIP/HIPRTC target to Bend 2.0.5. The compiler selects
HIP on Linux when ROCm is present and CUDA is absent. `ROCM_PATH` defaults to
`/opt/rocm`; `CC` can select a Clang with C23 `#embed` support (tested: Clang
21.1.8). Bun runs the compiler. There is no CUDA translation layer.

```sh
export ROCM_PATH=/opt/rocm
bun bend2/main.ts tests/compile/hip_handoff.bend -o /tmp/handoff
BEND_GPU_TRACE=1 /tmp/handoff --gpu on --threads 1
# stdout: host, then 60; stderr: two actual HIP passes
/tmp/handoff --gpu off --threads 6
```

On nyarlathotep, the NixOS package supplies these dependencies and environment
variables through `bend`. It pins this fork's source, rather than applying a
patch stored in the machine configuration. The editable checkout is
`~/code/bend2`; the GitHub repository is <https://github.com/pikdum/bend>.

## Memory and execution

The Metal/CUDA scheduler, device program, allocator layout, and 32-bit atomic
operations remain shared. HIP compiles that device program with HIPRTC, using
the GPU's reported architecture. HIP runs the bag's full 128 groups: the
shared L2 heuristic gave the RX 7900 XT (84 CUs, 6 MB L2 under an 80 MB L3)
64, and 128 ran the sixteen benches 16% faster in sum (lexer 1.8x, raytrace
1.4x, tree-bitonic 0.8x). `BEND_GPU_GROUPS=16..128` overrides it.

HIP uses a VRAM corpus and a matching host mapping. The corpus contains indices,
so the two mappings need not have the same address. CUDA maps one managed
corpus with the device as its preferred location and lets pages migrate on
fault; RDNA3 consumer parts have no page migration, so HIP copies explicitly.
A handoff (`gpu_sync`) moves only what the other side reads: the control
header, the ring cursors, the static image, the heap up to the bump pointer
and each bank's entries. The lanes' allocator rows and stacks stay in VRAM
(the host has its own), and ring slots hold nothing at a handoff except the
task the host just pushed on ring 0; a ring found holding entries moves the
whole slot region instead. Intermediate scheduler passes keep everything
resident; the host reads the header and clears the frontier cursor. The live
corpus returns before CPU evaluation resumes, including on runtime errors.
Window rendering re-uploads the live corpus because IO may have changed an
image after an offload. The device corpus is zeroed once at startup, as the
host mapping is.

A two-boundary handoff test moves about 0.6 MB instead of 8 GB, and its GPU
run fell from 1.9 s to about 0.08 s. Transfer still scales with the heap's
bump pointer (pages ever allocated), not with the pages the host touches, so
a program that allocates a large heap and then crosses several boundaries
(tree-radix: 17 passes, 3.5 GB moved) pays for the whole used heap each time.
`BEND_GPU_TRACE=1` reports actual passes, allocated/capacity pages and the
cumulative bytes moved. A GPU-enabled binary can execute an ignored mark on
CPU; require trace evidence when measuring a new workload.

The default heap is **2 GB** of VRAM; host memory grows only with the pages a
handoff touches. `--gpu 1GB` shrinks the reservation, but full-size hashmap
needs more than 1 GB. The rings and stacks alone take roughly 400 MB of the
span.

Launching a HIP process within about a second of the previous one exiting
costs about 1.0 s of startup (0.08 s otherwise): the open of `/dev/kfd`
flushes the KFD release workqueue, and the previous process's release sits
in `amdgpu_ih_wait_on_checkpoint_process_ts` for its full one-second timeout
(function_graph trace, kernel 7.1.13). The IH v6.0 re-init after a GPU reset
zeroes the hardware pointers of interrupt ring 1 but not the driver's
software copies; ring 1 learns its write pointer only from a self interrupt,
so after a reset that cut off a stream of ring 1 page faults the two never
reconcile, and every KFD process release waits on them. A reboot clears the
state until the next fault-and-reset; the patch in
`nixos-config/nixos/pkgs/amdgpu-ih-v6-reset-sw-pointers.patch` resets the
software pointers with the hardware ones. The gate's second sample of each
run pays the penalty; the first sample is the representative one.

## Heap exhaustion repair

On RX 7900 XT, full-size hashmap with a 1 GB heap reproducibly triggered a GPU
page fault/reset before the repair. The upstream performance gate uses its 2 GB
default for this case. Reduced-batch allocation telemetry showed the growing
live heap, and the full case passed with 2 GB.

The shared allocator previously reported exhaustion and then continued through
page zero. Generated constructors and reference wrappers could keep accessing
that failed allocation. This fork returns immediately from failed allocation,
stops affected evaluation paths, and skips allocator-bank compaction after a
device error. Both the original full hashmap at 1 GB and an oversized array now
exit with `out of memory` and status 1 without a driver fault. The repair is in
the compiler/runtime, not the language core (`bend2/bend.ts` is unchanged).

This explains and repairs a reproduced native HIP failure. It does not establish
the cause of every earlier SCALE/driver fault or certify arbitrary GPU programs.
Metal and NVIDIA hardware were unavailable for cross-device execution checks.

The four HIP commits were rebased onto upstream 2.0.5 before packaging; the
CLI retains upstream CUDA_HOME/lib handling and compiler fallback.

## Local validation

`gates/hip.py` is a Linux gate, independent of the upstream private Mac cluster.
Use a Nix shell for Python/Bun/Clang/ROCm if they are not already available. It
requires readable `journalctl -k` output and uses a persistent STOPPED latch on
faults or timeouts. Inspect a failure before any further GPU execution. Never
run multiple GPU validation processes at once on the desktop GPU.

```sh
export BEND_HIP_RESULTS=/tmp/bend-hip-results
python3 gates/hip.py tests
python3 gates/hip.py bench
python3 gates/hip.py errors
```

`tests` compares the eleven compilable upstream offload-mark tests and two new
regressions with CPU output. Two upstream marks are intentionally inert. The
new cases require dispatch: a host-updated array crossing two offload boundaries,
and a checked 64x64 distance field spanning 131 scheduler passes. `bench` runs
all sixteen full-size upstream programs, requires real dispatch, and compares
each result with Bend CPU execution. `errors` checks controlled exhaustion using
an oversized array and full-size hashmap at 1 GB. Raw results and commands go to
JSONL files outside the checkout. `BEND_HIP_HEAP` overrides the successful-run
heap (default 2 GB); error tests deliberately use 1 GB.

Validation host: nyarlathotep, Ryzen 5 5600X / RX 7900 XT (gfx1100), NixOS,
ROCm 7.2.3, Clang 21.1.8. These are correctness runs with process wall times,
not a reproduction of Apple's kernel timing or a statistically controlled
performance comparison. The current backend remains experimental.

### Validated corpus (2026-09-17)

All 16 full-size programs matched Bend CPU output on two GPU executions, at
64 and at 128 groups, with 2 GB throughout. No new kernel faults were
recorded after the heap-exhaustion repair. Wall times are whole-process
seconds from the gate's first sample (the second pays the launch penalty
above): the CPU at 6 threads, then HIP before live-range handoffs (full
2 GB copies, 64 groups), after them at 64 groups, and at the 128 default.
Passes are at 64/128 groups; the checksums are identical at both.

| Program | Checksum | Passes | CPU | Full copy | Live, 64 | Live, 128 |
|---|---:|---:|---:|---:|---:|---:|
| bfs | 651176970 | 1 | 0.97 | 1.11 | 0.42 | 0.37 |
| editdist | 2229810577 | 1 | 0.63 | 1.55 | 0.89 | 0.89 |
| gameoflife | 2016151040 | 1 | 2.45 | 2.48 | 1.75 | 1.46 |
| hashmap | 1307803744 | 1 | 0.82 | 1.39 | 0.89 | 0.93 |
| kmeans | 1616398086 | 160/180 | 1.41 | 1.64 | 0.89 | 0.81 |
| lexer | 2401049475 | 1 | 0.81 | 3.87 | 3.10 | 1.69 |
| mandelbrot | 3101455856 | 2 | 1.20 | 1.18 | 0.44 | 0.38 |
| merkle | 3104235417 | 3 | 1.32 | 1.17 | 0.17 | 0.17 |
| nbody | 3516450380 | 1 | 1.48 | 0.85 | 0.10 | 0.11 |
| queens | 2063750025 | 1 | 2.32 | 2.16 | 1.35 | 1.22 |
| raytrace | 1924309504 | 1 | 1.93 | 2.38 | 1.57 | 1.11 |
| symreg | 2383953211 | 1 | 1.17 | 1.56 | 0.80 | 0.72 |
| terrain | 2572468224 | 1 | 0.82 | 1.29 | 0.52 | 0.53 |
| tree-bitonic | 3787129428 | 210/240 | 3.18 | 1.79 | 0.96 | 1.17 |
| tree-matmul | 3797651056 | 19 | 1.16 | 1.08 | 0.55 | 0.49 |
| tree-radix | 1998173798 | 17/18 | 1.38 | 1.59 | 0.95 | 0.87 |

The live handoffs moved 8 MB (queens) to 3.5 GB (tree-radix) per program in
place of 4 GB per boundary.

Additional checks: CPU-only native and JavaScript builds of the handoff test
both printed `host` and `60`. `BEND_WINDOW_CHECK=1` makes a GPU binary's
window frames compare every device pixel with the host's traversal of the
same image tree and report the count on stderr. `tests/gfx/window.bend` with
its 64x64 image built by a marked call, run on the desktop (`DISPLAY=:2`
since the earlier reset), rendered four frames with 0 of 3072 pixels
differing each, including the frame after the offload and the shared image.
