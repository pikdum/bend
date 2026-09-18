# AMD HIP backend

This fork adds a native HIP/HIPRTC target to Bend 2.0.4. The compiler selects
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
the GPU's reported architecture. Group count follows the existing L2-based
heuristic; it is not hardcoded to gfx1100 or claimed to be optimal.

HIP uses a VRAM corpus and a matching host mapping. The corpus contains indices,
so the two mappings need not have the same address. At an offload boundary the
runtime copies the corpus to VRAM. Intermediate scheduler passes keep it there;
the host reads the control header and clears the frontier cursor. The complete
corpus returns before CPU evaluation resumes, including on runtime errors.
Window rendering uploads the host corpus again because IO may have changed an
image after an offload.

The default heap is **2 GB**, with approximately the same amount of host memory
when fully touched. `--gpu 1GB` reduces transfer cost for modest programs, but
full-size hashmap needs more than 1 GB. The runtime's rings and stacks alone
consume roughly 400 MB. Full-corpus transfers are still expensive; batch useful
work into substantial offloads. Small iterative fields can remain faster on
CPU. `BEND_GPU_TRACE=1` reports actual passes and allocated/capacity pages.
A GPU-enabled binary can execute an ignored mark on CPU; require trace evidence
when measuring a new workload.

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

All 16 full-size programs matched Bend CPU output on two GPU executions.
The final corpus run used 2 GB throughout. No new kernel faults were recorded
after the heap-exhaustion repair.

| Program | Checksum | HIP passes |
|---|---:|---:|
| bfs | 651176970 | 1 |
| editdist | 2229810577 | 1 |
| gameoflife | 2016151040 | 1 |
| hashmap | 1307803744 | 1 |
| kmeans | 1616398086 | 160 |
| lexer | 2401049475 | 1 |
| mandelbrot | 3101455856 | 2 |
| merkle | 3104235417 | 3 |
| nbody | 3516450380 | 1 |
| queens | 2063750025 | 1 |
| raytrace | 1924309504 | 1 |
| symreg | 2383953211 | 1 |
| terrain | 2572468224 | 1 |
| tree-bitonic | 3787129428 | 210 |
| tree-matmul | 3797651056 | 19 |
| tree-radix | 1998173798 | 17 |

Additional checks: CPU-only native and JavaScript builds of the handoff test
both printed `host` and `60`. A four-frame window test compared every rendered
HIP pixel against the CPU's traversal of the same Bend image tree, including
host-created and shared images; all pixels matched. The current desktop display
had to be rediscovered after the earlier reset (`DISPLAY=:2`).
