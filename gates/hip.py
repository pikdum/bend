#!/usr/bin/env python3
# Local Linux HIP gate. Requires bun, clang, ROCm/HIPRTC and readable kernel logs.
# Fails closed on driver faults and timeouts; retains a STOPPED latch for review.
import json, os, pathlib, re, signal, statistics, subprocess, sys, time

OUT = pathlib.Path(os.environ.get('BEND_HIP_RESULTS', '/tmp/bend-hip-results'))
OUT.mkdir(parents=True, exist_ok=True)
ROOT = pathlib.Path(__file__).resolve().parent.parent
BEND = [os.environ.get('BUN', 'bun'), str(ROOT/'bend2/main.ts')]
BASE = os.environ.copy()
BASE.pop('LD_PRELOAD', None)
BASE['BEND_GPU_TRACE'] = '1'
FAULT = re.compile(r'sq_intr: error|amdgpu.*(?:fault|reset|timeout|wedged|unrecoverable)|GPU.*(?:fault|reset|wedged)|ring .*timeout', re.I)
CURSOR = None

def journal():
    global CURSOR
    args = ['journalctl', '-k', '--no-pager', '--show-cursor']
    args += ['--after-cursor', CURSOR] if CURSOR else ['-n', '0']
    r = subprocess.run(args, capture_output=True, text=True, timeout=3)
    if r.returncode:
        raise RuntimeError('Cannot monitor kernel journal: ' + r.stderr)
    m = re.search(r'-- cursor: (.+)', r.stdout)
    if m:
        CURSOR = m[1]
    return '\n'.join(l for l in r.stdout.splitlines() if FAULT.search(l))

def abort(reason):
    (OUT / 'STOPPED').write_text(reason)
    raise RuntimeError(reason)

def run(cmd, env=BASE, limit=30, gpu=False, expected_exit=0):
    if (OUT / 'STOPPED').exists():
        raise RuntimeError('Runner latched stopped; inspect STOPPED before any further GPU work')
    fault = journal()
    if fault:
        abort(fault)
    start = time.perf_counter()
    p = subprocess.Popen([str(x) for x in cmd], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    reason = None
    while True:
        try:
            o, e = p.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            fault = journal()
            if fault or time.perf_counter() - start > limit:
                reason = fault or f'Timeout after {limit}s: {cmd}'
                os.killpg(p.pid, signal.SIGKILL)
                o, e = p.communicate()
                break
    elapsed = time.perf_counter() - start
    fault = journal()
    r = {'exit': p.returncode, 'seconds': elapsed, 'stdout': o.strip(), 'stderr': e.strip()}
    m = re.search(r'BEND_EXEC_SECONDS=([0-9.]+)', e)
    if m:
        r['execution_seconds'] = float(m[1])
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'command': [str(x) for x in cmd], **r, 'fault': fault, 'reason': reason}) + '\n')
    if fault or reason or (gpu and (p.returncode != expected_exit or re.search(r'memory fault|GPU trap|illegal instruction', e, re.I))):
        abort(fault or reason or e or repr(r))
    if p.returncode != expected_exit:
        raise RuntimeError(r)
    return r


def check(name, source, expected=None, passes=False):
    binary = OUT/name
    run(BEND + [source, '-o', binary], limit=60, gpu=True)
    cpu = run([binary, '--gpu', 'off', '--threads', '6'], limit=30)
    if expected is not None and cpu['stdout'] != expected:
        abort(f'CPU mismatch {name}: {cpu}')
    samples = []
    for rep in range(2):
        result = run([binary, '--gpu', os.environ.get('BEND_HIP_HEAP', '2GB'), '--threads', '1'], limit=8, gpu=True)
        if result['stdout'] != cpu['stdout']:
            abort(f'HIP mismatch {name}: {result}')
        count = len(re.findall(r'bend: HIP pass ', result['stderr']))
        if passes and not count:
            abort(f'No HIP dispatch in {name}')
        samples.append({'seconds': result['seconds'], 'passes': count})
    row = {'name': name, 'cpu_seconds': cpu['seconds'], 'stdout': cpu['stdout'], 'hip': samples}
    with (OUT/'results.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv)>1 else 'tests'
    if mode == 'tests':
        for p in sorted((ROOT/'tests').rglob('*.bend')):
            src = p.read_text()
            if not re.search(r'\w!\(', src) or 'import Base' not in src:
                continue
            if not re.search(r'^def main\(\)', src, re.M):
                continue
            expected = '\n'.join(l[2:] for l in src.splitlines() if l.startswith('#|')).strip()
            if expected.startswith('Error:'):
                continue
            check(p.parent.name+'_'+p.stem, p, expected, passes=p.stem.startswith('hip_'))
    elif mode == 'errors':
        source = OUT/'oversize.bend'
        source.write_text("""import Base

def make(n: Nat) -> Array<U32>:
  [0: U32 ^ n]

def result(r: Array<U32> & U32) -> U32:
  (a, n) = r
  n

def size(a: Array<U32>) -> U32:
  result(Array.size(U32, a))

def main() -> IO(Unit):
  IO.print(U32.show(size(make!(30n))))
""")
        for name, source in [('oversize',source),
                             ('hashmap_oom',ROOT/'bench/runtime/hashmap/main.bend')]:
            binary = OUT/name
            run(BEND+[source,'-o',binary],limit=60,gpu=True)
            r = run([binary,'--gpu','1GB','--threads','1'],limit=8,gpu=True,expected_exit=1)
            if 'out of memory' not in r['stderr']:
                abort(f'Expected controlled heap exhaustion: {r}')
            print(name+': controlled heap exhaustion',flush=True)
    elif mode == 'bench':
        for p in sorted((ROOT/'bench/runtime').glob('*/main.bend')):
            if len(sys.argv)>2 and p.parent.name not in sys.argv[2:]:
                continue
            check(p.parent.name, p, passes=True)
    else:
        for arg in sys.argv[1:]:
            p = pathlib.Path(arg).resolve()
            check(p.stem, p, passes=True)

if __name__ == '__main__':
    main()
