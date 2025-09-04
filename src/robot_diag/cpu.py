import math
import multiprocessing as mp
import time
from typing import Any, Dict

from .utils import cpu_freqs, sample_thermal


def _cpu_worker(duration: int) -> Dict[str, Any]:
    start = time.time()
    ops = 0
    n = 2
    # simple prime-testing workload
    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x % 2 == 0:
            return x == 2
        r = int(math.sqrt(x))
        i = 3
        while i <= r:
            if x % i == 0:
                return False
            i += 2
        return True

    while time.time() - start < duration:
        ops += 1
        is_prime(n)
        n += 1
    return {"ops": ops}


def stress(duration_sec: int = 20) -> Dict[str, Any]:
    cores = max(1, mp.cpu_count())
    with mp.Pool(processes=cores) as pool:
        t0 = time.time()
        freqs0 = cpu_freqs()
        therm0 = sample_thermal()
        res = pool.map(_cpu_worker, [duration_sec] * cores)
        t1 = time.time()
        freqs1 = cpu_freqs()
        therm1 = sample_thermal()

    total_ops = sum(r["ops"] for r in res)
    return {
        "cores": cores,
        "duration_sec": duration_sec,
        "total_ops": total_ops,
        "ops_per_sec": total_ops / max(1e-9, (t1 - t0)),
        "freqs_before": freqs0,
        "freqs_after": freqs1,
        "thermal_before": therm0,
        "thermal_after": therm1,
    }

