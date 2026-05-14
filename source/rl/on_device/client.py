#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson TX2 Client - FIXED VERSION for combined_tx2_fixed.py

This is the fixed client that works with:
- combined_tx2_fixed.py
- mambrl_d3qn_tx2_fixed.py
- mamfrl_d3qn_tx2_fixed.py

COMMUNICATION PROTOCOL:
=======================
Server sends:
{
    "applications": [
        {
            "id": 1,
            "benchmark": "fft",           # Benchmark name (routes to correct runner)
            "variant": "bin-omp-tasks",   # For BOTS, empty for PolyBench
            "input_arg": "5",             # Input argument
            "app_args": "5",              # Same as input_arg (for compatibility)
            "cores": "1,2,3",             # Comma-separated core list
            "frequencies": [6, 6, 6],     # Frequency indices per core
            "priority": 80,               # RT priority
            "action": "run"               # "profile" or "run"
        }
    ],
    "run_mode": "parallel"  # or "sequential"
}

Client responds:
{
    "profiling_data_list": [
        {
            "application_id": 1,
            "benchmark": "fft",
            "variant": "bin-omp-tasks",
            "time_elapsed": 1.234,
            "total_energy_consumption": 5.67,
            "thermal_zone0": 45.0,
            ...
        }
    ]
}

SUPPORTED BENCHMARKS:
====================
BOTS (omptasks): alignment, fft, fib, floorplan, health, concom, knapsack,
                 nqueens, sort, sparselu, strassen, uts
PolyBench:       gemm, gemver, 2mm, 3mm, jacobi-2d, heat-3d, syrk, etc.

TX2 SPECIFICS:
==============
- NUM_LOGICAL_CORES = 6 (CPUs 0..5)
- Power rails: SYSTEM/MAIN, CPU (big), DENVER, GPU, DDR from iio:device*
- CPU cpufreq/online paths adapted to TX2
"""

import os
import re
import json
import time
import socket
import argparse
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Defaults / Paths
# =============================================================================
IP_ADDRESS_DEFAULT = "<JETSON_IP>"
PORT_DEFAULT       = 8707

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

OMPTASKS_RUN_DIR = os.path.join(REPO_ROOT, "omptasks", "run")
OMPTASKS_BOTS    = os.path.join(OMPTASKS_RUN_DIR, "bots.sh")

POLYTASKS_DIR    = os.path.join(REPO_ROOT, "polytasks")
POLYTASKS_RUN    = os.path.join(POLYTASKS_DIR, "run_bench.sh")

REAL_TIME_TASK_DEFAULT = os.path.join(THIS_DIR, "real_time_task")

# =============================================================================
# Power sources (TX2 iio rails)
# =============================================================================
IIO_SYS = "/sys/bus/iio/devices/iio:device1"
IIO_GPU = "/sys/bus/iio/devices/iio:device0"
IIO_DDR = "/sys/bus/iio/devices/iio:device2"

SYSTEM_V = f"{IIO_SYS}/in_voltage0_input"
SYSTEM_A = f"{IIO_SYS}/in_current0_input"
SYSTEM_W = f"{IIO_SYS}/in_power0_input"

# Aliases for backward compatibility
MAIN_V = SYSTEM_V
MAIN_A = SYSTEM_A
MAIN_W = SYSTEM_W

CPU_V = f"{IIO_SYS}/in_voltage1_input"
CPU_A = f"{IIO_SYS}/in_current1_input"
CPU_W = f"{IIO_SYS}/in_power1_input"

DENVER_V = f"{IIO_SYS}/in_voltage2_input"
DENVER_A = f"{IIO_SYS}/in_current2_input"
DENVER_W = f"{IIO_SYS}/in_power2_input"

GPU_W = f"{IIO_GPU}/in_power0_input"
DDR_W = f"{IIO_DDR}/in_power0_input"

# =============================================================================
# CPU sysfs (TX2 per-CPU)
# =============================================================================
CPU_GOVERNOR   = "/sys/devices/system/cpu/cpu{idx}/cpufreq/scaling_governor"
CPU_SETSPEED   = "/sys/devices/system/cpu/cpu{idx}/cpufreq/scaling_setspeed"
CPU_AVAIL_FREQ = "/sys/devices/system/cpu/cpu{idx}/cpufreq/scaling_available_frequencies"
CPU_CUR_FREQ   = "/sys/devices/system/cpu/cpu{idx}/cpufreq/scaling_cur_freq"
CPU_ONLINE     = "/sys/devices/system/cpu/cpu{idx}/online"
CPU_MIN        = "/sys/devices/system/cpu/cpu{idx}/cpufreq/scaling_min_freq"
CPU_MAX        = "/sys/devices/system/cpu/cpu{idx}/cpufreq/scaling_max_freq"

NUM_LOGICAL_CORES = 6
DEFAULT_CORESET = list(range(1, NUM_LOGICAL_CORES))  # exclude core 0 by default: [1..5]

THERMAL_DIR = "/sys/devices/virtual/thermal"

# =============================================================================
# Benchmark families (same as RubikPi client)
# =============================================================================
OMPTASKS_BENCHES = {
    "alignment", "fft", "fib", "floorplan", "health",
    "concom", "knapsack", "nqueens", "sort", "sparselu",
    "strassen", "uts",
}

POLYBENCH_BENCHES = {
    "gemm", "gemver", "gesummv", "symm", "syr2k", "syrk", "trmm",
    "2mm", "3mm", "atax", "bicg", "doitgen", "mvt",
    "cholesky", "durbin", "gramschmidt", "lu", "ludcmp", "trisolv",
    "correlation", "covariance", "deriche", "floyd-warshall", "nussinov",
    "adi", "fdtd-2d", "heat-3d", "jacobi-1d", "jacobi-2d", "seidel-2d",
}

# =============================================================================
# Small utils
# =============================================================================
def read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def write_str(path: str, val: str) -> bool:
    try:
        with open(path, "w") as f:
            f.write(val)
        return True
    except Exception as e:
        logging.debug(f"Write failed {path}: {e}")
        return False

def ensure_executable(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    st = os.stat(path)
    if not (st.st_mode & 0o111):
        try:
            os.chmod(path, st.st_mode | 0o111)
        except Exception:
            pass

def ensure_core_online(i: int) -> bool:
    online_path = CPU_ONLINE.format(idx=i)
    if not os.path.exists(online_path):
        return True
    cur = read_int(online_path)
    if cur == 1:
        return True
    ok = write_str(online_path, "1")
    if ok:
        logging.info(f"[cpu{i}] online=1")
    else:
        logging.warning(f"[cpu{i}] failed to set online=1")
    return ok

def ensure_cores_online(cores: Sequence[int]):
    for i in sorted(set(c for c in cores if 0 <= c < NUM_LOGICAL_CORES)):
        ensure_core_online(i)

def set_userspace_governor(cores: Optional[Sequence[int]] = None):
    target = list(range(NUM_LOGICAL_CORES)) if not cores else list(cores)
    for i in target:
        ok = write_str(CPU_GOVERNOR.format(idx=i), "userspace")
        if ok:
            logging.info(f"[cpu{i}] governor->userspace")
        else:
            logging.warning(f"[cpu{i}] governor set failed")

def get_available_freqs_for_cpu(cpu_idx: int) -> List[int]:
    p = CPU_AVAIL_FREQ.format(idx=cpu_idx)
    try:
        with open(p, "r") as f:
            arr = f.read().strip().split()
            return [int(x) for x in arr if x.isdigit()]
    except Exception:
        return []

def get_available_freq_indices() -> List[int]:
    freqs = get_available_freqs_for_cpu(0)
    return list(range(len(freqs))) if freqs else []

def _set_bounds_if_needed(i: int, target: int):
    min_p = CPU_MIN.format(idx=i)
    max_p = CPU_MAX.format(idx=i)
    try:
        cur_min = read_int(min_p)
        cur_max = read_int(max_p)
        if cur_min is not None and target < cur_min:
            write_str(min_p, str(target))
        if cur_max is not None and target > cur_max:
            write_str(max_p, str(target))
    except Exception:
        pass

def _parse_cores_str(cores_str: str) -> List[int]:
    ids = []
    for tok in str(cores_str).split(","):
        tok = tok.strip()
        if tok.isdigit():
            v = int(tok)
            if 0 <= v < NUM_LOGICAL_CORES:
                ids.append(v)
    return ids if ids else DEFAULT_CORESET[:]

def set_cpu_freqs_assigned(levels: Sequence[int], cores_subset: Optional[List[int]] = None):
    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        logging.info("No assigned frequency levels; skip setspeed.")
        return

    if len(levels) == NUM_LOGICAL_CORES:
        target_cores = list(range(NUM_LOGICAL_CORES))
        per_core_idx = {c: int(levels[c]) for c in target_cores}
    else:
        selected = cores_subset[:] if cores_subset else DEFAULT_CORESET[:]
        target_cores = selected
        if len(levels) == 1:
            per_core_idx = {c: int(levels[0]) for c in selected}
        elif len(levels) == len(selected):
            per_core_idx = {c: int(levels[i]) for i, c in enumerate(selected)}
        else:
            logging.warning(f"Level vector len={len(levels)} mismatch; using first for selected cores.")
            per_core_idx = {c: int(levels[0]) for c in selected}

    ensure_cores_online(target_cores)
    set_userspace_governor(target_cores)

    for i in target_cores:
        idx = per_core_idx.get(i)
        if idx is None:
            continue

        avail_i = get_available_freqs_for_cpu(i)
        if not avail_i:
            logging.warning(f"[cpu{i}] no available freqs; skip")
            continue

        if idx < 0:
            idx = 0
        if idx >= len(avail_i):
            idx = len(avail_i) - 1

        target = int(avail_i[idx])
        _set_bounds_if_needed(i, target)
        if write_str(CPU_SETSPEED.format(idx=i), str(target)):
            logging.info(f"[cpu{i}] setspeed -> {target} Hz (level {idx})")
        else:
            logging.warning(f"[cpu{i}] setspeed failed (need sudo/userspace?)")

def read_cur_freqs_all_cores() -> List[int]:
    vals: List[int] = []
    for i in range(NUM_LOGICAL_CORES):
        v = read_int(CPU_CUR_FREQ.format(idx=i))
        vals.append(int(v) if v is not None else 0)
    return vals

# =============================================================================
# Power sampling — CLUSTER of sources (per-source W and integrated J)
# =============================================================================
def _read_power_w(path_w: str, path_v: Optional[str] = None, path_a: Optional[str] = None) -> float:
    """
    Return instantaneous power in Watts.
    TX2 iio power files (in_power*_input) are typically in microwatts; if absent, use mV×mA.
    """
    if path_w and os.path.exists(path_w):
        uw = read_int(path_w)
        if uw is not None:
            return max(0.0, float(uw) / 1_000_000.0)
    if path_v and path_a and os.path.exists(path_v) and os.path.exists(path_a):
        mv = read_int(path_v)
        ma = read_int(path_a)
        if mv is not None and ma is not None:
            return max(0.0, (mv * ma) / 1_000_000.0)
    return 0.0

def _read_power_sources_w() -> Dict[str, float]:
    """
    Read TX2 rails and return instantaneous power (W) by source name.
    """
    out: Dict[str, float] = {}
    out["system"] = _read_power_w(SYSTEM_W, SYSTEM_V, SYSTEM_A)
    out["main"]   = _read_power_w(MAIN_W, MAIN_V, MAIN_A)         # alias of system rail
    out["cpu"]    = _read_power_w(CPU_W, CPU_V, CPU_A)
    out["denver"] = _read_power_w(DENVER_W, DENVER_V, DENVER_A)
    out["gpu"]    = _read_power_w(GPU_W, None, None)
    out["ddr"]    = _read_power_w(DDR_W, None, None)
    return out

class PowerSampler(threading.Thread):
    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.energy_by_source: Dict[str, float] = {}   # Joules per source
        self.last_power_by_source: Dict[str, float] = {}
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        last_time = time.time()
        while not self._stop_evt.is_set():
            now = time.time()
            dt = now - last_time
            last_time = now

            powers = _read_power_sources_w()  # per-source W
            self.last_power_by_source = powers

            for src, w in powers.items():
                if src not in self.energy_by_source:
                    self.energy_by_source[src] = 0.0
                self.energy_by_source[src] += max(0.0, float(w)) * dt

            time.sleep(self.interval)

# =============================================================================
# Thermal — send all zones instead of average
# =============================================================================
def collect_thermal_zones_flat() -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        for name in sorted(os.listdir(THERMAL_DIR)):
            if not name.startswith("thermal_zone"):
                continue
            zpath = os.path.join(THERMAL_DIR, name)
            t_raw = read_int(os.path.join(zpath, "temp"))
            t_c = (t_raw / 1000.0) if t_raw is not None else None
            try:
                with open(os.path.join(zpath, "type"), "r") as f:
                    t_type = f.read().strip()
            except Exception:
                t_type = ""
            out[name] = float(t_c) if t_c is not None else None
            out[f"{name}_type"] = t_type
    except Exception:
        pass
    return out

def collect_avg_temp() -> float:
    temps = []
    try:
        for name in os.listdir(THERMAL_DIR):
            if not name.startswith("thermal_zone"):
                continue
            val = read_int(os.path.join(THERMAL_DIR, name, "temp"))
            if val is not None:
                temps.append(val / 1000.0)
    except Exception:
        pass
    return (sum(temps) / len(temps)) if temps else 0.0

# =============================================================================
# Perf parsing
# =============================================================================
PERF_COUNTER = re.compile(r"^([\d,\.]+)\s+(?:(\w+)\s+)?([A-Za-z0-9_\-:]+)\s*(?:#.*)?$")
TIME_LINE = re.compile(r"^([\d\.]+)\s+seconds\s+(time elapsed|user|sys)$", re.IGNORECASE)

DESIRED_PERF = {
    "cycles", "cache_references", "cache_misses", "branch_instructions", "task_clock",
    "context_switches", "minor_faults", "major_faults", "branch_misses", "branches",
    "instructions", "page_faults", "cpu_clock"
}

def _normalize_metric(tok: str) -> str:
    if not tok:
        return ""
    tok = re.sub(r":[A-Za-z]+$", "", tok)   # strip suffix like ':u'
    tok = tok.replace("-", "_").lower()
    return tok

def parse_perf_output(txt: str) -> Dict[str, float]:
    out: Dict[str, float] = {"time_elapsed": 0.0, "branch_misses": 0, "cache_misses": 0}
    in_perf = False
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if not in_perf and s.startswith("Performance counter stats for"):
            in_perf = True
            continue
        if in_perf:
            mtime = TIME_LINE.match(s)
            if mtime:
                val = float(mtime.group(1))
                kind = mtime.group(2).lower().replace(" ", "_")
                out[kind] = val
                if kind == "time_elapsed":
                    out["time_elapsed"] = val
                continue
            m = PERF_COUNTER.match(s)
            if m:
                value_str = (m.group(1) or "").replace(",", "")
                unit_tok  = (m.group(2) or "").lower()
                metric    = _normalize_metric(m.group(3) or "")
                if not metric:
                    continue
                try:
                    if unit_tok.startswith("msec"):
                        val = float(value_str)
                    else:
                        fv = float(value_str)
                        val = int(fv) if fv.is_integer() else fv
                except Exception:
                    try:
                        val = float(value_str)
                    except Exception:
                        val = 0
                if metric in DESIRED_PERF:
                    out[metric] = val
                if metric == "branch_misses":
                    out["branch_misses"] = int(val) if isinstance(val, int) else int(float(val))
                if metric == "cache_misses":
                    out["cache_misses"] = int(val) if isinstance(val, int) else int(float(val))
    return out

# =============================================================================
# Invocation mapping (same logic as RubikPi client)
# =============================================================================
def build_invocation(benchmark: str, variant: str, dataset_or_input: str) -> Tuple[str, List[str]]:
    """
    Route by benchmark family first (name-based), never just by file presence.
    """
    b = (benchmark or "").lower()

    # Family match
    if b in OMPTASKS_BENCHES:
        ensure_executable(OMPTASKS_BOTS)
        return (OMPTASKS_BOTS, [benchmark, variant, dataset_or_input])

    if b in POLYBENCH_BENCHES:
        ensure_executable(POLYTASKS_RUN)
        return (POLYTASKS_RUN, [benchmark, dataset_or_input])

    # Fallbacks
    run_script = os.path.join(OMPTASKS_RUN_DIR, f"run-{benchmark}.sh")
    if os.path.exists(run_script):
        ensure_executable(OMPTASKS_BOTS)
        return (OMPTASKS_BOTS, [benchmark, variant, dataset_or_input])

    if os.path.exists(POLYTASKS_RUN):
        ensure_executable(POLYTASKS_RUN)
        return (POLYTASKS_RUN, [benchmark, dataset_or_input])

    raise FileNotFoundError(f"Unknown benchmark '{benchmark}' and no suitable runner found.")

# =============================================================================
# Launcher (real_time_task like RubikPi client — no sudo; capture stderr)
# =============================================================================
def run_with_real_time_task(real_time_task: str,
                            app_path: str,
                            cores_csv: str,
                            app_args: List[str],
                            rt_priority: int,
                            num_cores_env: Optional[int] = None) -> Tuple[str, int]:
    """
    Execute:
      real_time_task <app_path> <rt_priority> <cores_csv> [app_args...]
    """
    ensure_executable(real_time_task)
    ensure_executable(app_path)

    cmd = [real_time_task, app_path, str(int(rt_priority)), str(cores_csv)] + list(app_args)
    env = os.environ.copy()
    if num_cores_env and num_cores_env > 0:
        env["OMP_NUM_THREADS"] = str(int(num_cores_env))
    env.setdefault("LC_ALL", "C")

    logging.info(f"exec: {' '.join(cmd)}  (OMP_NUM_THREADS={env.get('OMP_NUM_THREADS','inherit')})")
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        combined = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        if res.returncode != 0:
            logging.error(f"real_time_task failed ({res.returncode}).\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}")
        return (combined, res.returncode)
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        return ("", 1)

# =============================================================================
# Worker
# =============================================================================
def application_worker(app, results_list, lock, real_time_task_path: str):
    app_id     = app.get("id")
    benchmark  = app.get("benchmark", "")
    variant    = app.get("variant", "")
    input_arg  = str(app.get("input_arg", ""))
    assigned   = app.get("frequencies", [])
    cores_str  = app.get("cores", "")
    num_cores  = int(app.get("num_cores", 0) or 0)
    rt_prio    = int(app.get("rt_priority", 80) or 80)

    selected_cores = _parse_cores_str(cores_str)
    if num_cores > 0 and len(selected_cores) > num_cores:
        selected_cores = selected_cores[:num_cores]

    try:
        if isinstance(assigned, list) and len(assigned) > 0:
            set_cpu_freqs_assigned(assigned, selected_cores)
        else:
            ensure_cores_online(selected_cores)
            set_userspace_governor(selected_cores)
    except Exception as e:
        logging.warning(f"DVFS/online/governor setup failed: {e}")

    try:
        app_path, app_args = build_invocation(benchmark, variant, input_arg)
    except Exception as e:
        logging.error(f"Invocation mapping failed: {e}")
        return

    sampler = PowerSampler(interval=0.5)
    sampler.start()

    t0 = time.time()
    output, rc = run_with_real_time_task(
        real_time_task=real_time_task_path,
        app_path=app_path,
        cores_csv=",".join(str(c) for c in selected_cores),
        app_args=app_args,
        rt_priority=rt_prio,
        num_cores_env=len(selected_cores) if num_cores <= 0 else num_cores
    )
    t1 = time.time()

    sampler.stop()
    sampler.join(timeout=2.0)

    perf = parse_perf_output(output)
    if perf.get("time_elapsed", 0.0) <= 0:
        perf["time_elapsed"] = max(0.0, t1 - t0)

    freqs_read = read_cur_freqs_all_cores()

    # Build result: per-source energy + last power (plus total for compatibility)
    energy = sampler.energy_by_source
    lastpw = sampler.last_power_by_source
    total_energy = float(sum(energy.values())) if energy else 0.0

    result = {
        "application_id": app_id,
        "benchmark": benchmark,
        "variant": variant,
        "input_arg": input_arg,
        "frequencies": assigned,
        "frequencies_read": freqs_read,
        "cores": ",".join(str(c) for c in selected_cores),
        "time_elapsed": float(perf.get("time_elapsed", 0.0)),
        "branch_misses": int(perf.get("branch_misses", 0) or 0),
        "cache_misses": int(perf.get("cache_misses", 0) or 0),
        "total_energy_consumption": total_energy,  # sum of rails (compat)
    }

    # Flatten per-source energy (J) and last power (W)
    for src in ("system", "main", "cpu", "denver", "gpu", "ddr"):
        if src in energy:
            result[f"energy_{src}_j"] = float(energy.get(src, 0.0))
        if src in lastpw:
            result[f"power_{src}_w"] = float(lastpw.get(src, 0.0))

    # Thermal zones: thermal_zoneN + thermal_zoneN_type
    result.update(collect_thermal_zones_flat())

    # Merge any remaining perf counters (cycles, cache_references, etc.)
    for k, v in perf.items():
        if k not in result:
            result[k] = v

    with lock:
        results_list.append(result)

# =============================================================================
# Networking
# =============================================================================
def connect_to_server(ip: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    logging.info(f"Connecting to {ip}:{port}")
    while True:
        try:
            s.connect((ip, port))
            logging.info("Connected.")
            return s
        except Exception:
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--IP_ADDR", type=str, default=IP_ADDRESS_DEFAULT)
    parser.add_argument("--PORT", type=int, default=PORT_DEFAULT)
    parser.add_argument("--REAL_TIME_TASK", type=str, default=REAL_TIME_TASK_DEFAULT,
                        help="Path to the real_time_task launcher script/binary")
    args = parser.parse_args()

    try:
        ensure_executable(args.REAL_TIME_TASK)
    except Exception as e:
        logging.warning(f"real_time_task not executable yet: {e}")

    sock = connect_to_server(args.IP_ADDR, args.PORT)

    recv_buffer = ""
    while True:
        try:
            data = sock.recv(4096)
            if not data:
                time.sleep(0.2)
                continue
            recv_buffer += data.decode()

            while "\n" in recv_buffer:
                raw, recv_buffer = recv_buffer.split("\n", 1)
                if not raw.strip():
                    continue

                logging.info(f"Action: {raw}")
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logging.error("Invalid JSON from server.")
                    continue

                # CAPABILITIES
                if "control" in msg:
                    ctrl = msg.get("control", {})
                    if ctrl.get("action") == "profile_caps":
                        caps = {"available_freq_indices": get_available_freq_indices()}
                        # Include initial thermal snapshot so server can seed thermal_zone_before
                        caps.update(collect_thermal_zones_flat())
                        sock.send((json.dumps({"caps": caps}) + "\n").encode())
                        logging.info("Sent caps (freq indices + initial thermals).")
                        continue

                # Ignore server keep-alives/acks without warning
                if "status" in msg and "applications" not in msg:
                    continue

                apps = msg.get("applications", [])
                if not apps:
                    if "control" not in msg and "status" not in msg:
                        logging.warning("No applications in message.")
                    continue

                run_mode = msg.get("run_mode", "sequential")
                results: List[Dict] = []
                lock = threading.Lock()

                if run_mode == "parallel" and len(apps) > 1:
                    threads = []
                    for app in apps:
                        t = threading.Thread(
                            target=application_worker,
                            args=(app, results, lock, args.REAL_TIME_TASK),
                            daemon=True
                        )
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                else:
                    for app in apps:
                        application_worker(app, results, lock, args.REAL_TIME_TASK)

                out = {"profiling_data_list": results}
                sock.send((json.dumps(out) + "\n").encode())
                logging.info(f"Sent {len(results)} result(s).")

        except (BrokenPipeError, ConnectionResetError):
            logging.error("Server connection lost. Reconnecting...")
            try:
                sock.close()
            except Exception:
                pass
            sock = connect_to_server(args.IP_ADDR, args.PORT)
        except KeyboardInterrupt:
            logging.info("Client interrupted. Exiting.")
            break
        except Exception as e:
            logging.error(f"Client loop error: {e}")
            time.sleep(1)

    try:
        sock.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
