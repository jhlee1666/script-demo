#!/usr/bin/env python3
"""
KV Cache 저장·복원 vs 재계산 벤치마크.

llama-server의 슬롯 저장/복원 API로 KV Cache를 SSD에 저장했다가 다시 불러오고,
프롬프트를 처음부터 다시 처리(재계산)하는 경우와 시간을 비교한다.

조건 하나(모델, 길이 N, 반복 r)마다:
  1. 재계산      : 슬롯 비우기 -> 프롬프트 처리 (재계산 시간)
  2. RAW 저장    : 슬롯 저장 -> fsync (실제 디스크 쓰기까지)
  3. 기준 출력 A : 저장 직후의 KV 상태에서 생성
  4. RAW 복원    : cold(캐시 비움) / warm 각각 복원 -> 같은 요청 (출력 B, A와 비교)
  5. LZ4 저장    : RAW 파일을 LZ4로 압축 -> fsync, RAW 삭제
  6. LZ4 복원    : cold / warm 각각 압축 해제 -> 복원 -> 같은 요청

A와 B는 "KV 상태 + 마지막 토큰 1개 계산 + 생성"이라는 같은 경로를 거치므로,
복원이 정확하면 출력이 비트 단위로 같아야 한다. (프롬프트 전체를 다시 처리한
결과와 비교하면 배치 크기 차이로 생기는 미세한 수치 차이 때문에 다를 수 있다.)
결과는 조건마다 CSV 한 줄씩 바로 기록한다(중간에 멈춰도 그때까지 남음).

예시 (짧은 시험):
  ~/aiwork/venv/bin/python scripts/run_bench.py --model qwen --lengths 256,1024 --repeats 1
"""

import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "llama": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}

CSV_FIELDS = [
    "model", "n_tokens", "rep", "storage", "cache",
    "recompute_prompt_n", "recompute_ms",
    "save_ms", "fsync_ms", "raw_bytes",
    "lz4_bytes", "compress_ms", "decompress_ms",
    "restore_ms", "n_restored",
    "resume_prompt_n", "resume_prompt_ms", "resume_total_ms",
    "speedup", "output_match",
    "t_start", "t_end",
]


def log(msg):
    now = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def ms_since(t0):
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------- HTTP

class Server:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def _req(self, method, path, body=None, timeout=3600):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}")

    def healthy(self):
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def tokenize(self, text):
        return self._req("POST", "/tokenize", {"content": text})["tokens"]

    def completion(self, tokens, n_predict):
        return self._req("POST", "/completion", {
            "prompt": tokens, "n_predict": n_predict, "temperature": 0,
            "cache_prompt": True, "id_slot": 0,
        })

    def slot(self, action, filename=None):
        body = {"filename": filename} if filename else {}
        res = self._req("POST", f"/slots/0?action={action}", body)
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"slot {action} 실패: {res['error']}")
        return res


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server(args, model_path, raw_dir, out_dir, tag):
    cmd = [
        str(Path(args.llama_dir) / "build/bin/llama-server"),
        "-m", str(model_path), "-t", str(args.threads), "-c", str(args.ctx),
        "-np", "1", "--host", "127.0.0.1", "--port", str(args.port),
        "--slot-save-path", str(raw_dir) + "/",
    ]
    env = os.environ.copy()
    if args.mem_limit:
        # cgroup v2로 llama-server 하나만 메모리 제한
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        cmd = ["systemd-run", "--user", "--scope", "--quiet",
               "-p", f"MemoryMax={args.mem_limit}",
               "-p", f"MemorySwapMax={args.swap_max}"] + cmd
    log_f = open(out_dir / f"server_{tag}.log", "w")
    log(f"서버 시작: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    srv = Server(args.port)
    t0 = time.time()
    while time.time() - t0 < args.server_timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"서버가 바로 종료됨 (exit {proc.returncode}). {log_f.name} 확인")
        if srv.healthy():
            log(f"서버 준비 완료 ({time.time() - t0:.1f}s)")
            return proc, srv, log_f
        time.sleep(1)
    raise RuntimeError("서버 준비 시간 초과")


def stop_server(proc, log_f):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    if log_f:
        log_f.close()


# ---------------------------------------------------------------- 파일, 캐시

def fsync_file(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def evict(path, method):
    """cold 조건 만들기: 파일이 RAM(페이지 캐시)에 남지 않게 한다."""
    if method == "drop":
        subprocess.run(["sudo", "-n", "/usr/local/sbin/kvb-drop-caches"], check=True)
    else:  # fadvise: 그 파일만 캐시에서 내림 (sudo 불필요)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def lz4_compress(src, dst):
    import lz4.frame
    t0 = time.perf_counter()
    with open(src, "rb") as f:
        data = f.read()
    comp = lz4.frame.compress(data)
    with open(dst, "wb") as f:
        f.write(comp)
        f.flush()
        os.fsync(f.fileno())
    return ms_since(t0), len(comp)


def lz4_decompress(src, dst):
    import lz4.frame
    t0 = time.perf_counter()
    with open(src, "rb") as f:
        comp = f.read()
    data = lz4.frame.decompress(comp)
    with open(dst, "wb") as f:
        f.write(data)
    return ms_since(t0)


# ---------------------------------------------------------------- 메타, 모니터링

def run_cmd(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def collect_meta(args):
    cpu = ""
    for line in read_text("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    return {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "os": read_text("/etc/os-release").split("\n")[0] if read_text("/etc/os-release") else "",
        "cpu": cpu,
        "cpu_governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "intel_no_turbo": read_text("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "mem_total": next((l for l in read_text("/proc/meminfo").splitlines()
                           if l.startswith("MemTotal")), ""),
        "swap": run_cmd(["swapon", "--show"]),
        "disks": run_cmd(["lsblk", "-o", "NAME,SIZE,MODEL,MOUNTPOINTS"]),
        "llama_cpp_commit": run_cmd(["git", "-C", args.llama_dir, "rev-parse", "HEAD"]),
        "bench_repo_commit": run_cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "model_sha256": read_text(Path(args.models_dir) / "SHA256SUMS"),
        "python": sys.version.split()[0],
        "args": vars(args),
    }


def start_monitors(out_dir):
    procs = []
    if shutil.which("iostat"):
        f = open(out_dir / "iostat.log", "w")
        procs.append((subprocess.Popen(["iostat", "-x", "-m", "-t", "1"], stdout=f,
                                       stderr=subprocess.DEVNULL), f))
    if shutil.which("vmstat"):
        f = open(out_dir / "vmstat.log", "w")
        procs.append((subprocess.Popen(["vmstat", "-t", "1"], stdout=f,
                                       stderr=subprocess.DEVNULL), f))
    return procs


def stop_monitors(procs):
    for p, f in procs:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
        f.close()


# ---------------------------------------------------------------- 측정

def resume(srv, tokens, args, ref_content):
    """복원된 슬롯으로 같은 요청을 보내고 결과를 확인한다."""
    res = srv.completion(tokens, args.n_predict)
    t = res.get("timings", {})
    return t.get("prompt_n"), t.get("prompt_ms"), res.get("content") == ref_content


def bench_model(args, key, writer, csv_f, out_dir, progress):
    model_path = Path(args.models_dir) / MODELS[key]
    raw_dir = Path(args.kv_dir) / "raw"
    lz4_dir = Path(args.kv_dir) / "lz4"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lz4_dir.mkdir(parents=True, exist_ok=True)

    proc = log_f = None
    if not args.no_server:
        proc, srv, log_f = start_server(args, model_path, raw_dir, out_dir, key)
    else:
        srv = Server(args.port)
        if not srv.healthy():
            raise RuntimeError(f"--no-server인데 {args.port} 포트에 서버가 없음")

    try:
        corpus = Path(args.corpus).read_text(encoding="utf-8", errors="ignore")
        all_tokens = srv.tokenize(corpus)
        need = max(args.lengths)
        if len(all_tokens) < need:
            log(f"경고: 재료 토큰 {len(all_tokens)}개 < 필요 {need}개, 재료를 반복해서 씀")
            all_tokens = (all_tokens * (need // len(all_tokens) + 1))
        log(f"[{key}] 재료 토큰 {len(all_tokens)}개")

        # 워밍업: 서버를 켠 뒤 첫 요청은 메모리 할당 등으로 느려서 측정에서 제외
        for _ in range(args.warmup):
            srv.slot("erase")
            srv.completion(all_tokens[:min(512, max(args.lengths))], 4)
        srv.slot("erase")
        if args.warmup:
            log(f"[{key}] 워밍업 {args.warmup}회 완료")

        for n in args.lengths:
            tokens = all_tokens[:n]
            for rep in range(1, args.repeats + 1):
                progress["done"] += 1
                log(f"[{progress['done']}/{progress['total']}] {key} N={n} rep={rep}")
                fname = f"kv_{key}_N{n}_r{rep}.bin"
                raw_path = raw_dir / fname
                lz4_path = lz4_dir / (fname + ".lz4")

                # 1. 재계산: 프롬프트 처리 시간 (생성은 1토큰만)
                srv.slot("erase")
                a = srv.completion(tokens, 1)
                ta = a.get("timings", {})
                base = {
                    "model": key, "n_tokens": n, "rep": rep,
                    "recompute_prompt_n": ta.get("prompt_n"),
                    "recompute_ms": round(ta.get("prompt_ms", 0), 3),
                }

                # 2. RAW 저장 (+ fsync로 실제 디스크 쓰기까지)
                s = srv.slot("save", fname)
                t0 = time.perf_counter()
                fsync_file(raw_path)
                base["fsync_ms"] = round(ms_since(t0), 3)
                base["save_ms"] = round(s.get("timings", {}).get("save_ms", 0), 3)
                base["raw_bytes"] = raw_path.stat().st_size

                # 3. 기준 출력 A: 저장한 것과 같은 KV 상태에서 생성
                ref = srv.completion(tokens, args.n_predict).get("content")

                def record(storage, cache, extra, restore_res, t_start):
                    pn, pms, match = resume(srv, tokens, args, ref)
                    restore_ms = restore_res.get("timings", {}).get("restore_ms", 0)
                    total = restore_ms + (pms or 0) + extra.get("decompress_ms", 0)
                    row = dict(base, storage=storage, cache=cache,
                               restore_ms=round(restore_ms, 3),
                               n_restored=restore_res.get("n_restored"),
                               resume_prompt_n=pn, resume_prompt_ms=round(pms or 0, 3),
                               resume_total_ms=round(total, 3),
                               speedup=round(base["recompute_ms"] / total, 2) if total else "",
                               output_match=match,
                               t_start=round(t_start, 3), t_end=round(time.time(), 3),
                               **{k: round(v, 3) if isinstance(v, float) else v
                                  for k, v in extra.items()})
                    writer.writerow(row)
                    csv_f.flush()
                    flag = "" if match else "  <-- 출력 불일치!"
                    log(f"    {storage:3s} {cache:4s}: 복원 {total:8.1f} ms vs 재계산 "
                        f"{base['recompute_ms']:8.1f} ms (x{row['speedup']}){flag}")

                # 3. RAW 복원
                if "raw" in args.storage:
                    for cache in args.cache:
                        t_start = time.time()
                        srv.slot("erase")
                        if cache == "cold":
                            evict(raw_path, args.evict)
                        r = srv.slot("restore", fname)
                        record("raw", cache, {}, r, t_start)

                # 4~5. LZ4
                if "lz4" in args.storage:
                    comp_ms, comp_bytes = lz4_compress(raw_path, lz4_path)
                    raw_path.unlink()
                    lz4_extra = {"lz4_bytes": comp_bytes, "compress_ms": comp_ms}
                    for cache in args.cache:
                        t_start = time.time()
                        srv.slot("erase")
                        if cache == "cold":
                            evict(lz4_path, args.evict)
                        dec_ms = lz4_decompress(lz4_path, raw_path)
                        r = srv.slot("restore", fname)
                        record("lz4", cache, dict(lz4_extra, decompress_ms=dec_ms), r, t_start)
                        raw_path.unlink()

                for p in (raw_path, lz4_path):
                    if p.exists():
                        p.unlink()
    finally:
        stop_server(proc, log_f)


def summarize(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    groups = {}
    for r in rows:
        k = (r["model"], int(r["n_tokens"]), r["storage"], r["cache"])
        groups.setdefault(k, []).append(r)
    lines = ["model,n_tokens,storage,cache,runs,recompute_ms_mean,resume_total_ms_mean,"
             "speedup_mean,raw_bytes,lz4_bytes,all_match"]
    print("\n=== 요약 (평균) ===", flush=True)
    print(f"{'model':6s} {'N':>6s} {'stor':4s} {'cache':5s} {'재계산ms':>10s} "
          f"{'복원ms':>9s} {'배율':>6s} {'일치':>4s}")
    for k in sorted(groups):
        g = groups[k]
        rec = statistics.mean(float(x["recompute_ms"]) for x in g)
        res = statistics.mean(float(x["resume_total_ms"]) for x in g)
        sp = rec / res if res else 0
        match = all(x["output_match"] == "True" for x in g)
        lz = g[0].get("lz4_bytes", "")
        print(f"{k[0]:6s} {k[1]:6d} {k[2]:4s} {k[3]:5s} {rec:10.1f} {res:9.1f} "
              f"{sp:6.1f} {'O' if match else 'X':>4s}")
        lines.append(f"{k[0]},{k[1]},{k[2]},{k[3]},{len(g)},{rec:.3f},{res:.3f},{sp:.2f},"
                     f"{g[0]['raw_bytes']},{lz},{match}")
    (csv_path.parent / "summary.csv").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description="KV Cache 저장·복원 vs 재계산 벤치마크")
    p.add_argument("--model", default="qwen", choices=["qwen", "llama", "both"])
    p.add_argument("--lengths", default="128,512,1024,2048,4096",
                   help="프롬프트 토큰 수, 쉼표로 구분")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--storage", default="raw,lz4", help="raw, lz4 중 선택 (쉼표)")
    p.add_argument("--cache", default="cold,warm", help="cold, warm 중 선택 (쉼표)")
    p.add_argument("--evict", default="drop", choices=["drop", "fadvise"],
                   help="cold 만드는 방법: drop(전체 캐시 비우기, sudo) / fadvise(해당 파일만)")
    p.add_argument("--n-predict", type=int, default=32, help="생성 토큰 수 (출력 비교용)")
    p.add_argument("--warmup", type=int, default=2, help="측정 전 워밍업 요청 횟수")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--ctx", type=int, default=0, help="컨텍스트 크기 (0이면 자동)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--mem-limit", default="", help="예: 2G, 768M (비우면 제한 없음)")
    p.add_argument("--swap-max", default="0", help="--mem-limit 때 swap 허용량 (예: 0, 2G)")
    p.add_argument("--out", default="", help="결과 폴더 (기본: ~/aiwork/results/날짜_시각)")
    p.add_argument("--llama-dir", default=str(HOME / "aiwork/llama.cpp"))
    p.add_argument("--models-dir", default=str(HOME / "aiwork/models"))
    p.add_argument("--kv-dir", default="/mnt/ssd/kvcache")
    p.add_argument("--corpus", default=str(REPO_ROOT / "prompts/corpus.txt"))
    p.add_argument("--server-timeout", type=int, default=300)
    p.add_argument("--no-server", action="store_true",
                   help="서버를 띄우지 않고 이미 켜진 서버 사용 (디버깅용)")
    p.add_argument("--no-monitor", action="store_true", help="iostat/vmstat 기록 끄기")
    a = p.parse_args()

    a.lengths = sorted(int(x) for x in a.lengths.split(",") if x.strip())
    a.storage = [x.strip() for x in a.storage.split(",") if x.strip()]
    a.cache = [x.strip() for x in a.cache.split(",") if x.strip()]
    if not set(a.storage) <= {"raw", "lz4"} or not set(a.cache) <= {"cold", "warm"}:
        p.error("--storage는 raw/lz4, --cache는 cold/warm만 가능")
    if not a.ctx:
        a.ctx = max(a.lengths) + a.n_predict + 64
    if not a.out:
        a.out = str(HOME / "aiwork/results" / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    return a


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = ["qwen", "llama"] if args.model == "both" else [args.model]

    if not args.no_server and port_in_use(args.port):
        sys.exit(f"{args.port} 포트를 이미 누가 쓰고 있음. `pgrep -a llama` 로 남은 서버를 확인하세요.")
    if "lz4" in args.storage:
        import lz4.frame  # noqa: F401  (없으면 여기서 바로 실패)

    (out_dir / "meta.json").write_text(json.dumps(collect_meta(args), indent=2, ensure_ascii=False))
    log(f"결과 폴더: {out_dir}")

    csv_path = out_dir / "results.csv"
    monitors = [] if args.no_monitor else start_monitors(out_dir)
    progress = {"done": 0, "total": len(models) * len(args.lengths) * args.repeats}
    t_all = time.time()
    try:
        with open(csv_path, "w", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for key in models:
                bench_model(args, key, writer, csv_f, out_dir, progress)
    finally:
        stop_monitors(monitors)
    log(f"전체 소요: {(time.time() - t_all) / 60:.1f}분")
    summarize(csv_path)

    mismatches = sum(1 for r in csv.DictReader(open(csv_path)) if r["output_match"] != "True")
    if mismatches:
        log(f"주의: 출력 불일치 {mismatches}건")
        sys.exit(2)


if __name__ == "__main__":
    main()
