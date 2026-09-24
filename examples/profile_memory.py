"""Sample Linux process-tree PSS and optional NVIDIA process memory outside timing."""
import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path


def process_tree(pid):
    result = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    with args.output.open("w") as output:
        writer = csv.writer(output)
        writer.writerow(["monotonic_seconds", "host_pss_bytes", "gpu_process_bytes", "process_count"])
        first = True
        while running:
            pids = process_tree(args.pid) - process_tree(os.getpid())
            pss = 0
            for pid in pids:
                try:
                    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                        if line.startswith("Pss:"):
                            pss += int(line.split()[1]) * 1024
                except (OSError, ProcessLookupError):
                    pass
            gpu = 0
            if args.gpu:
                query = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True,
                )
                for line in query.stdout.splitlines():
                    pid, memory = line.split(",")
                    if int(pid) in pids:
                        gpu += int(memory) * 1024 * 1024
            writer.writerow([time.perf_counter(), pss, gpu, len(pids)])
            output.flush()
            if first:
                print("ready", flush=True)
                first = False
            time.sleep(0.1)


if __name__ == "__main__":
    main()
