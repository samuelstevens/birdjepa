"""Benchmark Rust loader throughput on cluster with different worker counts."""

import os
import resource
import time

import polars as pl

import birdjepa.helpers


def get_arrow_files() -> list[str]:
    import datasets

    ds = datasets.load_dataset("samuelstevens/BirdSet", "XCL", split="train")
    return [f["filename"] for f in ds.cache_files]


def get_rss_mb() -> float:
    """Get current RSS (resident set size) in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def bench_single_config(
    n_workers: int,
    batch_size: int = 64,
    shuffle_buffer_size: int = 4000,
    warmup_batches: int = 20,
    measure_batches: int = 100,
) -> dict:
    from birdjepa._rs import Loader

    arrow_files = get_arrow_files()

    rss_before = get_rss_mb()

    loader = Loader(
        arrow_files,
        seed=42,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_min_size=shuffle_buffer_size // 2,
        n_workers=n_workers,
        infinite=True,
    )

    # Warmup
    for i, _ in enumerate(loader):
        if i >= warmup_batches:
            break

    rss_after_warmup = get_rss_mb()

    # Measure
    n_samples = 0
    start = time.perf_counter()
    for i, batch in enumerate(birdjepa.helpers.progress(loader)):
        n_samples += len(batch["labels"])
        if i >= measure_batches:
            break
    elapsed = time.perf_counter() - start

    rss_peak = get_rss_mb()

    return {
        "n_workers": n_workers,
        "batch_size": batch_size,
        "shuffle_buffer_size": shuffle_buffer_size,
        "warmup_batches": warmup_batches,
        "measure_batches": measure_batches,
        "n_samples": n_samples,
        "elapsed_sec": round(elapsed, 2),
        "throughput_samples_per_sec": round(n_samples / elapsed, 1),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_warmup_mb": round(rss_after_warmup, 1),
        "rss_peak_mb": round(rss_peak, 1),
    }


def main():
    import argparse
    import submitit
    import submitit.core.utils

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated worker counts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rust/bench_workers_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    worker_counts = [int(x) for x in args.workers.split(",")]

    executor = submitit.SlurmExecutor(folder="logs/bench_rust_loader")
    # OSC allocates ~10GB RAM per CPU, so request enough CPUs for desired memory
    mem_gb = 64
    n_cpus = max(48, mem_gb // 10)
    executor.update_parameters(
        partition="preemptible-nextgen",
        account="PAS2136",
        time=30,  # Each single-worker bench takes ~15 min
        cpus_per_task=n_cpus,
        stderr_to_stdout=True,
        setup=[
            "module load ffmpeg/6.1.1",
        ],
    )

    # Submit one job per worker count in parallel
    with executor.batch():
        jobs = [
            executor.submit(bench_single_config, n_workers)
            for n_workers in worker_counts
        ]

    time.sleep(5.0)
    for j, (n_workers, job) in enumerate(zip(worker_counts, jobs)):
        print(
            f"Job {j + 1}/{len(jobs)}: workers={n_workers} job_id={job.job_id} state={job.state}"
        )

    results = []
    for j, job in enumerate(jobs):
        try:
            result = job.result()
            results.append(result)
            print(
                f"Job {j + 1}/{len(jobs)} finished: {result['throughput_samples_per_sec']} samples/sec"
            )
        except submitit.core.utils.UncompletedJobError:
            print(f"Job {job.job_id} did not finish")

    if not results:
        print("No results collected")
        return

    # Write CSV using polars
    df = pl.DataFrame(results).sort("n_workers")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.write_csv(args.output)

    print(f"\nResults written to {args.output}")
    print("\nSummary:")
    print(df.select("n_workers", "throughput_samples_per_sec", "rss_peak_mb"))


if __name__ == "__main__":
    main()
