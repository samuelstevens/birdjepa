# TODO

## High Priority

### Fix multi-GPU JAX distributed training failures
**STATUS: FIXED - Use explicit coordinator from submitit**

Using `submitit.helpers.TorchDistributedEnvironment()` to get the coordinator address and passing it explicitly to `jax.distributed.initialize()` fixes the heartbeat issue.

**Previous issue:** Jobs died at exactly `heartbeat_timeout_seconds` after start when relying on JAX's auto-detection. Training continued fine (NCCL collectives worked) but coordinator heartbeat gRPC channel failed.

**Key insight from debugging:**
- The JAX coordinator uses a separate gRPC channel from training collectives
- Training can succeed while heartbeats fail (different network paths)
- Auto-detected port (65422 from `job_id % 4096 + 61440`) may have issues
- Explicit coordinator port (27508 from submitit) works correctly
- 1800s timeout survives the issue; explicit coordinator fixes it properly

**Fix in `pretrain.py`:**
```python
dist_env = submitit.helpers.TorchDistributedEnvironment()
coordinator_address = f"{dist_env.master_addr}:{dist_env.master_port}"
jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=dist_env.world_size,
    process_id=dist_env.rank,
    initialization_timeout=300,
    heartbeat_timeout_seconds=120,
)
```

**Verified:** Job 3192373 ran 12+ min with training, using 120s heartbeat timeout.

**Related issues:**
- https://github.com/jax-ml/jax/issues/33852 (XLA compilation hangs)
- https://github.com/jax-ml/jax/issues/16788 (Slurm GPU detection)
- https://github.com/jax-ml/jax/issues/23452 (GPU binding)

### Cache resamplers in Rust loader
Currently re-initializing rubato SincFixedIn resampler for every sample. This is expensive (~10ms overhead per sample). Should cache resamplers by (source_rate, target_rate) tuple.
- Location: `rust/src/decode.rs` in `DecodedAudio::resample()`
- Use thread-local cache or Arc<Mutex<HashMap>>
- Benchmark showed 650 samples/sec with 48 threads - caching could improve this

## Medium Priority

### Clean up grain-based code in data/__init__.py
Now that pretrain.py uses Rust loader exclusively, these classes are unused:
- `IndexedXenoCantoDataset` - old random-access dataset
- `ShuffledXenoCantoDataset` - grain-based sequential loader
- `make_shuffled_dataloader()` - grain pipeline builder

Keep for now if needed for other scripts, or remove to reduce maintenance burden.

### Update sweep configs
- `sweeps/pretrain_xcm.py` - updated but untested
- `sweeps/001_freewins/adamw_vits_xcl_lr_sweep.py` - fixed `buffer_size`/`min_size` -> `window_size`
- Other sweeps may have stale config options

## Completed

- [x] Match Rust SpectrogramTransform to Python kaldi_fbank exactly
- [x] Use rubato for proper sinc resampling (was linear interpolation)
- [x] Remove string label hashing from arrow.rs (error on non-int64 labels)
- [x] Integrate Rust loader into pretrain.py
- [x] Remove grain from pretrain.py (checkpointing, imports)
- [x] Update CPU request logic: `max(mem_gb // 10, n_workers)`

## Notes

### Rust loader pipeline
```
Arrow files -> I/O thread -> Worker pool (decode+resample+spectrogram) -> Shuffle buffer -> Python postprocess (pad to 512, normalize) -> Training
```

### Benchmark results (uncached resamplers)
| n_workers | throughput (samples/sec) | RSS (MB) |
|-----------|-------------------------|----------|
| 48        | 653.8                   | 9498     |
| 64        | 559.4                   | 10260    |

Peak at 48 workers due to CPU contention (48 CPUs requested).

