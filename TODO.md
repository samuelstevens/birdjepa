# TODO

## High Priority

### Fix multi-GPU JAX distributed training failures
**STATUS: FIXED - Increase heartbeat_timeout_seconds to 300**

Root cause: XLA compilation of first train step with gradient checkpointing + multi-device sharding takes longer than the default 100s heartbeat timeout, causing the coordination service to declare both processes dead.

**Fix applied in `pretrain.py`:**
```python
jax.distributed.initialize(
    initialization_timeout=600,
    heartbeat_timeout_seconds=300,  # Increased from default 100s
)
```

**Verified working:** Job 3189658 trained 30+ steps on 2 GPUs (a0131) with loss decreasing normally.

**Related issue:** https://github.com/jax-ml/jax/issues/33852

**Logs:** `logs/3189658_0_log.out` (multi-GPU success with fix)

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

