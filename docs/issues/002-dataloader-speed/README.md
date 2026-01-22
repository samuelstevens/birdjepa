# Dataloader Speed

## Summary of Findings (Jan 2026)

We implemented the hierarchical shuffle pattern described below. Key observations:

What we built:
- Shuffle Arrow file paths (cheap random access on ~200 paths)
- Read each Arrow file sequentially via `InterleaveIterDataset`
- Window shuffle buffer for local randomness
- TMPDIR caching (copy shard to local SSD on first access)
- `mp_prefetch` with multiple workers

Performance observations (job 3180169, 4 workers, cycle_length=4):
- Fast steps: 1.5-1.6 sec (workers have data ready in buffers)
- Medium steps: 100-140 sec (TMPDIR cached shard load)
- Slow steps: 370-470 sec (fresh NFS load for ~700 MB shard)

Open questions:
1. Why does 8 workers perform worse than 4 workers? (step 1: 1086s vs 640s)
   - Is it NFS bandwidth contention?
   - Memory pressure / swapping?
   - CPU contention from audio decoding?
   - We assumed "shared NFS bandwidth" but that's not obviously true for separate processes

2. Can we truly overlap shard loading with GPU compute?
   - `ThreadPrefetchIterDataset` before `InterleaveIterDataset` fails (set_slice subscripting)
   - `mp_prefetch` workers load independently but still hit synchronous blocking within each worker
   - Need to understand grain's threading/process model better

3. What is the actual NFS throughput per process vs total?
   - Need to measure with `iostat` or similar during training
   - Is the bottleneck client-side or server-side?

Current best config:
```python
n_workers = 4
cycle_length = 4
per_worker_buffer_size = 4
iter_buffer_size = 256
# Plus TMPDIR caching in make_shard_iter
```

---

Let's switch to grain's Dataset API to handle poor random access on NFS systems.

Resources

- https://google-grain.readthedocs.io/en/stable/tutorials/dataset_advanced_tutorial.html#hierarchical-shuffle
- https://google-grain.readthedocs.io/en/stable/grain.experimental.html
- https://google-grain.readthedocs.io/en/stable/_autosummary/grain.experimental.InterleaveIterDataset.html#grain.experimental.InterleaveIterDataset

Don't forget the bad indices in XCL.

Below are some ideas from different LLMs.

## Proposed Approach

Okay, I think map, InterleaveIterDataset and WindowShuffleIterDataset would work. How do I use a huggingface dataset in this context? I actually have a bunch of arrow files, and I need to cast some of the data to audio. 

If I use map(make_source), is it lazy? Or will it eagerly try to load the whole dataset into memory?

I load my dataset right now with 

```
self.ds = datasets.load_dataset(
    "samuelstevens/BirdSet", cfg.subset, split=cfg.split
)
# Decode audio bytes to arrays
self.ds = self.ds.cast_column("audio", datasets.Audio(sampling_rate=SR_HZ))
```

I'm not sure how to split this up into shards, but ds.cache_files has a bunch of files like

['/fs/scratch/PAS2136/samuelstevens/cache/huggingface/datasets/samuelstevens___bird_set/XCM/0.0.0/fe90ff01a49abb9f8ee55f0d484ccc783a05b71f/bird_set-train-00000-of-00185.arrow', '/fs/scratch/PAS2136/samuelstevens/cache/huggingface/datasets/samuelstevens___bird_set/XCM/0.0.0/fe90ff01a49abb9f8ee55f0d484ccc783a05b71f/bird_set-train-00001-of-00185.arrow', '/fs/scratch/PAS2136/samuelstevens/cache/huggingface/datasets/samuelstevens___bird_set/XCM/0.0.0/fe90ff01a49abb9f8ee55f0d484ccc783a05b71f/bird_set-train-00002-of-00185.arrow', '/fs/scratch/PAS2136/samuelstevens/cache/huggingface/datasets/samuelstevens___bird_set/XCM/0.0.0/fe90ff01a49abb9f8ee55f0d484ccc783a05b71f/bird_set-train-00003-of-00185.arrow', '/fs/scratch/PAS2136/samuelstevens/cache/huggingface/datasets/samuelstevens___bird_set/XCM/0.0.0/fe90ff01a49abb9f8ee55f0d484ccc783a05b71f/bird_set-train-00004-of-00185.arrow']

## GPT 5.2 Thinking

You do not need to build an `IterDataset` “from scratch” for this case.

Grain already provides an `IterDataset` implementation that reads Parquet sequentially: `grain.experimental.ParquetIterDataset(path=...)` (it is explicitly “an IterDataset for a parquet format file”). ([google-grain.readthedocs.io][1])

What’s tripping you up is that many examples start from `MapDataset.source(...)` because that’s the common path when you have efficient random access. For Parquet-on-NFS, the pattern is instead:

1. Keep random access only at the *file list* level (in memory, cheap).
2. Turn each filename into a per-file streaming `IterDataset` (`ParquetIterDataset`).
3. Interleave multiple file streams to get mixing + hide latency (still sequential within each file).

A concrete “many parquet shards” template

```python
import glob
import grain
from grain import experimental as gx

parquet_files = sorted(glob.glob("/nfs/path/to/shards/*.parquet"))

def make_source(path: str) -> grain.IterDataset:
    # Sequential reader for ONE parquet file.
    return gx.ParquetIterDataset(path)

# Random access is only over the Python list of filenames (fine).
file_ds = grain.MapDataset.source(parquet_files).shuffle(seed=42).map(make_source)

# Interleave a handful of per-file iterators concurrently.
ds = gx.InterleaveIterDataset(file_ds, cycle_length=8, iter_buffer_size=2)

# Optional: extra mixing without global random access.
ds = ds.pipe(gx.WindowShuffleIterDataset, window_size=10_000, seed=123)

# From here on, use normal IterDataset transforms (map/random_map/batch/...)
# ds = ds.random_map(...)
# ds = ds.batch(...)
```

This is the usage Grain’s `InterleaveIterDataset` docs describe: build a `MapDataset` of datasets from filenames, then interleave them, specifically for sources “that do not support random access and are implemented as IterDataset”. ([google-grain.readthedocs.io][2])

So, do you need to subclass `IterDataset` yourself?

Only if you have a source that Grain doesn’t already wrap (or you need custom checkpoint/state behavior beyond what the provided datasets do). In that case, yes: you implement a `grain.IterDataset` whose `__iter__` returns a `grain.DatasetIterator`, and the iterator must implement `__next__`, `get_state`, and `set_state` (and should support `close`). ([google-grain.readthedocs.io][3])

But for “HF dataset stored as parquet shards on NFS”, the best option is usually: `ParquetIterDataset` per shard + `InterleaveIterDataset` (+ optional `WindowShuffleIterDataset`), and then put your randomness (e.g., random cropping) in `random_map` so you still read sequentially. ([google-grain.readthedocs.io][1])

[1]: https://google-grain.readthedocs.io/en/latest/tutorials/data_sources/parquet_dataset_tutorial.html "Reading Apache Parquet Files — Grain"
[2]: https://google-grain.readthedocs.io/en/latest/_autosummary/grain.experimental.InterleaveIterDataset.html "grain.experimental.InterleaveIterDataset — Grain"
[3]: https://google-grain.readthedocs.io/en/stable/grain.dataset.html "grain Dataset — Grain"


## Claude Opus 4.5

Let me fetch that documentation page to understand the current Grain API options, and then I can give you concrete guidance.Let me search for Grain's IterDataset documentation:This is helpful! I found a key pattern. Let me get more details on how to create an IterDataset from scratch:Let me check the Grain source code to see how to subclass IterDataset:Based on my research, here's the situation with Grain and your use case:

**The Core Problem**

Grain's design philosophy assumes you start with a `MapDataset` (random access) and convert to `IterDataset` late in the pipeline. There's no obvious `IterDataset.from_iterable()` factory method in the public API.

**Best Option: `InterleaveIterDataset` Pattern**

The recommended approach for sources without random access is `grain.experimental.InterleaveIterDataset`. The pattern documented is:

```python
def make_source(filename: str) -> grain.IterDataset:
    # You need to implement this - return an IterDataset for one shard
    ...

# Shuffle at the shard level, then interleave
ds = grain.MapDataset.source(filenames).shuffle(seed=42).map(make_source)
ds = grain.experimental.InterleaveIterDataset(ds, cycle_length=4)
ds = ds.mp_prefetch(2)
```

**The Missing Piece: How to Create `grain.IterDataset` from a Generator**

This is where Grain's documentation is frustratingly thin. You have a few options:

1. **Wrap via MapDataset (if you can buffer a shard)**: If each parquet file fits in memory, you could read it eagerly into a list and use `MapDataset.source(list_of_rows).to_iter_dataset()`. This defeats pure streaming but works at the shard level.

2. **Use HuggingFace's streaming directly and skip Grain's Dataset API**: Load with `streaming=True` to get an `IterableDataset`, then just iterate directly without Grain's Dataset abstractions. You lose Grain's checkpointing and transformations.

3. **Subclass `IterDataset`**: Looking at the Grain source, `IterDataset` is an abstract class requiring `__iter__` to return a `DatasetIterator`. This is underdocumented but possible.

**Practical Recommendation for Your NFS + Parquet Setup**

Given you're on NFS where random access is slow, I'd try this hybrid approach:

```python
import grain
import pyarrow.parquet as pq
from datasets import load_dataset

# Option A: Stream one parquet file into a list, convert to IterDataset
def make_source(parquet_path: str) -> grain.IterDataset:
    # Read the entire parquet file sequentially (fast on NFS)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()  # or to_pandas().to_dict('records')
    return grain.MapDataset.source(rows).to_iter_dataset()

parquet_files = [...]  # list of your parquet file paths
ds = (
    grain.MapDataset.source(parquet_files)
    .shuffle(seed=42)
    .map(make_source)
)
ds = grain.experimental.InterleaveIterDataset(ds, cycle_length=4)
ds = ds.map(...).batch(...)
```

This gives you shard-level shuffling (good enough for most training), sequential reads within each file (NFS-friendly), and Grain's transformation pipeline.

**Alternative: HuggingFace Streaming + Manual Integration**

```python
from datasets import load_dataset

# HuggingFace handles the streaming natively
hf_ds = load_dataset("parquet", data_files={"train": "path/to/*.parquet"}, 
                      split="train", streaming=True)

# Just iterate directly - no Grain Dataset API needed
for batch in hf_ds.batch(32):
    # your training loop
```

Would you like me to dig deeper into the `IterDataset` subclassing approach, or does one of these patterns work for your situation?

Great questions. Let me address each:

**Is `map(make_source)` lazy?**

Yes! `MapDataset.map()` is lazy - it only calls `make_source` when you actually access an element. So `grain.MapDataset.source(filenames).map(make_source)` doesn't load anything until `InterleaveIterDataset` starts pulling from it.

**Working with your Arrow shards + Audio casting**

Here's how I'd structure this:

```python
import grain
import pyarrow as pa
import datasets
import numpy as np

SR_HZ = 32000  # or whatever your sampling rate is

def make_source(arrow_path: str) -> grain.IterDataset:
    """Load one Arrow shard and return as IterDataset."""
    # Read the Arrow file (sequential read - NFS friendly)
    mmap = pa.memory_map(arrow_path, 'r')
    table = pa.ipc.open_file(mmap).read_all()
    
    # Convert to list of dicts
    # Each row is a dict with your columns
    rows = table.to_pylist()
    
    # Return as IterDataset
    # This is where the data actually lives in memory (per shard)
    return grain.MapDataset.source(rows).to_iter_dataset()


def decode_audio(example: dict) -> dict:
    """Decode audio bytes to array - runs lazily during iteration."""
    audio_bytes = example["audio"]["bytes"]
    # Use HuggingFace's Audio feature for decoding
    audio_feature = datasets.Audio(sampling_rate=SR_HZ)
    decoded = audio_feature.decode_example(example["audio"])
    example["audio"] = decoded
    return example


# Get your shard paths
# Option 1: From an already-loaded dataset
ds = datasets.load_dataset("samuelstevens/BirdSet", cfg.subset, split=cfg.split)
arrow_files = [f["filename"] for f in ds.cache_files]

# Option 2: Just glob them directly
# import glob
# arrow_files = sorted(glob.glob("/fs/scratch/.../bird_set-train-*.arrow"))

# Build the Grain pipeline
ds = (
    grain.MapDataset.source(arrow_files)
    .shuffle(seed=42)  # Shuffle shard order
    .map(make_source)  # LAZY - only called when InterleaveIterDataset pulls
)

ds = grain.experimental.InterleaveIterDataset(
    ds, 
    cycle_length=4  # Read from 4 shards concurrently
)

ds = (
    ds
    .map(decode_audio)  # Decode audio bytes -> arrays
    .map(your_other_transforms)
    .batch(batch_size=32)
)

# Optional: Add shuffle buffer for better randomness within the interleave window
# ds = grain.experimental.WindowShuffleIterDataset(ds, window_size=1000)
```

**One important caveat about memory:**

Inside `make_source`, when you call `table.to_pylist()`, that entire shard gets loaded into memory. With 185 shards and `cycle_length=4`, you'll have ~4 shards in memory at once. If each shard is large, this could be problematic.

**Alternative: Stream rows from Arrow without full materialization**

If memory is tight, you could iterate through the Arrow file row by row:

```python
def make_source(arrow_path: str) -> grain.IterDataset:
    """Stream rows from Arrow file without loading entire shard."""
    mmap = pa.memory_map(arrow_path, 'r')
    reader = pa.ipc.open_file(mmap)
    
    # Read row-by-row via record batches
    rows = []
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        rows.extend(batch.to_pylist())
    
    return grain.MapDataset.source(rows).to_iter_dataset()
```

But honestly, this still materializes the shard. The fundamental tension is that Grain's `MapDataset.source()` needs a sequence with `__len__` and `__getitem__`.

**If you truly need streaming within shards**, you'd need to subclass `IterDataset` directly, which is messier. But for most cases, shard-level loading (4 shards in memory) should be fine.

Want me to dig into the true streaming approach, or does the shard-level loading work for your memory constraints?

## Gemini 3 Thinking

To leverage Grain on NFS with HF Arrow shards, you should treat the **file paths** as the `MapDataset` source. This ensures Grain handles the sharding/distribution logic at the file level, while the I/O remains sequential.

### 1. Is `map` lazy?

In Grain, `map` on an `IterDataset` is **lazy**. It creates a transformation pipeline. It will only call your `make_source` function as it pulls elements through the iterator. However, you must be careful with how you load data inside that map. If your map function calls `ds.to_list()`, it will eagerly load that specific shard into memory. To stay memory-efficient, your map should return an iterator or use `flat_map`.

### 2. Implementation: The File-Based Pipeline

Since you have 185 Arrow files, use those as your Grain source. This bypasses the HF `Dataset` object's random access metadata overhead.

```python
import grain.python as grain
import datasets
import pyarrow as pa

# 1. Get your list of shards
# You can get these from ds.cache_files or globbing the directory
shard_paths = [f['filename'] for f in self.ds.cache_files]

# 2. Define a loader for a single shard
def load_shard_sequential(path: str):
    # Use memory_map=True for NFS if the files are large, 
    # but for sequential streaming, False is often safer to avoid NFS hang
    table = pa.ipc.open_stream(path).read_all() 
    
    # Cast to HF dataset for the convenient .cast_column() logic
    # This is a local operation on the loaded table
    shard_ds = datasets.Dataset(table)
    shard_ds = shard_ds.cast_column("audio", datasets.Audio(sampling_rate=SR_HZ))
    
    # Return as an iterable of dicts
    for row in shard_ds:
        yield row

# 3. Build the Grain Pipeline
# Start with a MapDataset of strings (paths)
ds = grain.MapDataset.source(shard_paths)

# Transition to IterDataset for sequential NFS access
ds = ds.to_iter_dataset()

# flat_map turns 1 path -> N rows
ds = ds.flat_map(load_shard_sequential)

# Now you can use the transformations you mentioned
ds = ds.shuffle(seed=42) # This is a WindowShuffleIterDataset
ds = ds.batch(batch_size=cfg.batch_size)

```

### 3. Why `InterleaveIterDataset`?

If you use the `flat_map` approach above, you are reading shards 0, then 1, then 2. This is bad for mixing. To get better data variety before your `WindowShuffleIterDataset` hits its buffer limit, use `InterleaveIterDataset`.

Grain’s `InterleaveIterDataset` takes multiple `IterDatasets` and samples from them.

```python
# Create a list of IterDatasets, one per shard
shard_datasets = [
    grain.IterDataset.from_iterable(load_shard_sequential(p)) 
    for p in shard_paths
]

# Interleave them to read from multiple shards in parallel
# cycle_length determines how many shards are open at once
ds = grain.InterleaveIterDataset(
    shard_datasets, 
    cycle_length=8 
)

ds = ds.shuffle(seed=42)

```

### Critical NFS/JAX Optimizations

* **Decouple Decoding:** Do not perform `cast_column("audio", ...)` inside the `load_shard_sequential` if it's slow. Instead, return the raw bytes and use a Grain `.map(decode_audio_fn)` transformation. This allows Grain to parallelize the CPU-heavy audio decoding across multiple worker processes.
* **Prefetching:** Always end your pipeline with `.prefetch(buffer_size=N)`. On NFS, I/O bursts are common; prefetching hides the latency of the next shard load.
* **Arrow vs Parquet:** Arrow files (IPC format) are actually better for this than Parquet because they are designed for "unmapping" directly into memory without complex decompression of columnar blocks.

### Handling Multi-Process JAX

If you are running on multiple nodes or multi-process JAX, Grain needs to know which worker gets which shards.

```python
# Before creating the source, shard the paths based on worker index
worker_info = grain.get_worker_info() # Usually provided by Grain's DataLoader
# or manually via jax.process_index()
my_shards = shard_paths[jax.process_index()::jax.process_count()]
ds = grain.MapDataset.source(my_shards)

```

Would you like me to write a specific `decode_audio_fn` that uses `librosa` or `scipy` instead of the HF `Audio` feature to make the Grain `.map()` call more efficient?

