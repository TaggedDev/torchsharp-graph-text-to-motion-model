# CUDA Performance Analysis Guide

## Issues Identified

### 1. **Adjacency Matrix Transfer (GcnSpatialTemporalModel.cs:107-108)**
**Problem**: Moving adjacency matrix to GPU EVERY forward pass
```csharp
// SLOW (happens 60+ times per batch)
var adj = _adj.to(x.device);
var adjBatch = adj.unsqueeze(0);
```

**Impact**: Even small CPU→GPU transfers add up across forward passes
- Movement: ~1.1KB per transfer (22×22 float32 matrix)
- Frequency: 1 transfer per GCN layer per forward pass
- With 2 GCN layers: 2 transfers × batches × time steps = significant overhead

**Fix**: Move adjacency matrix to device during initialization and keep it there
```csharp
// In constructor (after RegisterComponents):
_adj = _adj.to(device);

// In forward() - use directly
var adjBatch = _adj.unsqueeze(0);
var agg = matmul(adjBatch, h);
```

### 2. **Data Loading Blocking (HumanML3DDataset.cs:243-279)**
**Problem**: Synchronous tensor creation and device transfer in main training loop
```csharp
// CreateTensor happens sequentially, blocks GPU
var tensor = from_array(flatArray);
tensor = tensor.reshape(batchSize, featureSize);
return tensor.to(device);  // CPU→GPU transfer here
```

**Impact**: GPU idle while waiting for data loading
- Batch assembly: ~5-10ms per batch
- Tensor creation: ~2-5ms per batch
- Memory copy to GPU: variable based on batch size

**Optimization**: Pre-load next batch while GPU processes current batch (requires async refactor)

### 3. **Missing GPU Synchronization Points**
**Problem**: No explicit synchronization between operations means timing is inaccurate
- Model forward/backward may not actually complete when we measure
- Loss computation on GPU doesn't sync before returning to CPU

**Partial Fix**: Added `CudaDebugger.SynchronizeGpu()` calls in hot paths

### 4. **Tensor Reallocation Each Batch**
**Problem**: GetBatch() allocates fresh tensors every iteration
```csharp
var textEmbTensor = CreateTensor(textEmbList, device);  // New allocation
var motionTensor = CreateTensor(motionFramesList, device);  // New allocation
```

**Impact**: Fragmentation, extra allocation overhead
- Consider tensor pool or pre-allocated buffers for fixed batch sizes

---

## Logging Output Interpretation

### Per-Batch Timing
```
[TIMING] Batch 1: data=X ms, fwd=Y ms, loss=Z ms, bwd=A ms, step=B ms
```

**Expected Breakdown** (for CPU):
- data: 5-15ms (data assembly + CPU→GPU transfer)
- fwd: 10-50ms (model inference)
- loss: 1-5ms (MSE loss)
- bwd: 20-100ms (backprop)
- step: 5-20ms (optimizer update)

**GPU Utilization Check**:
- If `data` > `fwd + bwd` = GPU idle during data loading
- If `fwd ≈ bwd` ≈ normal (backprop ~1.5-2× forward)
- If `step` > 10ms = memory limited, not compute bound

### GPU Memory
```
=== GPU MEMORY ===
Allocated: XMB
Reserved: YMB
Total: ZMB
Available: ~AMB
```

- **Allocated**: Actual tensors in use
- **Reserved**: Pool reserved by TorchSharp runtime
- If reserved >> allocated: memory fragmentation

---

## Quick Wins (Priority Order)

### P0: Fix Adjacency Matrix Transfer
File: `MotionModel/GcnSpatialTemporalModel.cs`
- Make `_adj` a class field, move to device once in constructor
- Expected gain: 5-10% wall-clock time

### P1: Add Device Awareness to Model
Current: Model takes `IOptions<GcnSpatialTemporalConfig>` + `IOptions<DatasetSettings>`
Add: Pass Device as parameter so model can initialize tensors on correct device

### P2: Profile Data Loading vs Model
Once timings show bottleneck clearly, consider:
- Async batch prefetch (spawn data loading thread)
- Batch pool (reuse tensor allocations)
- DataLoader with background workers

---

## How to Use New Profiling Tools

### 1. Check Device Detection
```
=== DEVICE INFO ===
Device Type: CUDA
Device Index: 0
CUDA Available: True
Current CUDA Device: 0
```
If `CUDA Available: False` = GPU not being used at all

### 2. Watch Memory Growth
```
=== GPU MEMORY ===
Allocated: 500MB
Reserved: 1000MB
```
If reserved >> allocated = model too large or many small allocations

### 3. First-Batch Timing
```
[TIMING] Batch 1: data=12ms, fwd=18ms, loss=1ms, bwd=35ms, step=8ms
```
- Compare across epochs: if consistent, not a leak
- data > fwd+bwd = data loading is bottleneck, not GPU

---

## Next Steps

1. Run training with logging enabled (you already modified trainer)
2. Capture one full epoch output
3. Look for:
   - Are we on CUDA? (check device info)
   - What's the slowest operation per batch?
   - Is GPU memory growing?
   - Is `data` time >> `fwd+bwd` time?
4. Report findings, then we optimize accordingly
