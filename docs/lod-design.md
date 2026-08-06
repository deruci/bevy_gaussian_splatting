# Streamed SOG LOD — design

Port of the PlayCanvas streamed-LOD architecture (engine `src/scene/gsplat-unified/`,
`splat-transform` `write-lod.ts` / `read-sog.ts`, both MIT) onto this crate.
Offline tooling is reused verbatim: LOD generation, decimation, chunking and
encoding stay in `splat-transform`; this crate only ever *reads* its output.

## Source formats

### SOG v2 (single splat payload)

A SOG unit is a `meta.json` plus WebP textures, optionally bundled as one `.sog`
(ZIP, stored or deflate). `meta.json`:

```jsonc
{
  "version": 2,
  "count": 123456,                 // gaussians
  "means":  { "mins": [x,y,z], "maxs": [x,y,z], "files": ["means_l.webp", "means_u.webp"] },
  "scales": { "codebook": [f32; 256], "files": ["scales.webp"] },
  "quats":  { "files": ["quats.webp"] },
  "sh0":    { "codebook": [f32; 256], "files": ["sh0.webp"] },
  "shN":    { "count": N, "bands": 1|2|3, "codebook": [f32; 256],
              "files": ["shN_centroids.webp", "shN_labels.webp"] }   // optional
}
```

Per-gaussian texel `g` (all textures RGBA8, row-major, width*height >= count;
splats are Morton-ordered):

| attribute | decode |
|---|---|
| position | 16-bit `v = lo[c] \| hi[c]<<8` per channel; `p = inv_log(min + (max-min) * v/65535)` where `inv_log(v) = sign(v)*(exp(\|v\|)-1)` |
| rotation | smallest-three: alpha tag `252+maxComp`; components `(b/255*2-1)/sqrt(2)` placed per `QUAT_IDX`, max component reconstructed positive; output order `[w,x,y,z]` |
| scale | `scales.codebook[byte]` = *log* scale (PLY convention) |
| opacity | `sh0` alpha byte / 255 = already-sigmoided opacity |
| SH dc | `sh0.codebook[byte]` per channel |
| SH rest | 16-bit label from `shN_labels` (r,g channels); centroid texel row `label/64`, column `(label%64)*coeffs + j`; channel value indexes `shN.codebook`; layout channel-major per coefficient |

### Crate-side convention mapping (matches `io/ply.rs`)

- `scale`: `exp(codebook value)` (linear, like the PLY path post-`exp()`).
- `opacity`: use the alpha byte directly (PLY path stores sigmoided opacity).
- `rotation`: `[w,x,y,z]` normalized — same as PLY `rot_0..rot_3`.
- SH storage is interleaved: index `= coefficient * SH_CHANNELS + channel`;
  dc at 0..2. SOG shN is channel-major per coefficient → re-interleave on read.
  Bands beyond the compiled `SH_DEGREE` are dropped; missing bands are zeroed.
- Cloud padded with default gaussians to a multiple of 32 (sort requirement).

### lod-meta.json (streamed SOG)

Written by `splat-transform` (`write-lod.ts`), consumed by the engine octree
parser. Despite the name, the spatial index is a binary kd-tree; only leaves
matter at runtime.

```jsonc
{
  "version": 1,
  "asset": { "generator": "splat-transform vX" },
  "count": 999,            // total splats
  "counts": [n0, n1, ...], // per LOD level
  "lodLevels": L,
  "environment": "env/meta.json",      // optional skydome unit
  "filenames": ["0_0/meta.json", "1_0/meta.json", ...],  // SOG units, named {lod}_{index}
  "tree": {
    "bound": { "min": [x,y,z], "max": [x,y,z] },
    "children": [ <node>, <node> ],     // interior
    "lods": { "0": { "file": fi, "offset": o, "count": c }, "1": {...} }  // leaf
  }
}
```

Invariant that makes the runtime cheap: each leaf×LOD is one **contiguous,
Morton-ordered row range** inside one SOG unit. Loading a leaf at LOD `l` is a
sub-range decode of unit `filenames[file]`.

## Phasing

1. **`io_sog` feature — SOG v2 reader** (this PR series):
   `src/io/sog.rs`. Decodes bundled `.sog` (ZIP → single-file Bevy asset) and
   unbundled `meta.json` + sibling textures. Core API decodes an arbitrary row
   range so the LOD loader can reuse it for leaf intervals. WebP via the `image`
   crate (pure Rust, MIT/Apache), ZIP via `zip` (MIT); `serde_json` already a dep.
2. **lod-meta asset**: parse to a flat leaf array
   (`Vec<LodLeaf { aabb, lods: [Option<Interval>] }>`), Bevy asset type +
   loader keyed on the `lod-meta.json` file name. MVP runtime: spawn one
   `PlanarGaussian3d` entity per leaf at a fixed LOD (reuses the existing
   multi-cloud `GaussianScene` spawn pattern; per-cloud sort is acceptable at
   house scale, ~tens of leaves).
3. **Distance-band LOD selection** (port of `gsplat-octree-instance.js`):
   per-leaf LOD from camera distance bands `base * mult^i` (defaults 5, 3),
   FOV-compensated, re-evaluated on >1 m camera movement; underfill (show
   already-loaded coarser LOD while the target streams); refcount + cooldown
   eviction of unit assets mapped onto Bevy `Handle` semantics.
4. **Merged work buffer + single global sort** (port of the work-buffer /
   `gsplat-budget-balancer.js` design, wgpu-native): block-allocated global
   buffer, compute gather of active intervals, one radix sort across clouds,
   splat budget with sqrt-distance bucket demotion. Replaces per-cloud sort.

Phases 1–2 are the current milestone; 3 is runtime logic with no new formats;
4 is the perf endgame and independent of 1–3 correctness.
