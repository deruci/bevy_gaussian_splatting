//! Composite cloud: one budget-sized GPU cloud per LOD scene.
//!
//! Port of the PlayCanvas work-buffer design (`gsplat-work-buffer.js`, MIT):
//! instead of one cloud entity per kd-tree leaf, the streaming runtime rents
//! block ranges inside a single fixed-capacity cloud and patches leaf swaps
//! straight into its GPU plane buffers (`write_buffer`), so the whole scene
//! renders with one global depth sort and one draw — inter-leaf blending
//! order becomes exact, and per-leaf sort dispatch overhead disappears.
//!
//! The composite asset itself is never mutated on the CPU after creation;
//! only its GPU buffers change. Consequently composite mode requires a GPU
//! sort (radix/bitonic) — CPU sorts read the main-world asset and would sort
//! stale data.

use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        render_asset::RenderAssets,
        renderer::RenderQueue,
    },
};

use bevy_interleave::prelude::Planar;

use crate::gaussian::formats::planar_3d::{PlanarGaussian3d, PlanarStorageGaussian3d};

/// First-fit block allocator over splat indices `0..capacity`, with free-list
/// coalescing. Blocks are leaf-interval sized (thousands to ~512k splats), so
/// the free list stays short.
#[derive(Debug)]
pub struct BlockAllocator {
    capacity: usize,
    /// disjoint, sorted, coalesced (offset, len) spans
    free: Vec<(usize, usize)>,
}

impl BlockAllocator {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free: vec![(0, capacity)],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn used(&self) -> usize {
        self.capacity - self.free.iter().map(|(_, len)| len).sum::<usize>()
    }

    pub fn alloc(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        let slot = self.free.iter().position(|&(_, free_len)| free_len >= len)?;
        let (offset, free_len) = self.free[slot];
        if free_len == len {
            self.free.remove(slot);
        } else {
            self.free[slot] = (offset + len, free_len - len);
        }
        Some(offset)
    }

    pub fn free(&mut self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let index = self
            .free
            .partition_point(|&(free_offset, _)| free_offset < offset);

        debug_assert!(
            index == 0 || {
                let (prev_offset, prev_len) = self.free[index - 1];
                prev_offset + prev_len <= offset
            },
            "double free or overlapping free"
        );

        self.free.insert(index, (offset, len));

        // coalesce with the next span, then the previous one
        if index + 1 < self.free.len() {
            let (next_offset, next_len) = self.free[index + 1];
            if offset + len == next_offset {
                self.free[index].1 += next_len;
                self.free.remove(index + 1);
            }
        }
        if index > 0 {
            let (prev_offset, prev_len) = self.free[index - 1];
            if prev_offset + prev_len == self.free[index].0 {
                self.free[index - 1].1 += self.free[index].1;
                self.free.remove(index);
            }
        }
    }
}

/// A GPU patch destined for the composite cloud's plane buffers.
pub enum CompositeWrite {
    /// upload a decoded block at splat index `offset`
    Block {
        asset: AssetId<PlanarGaussian3d>,
        offset: usize,
        data: PlanarGaussian3d,
    },
    /// hide a freed block: zero its scale/opacity plane (positions and SH may
    /// keep stale bytes — zero scale and opacity render nothing)
    Clear {
        asset: AssetId<PlanarGaussian3d>,
        offset: usize,
        len: usize,
    },
}

/// Cross-world write queue: the streaming runtime (main world) pushes patches,
/// the render-world system drains them into `write_buffer` calls. Writes for
/// an asset whose GPU buffers aren't prepared yet are retried next frame.
#[derive(Resource, Clone, Default)]
pub struct CompositeWriteQueue(pub Arc<Mutex<Vec<CompositeWrite>>>);

impl CompositeWriteQueue {
    pub fn push(&self, write: CompositeWrite) {
        self.0.lock().unwrap().push(write);
    }

    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Default)]
pub struct CompositeCloudPlugin;

impl Plugin for CompositeCloudPlugin {
    fn build(&self, app: &mut App) {
        let queue = CompositeWriteQueue::default();
        app.insert_resource(queue.clone());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(queue);
            render_app.add_systems(
                Render,
                apply_composite_writes.in_set(RenderSystems::PrepareResources),
            );
        }
    }
}

fn apply_composite_writes(
    write_queue: Res<CompositeWriteQueue>,
    gpu_clouds: Res<RenderAssets<PlanarStorageGaussian3d>>,
    render_queue: Res<RenderQueue>,
) {
    let mut queue = write_queue.0.lock().unwrap();
    if queue.is_empty() {
        return;
    }

    let mut retry = Vec::new();

    for write in queue.drain(..) {
        let asset = match &write {
            CompositeWrite::Block { asset, .. } | CompositeWrite::Clear { asset, .. } => *asset,
        };
        let Some(gpu) = gpu_clouds.get(asset) else {
            retry.push(write); // GPU buffers not prepared yet
            continue;
        };

        match write {
            CompositeWrite::Block { offset, data, .. } => {
                let end = offset + data.len();
                if end > gpu.count {
                    warn!("composite write {offset}..{end} exceeds capacity {}", gpu.count);
                    continue;
                }

                write_plane(&render_queue, &gpu.position_visibility, offset, &data.position_visibility);
                write_plane(&render_queue, &gpu.spherical_harmonic, offset, &data.spherical_harmonic);
                write_plane(&render_queue, &gpu.rotation, offset, &data.rotation);
                write_plane(&render_queue, &gpu.scale_opacity, offset, &data.scale_opacity);
            }
            CompositeWrite::Clear { offset, len, .. } => {
                let end = offset + len;
                if end > gpu.count {
                    warn!("composite clear {offset}..{end} exceeds capacity {}", gpu.count);
                    continue;
                }

                let zeros =
                    vec![crate::gaussian::f32::ScaleOpacity::default(); len];
                write_plane(&render_queue, &gpu.scale_opacity, offset, &zeros);
            }
        }
    }

    *queue = retry;
}

fn write_plane<T: bytemuck::Pod>(
    render_queue: &RenderQueue,
    buffer: &bevy::render::render_resource::Buffer,
    splat_offset: usize,
    data: &[T],
) {
    let stride = std::mem::size_of::<T>();
    render_queue.write_buffer(
        buffer,
        (splat_offset * stride) as u64,
        bytemuck::cast_slice(data),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_first_fit_and_exhaustion() {
        let mut allocator = BlockAllocator::new(100);
        assert_eq!(allocator.alloc(40), Some(0));
        assert_eq!(allocator.alloc(40), Some(40));
        assert_eq!(allocator.alloc(40), None); // only 20 left
        assert_eq!(allocator.alloc(20), Some(80));
        assert_eq!(allocator.used(), 100);
        assert_eq!(allocator.alloc(1), None);
    }

    #[test]
    fn free_coalesces_both_sides() {
        let mut allocator = BlockAllocator::new(90);
        let a = allocator.alloc(30).unwrap();
        let b = allocator.alloc(30).unwrap();
        let c = allocator.alloc(30).unwrap();

        allocator.free(a, 30);
        allocator.free(c, 30);
        // fragmented: two disjoint 30-spans, no room for 40
        assert_eq!(allocator.alloc(40), None);

        allocator.free(b, 30);
        // all coalesced back into one span
        assert_eq!(allocator.used(), 0);
        assert_eq!(allocator.alloc(90), Some(0));
    }

    #[test]
    fn free_reuses_space() {
        let mut allocator = BlockAllocator::new(100);
        let a = allocator.alloc(60).unwrap();
        assert_eq!(allocator.alloc(60), None);
        allocator.free(a, 60);
        assert_eq!(allocator.alloc(60), Some(0));
    }

    #[test]
    fn zero_len_noops() {
        let mut allocator = BlockAllocator::new(10);
        assert_eq!(allocator.alloc(0), None);
        allocator.free(0, 0);
        assert_eq!(allocator.used(), 0);
    }
}
