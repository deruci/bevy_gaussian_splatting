//! Distance-based LOD streaming for SOG LOD scenes.
//!
//! Port of the PlayCanvas streamed-LOD runtime (`gsplat-octree-instance.js`,
//! MIT) onto Bevy systems. Per kd-tree leaf, a desired LOD level is picked
//! from FOV-compensated camera distance bands (`base * multiplier^i`),
//! re-evaluated when the camera moves beyond a threshold. SOG units load and
//! decode in background tasks; a leaf keeps showing its current cloud until
//! the replacement is ready (underfill), then swaps atomically. Unit textures
//! are cached with ref-counting and cooldown eviction.
//!
//! Not yet ported: step-wise coarse-first prefetch and the global splat
//! budget balancer (see docs/lod-design.md phase 4).

use std::collections::HashMap;
use std::sync::Arc;

use bevy::{
    asset::AssetPath,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use bevy_interleave::prelude::Planar;

use bevy::camera::primitives::Aabb;

use crate::{
    camera::GaussianCamera,
    gaussian::{
        formats::planar_3d::{Gaussian3d, PlanarGaussian3d, PlanarGaussian3dHandle},
        settings::CloudSettings,
    },
    io::{
        lod::{GaussianLodScene, GaussianLodSceneHandle},
        sog::{SogMeta, SogTextures, decode_range, pad_to_multiple_of_32},
    },
    stream::composite::{BlockAllocator, CompositeWrite, CompositeWriteQueue},
};

/// Tunables for distance-band LOD selection, mirroring the PlayCanvas
/// defaults (`lodBaseDistance` 5, `lodMultiplier` 3, `lodUpdateDistance` 1,
/// eviction cooldown 100 ticks).
#[derive(Component, Clone, Debug, Reflect)]
pub struct LodSettings {
    /// distance of the first LOD band, in scene units
    pub base_distance: f32,
    /// each further band spans `base_distance * multiplier^i`
    pub multiplier: f32,
    /// re-evaluate only after the camera moves this far
    pub update_distance: f32,
    /// pin every leaf to one level (bypasses distance selection)
    pub forced_lod: Option<usize>,
    /// frames a unit stays cached after its last user releases it
    pub cooldown_frames: u32,
    /// cap on concurrent leaf interval decodes
    pub max_decodes_in_flight: usize,
    /// render all leaves through one budget-sized composite cloud (one global
    /// sort + one draw, exact inter-leaf blending). Requires a GPU sort —
    /// CPU sorts read the main-world asset, which composite mode never
    /// mutates. `false` spawns one cloud entity per leaf instead.
    pub composite: bool,
    /// composite cloud capacity in splats; when a leaf's desired level does
    /// not fit, coarser levels are tried (degraded budget balancing)
    pub splat_budget: usize,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            base_distance: 5.0,
            multiplier: 3.0,
            update_distance: 1.0,
            forced_lod: None,
            cooldown_frames: 100,
            max_decodes_in_flight: 4,
            composite: true,
            splat_budget: 2_000_000,
        }
    }
}

/// Distance-band level pick: level `i` covers `[base * mult^(i-1), base * mult^i)`.
pub fn select_lod_level(distance: f32, base_distance: f32, multiplier: f32, levels: usize) -> usize {
    if levels <= 1 {
        return 0;
    }
    let mut threshold = base_distance;
    for level in 0..levels - 1 {
        if distance < threshold {
            return level;
        }
        threshold *= multiplier;
    }
    levels - 1
}

/// A fully decoded SOG unit held resident for interval expansion.
pub struct DecodedUnit {
    pub meta: SogMeta,
    pub textures: SogTextures,
}

enum UnitState {
    Loading(Task<Result<DecodedUnit, String>>),
    Ready(Arc<DecodedUnit>),
    Failed,
}

struct UnitEntry {
    state: UnitState,
    /// number of in-flight leaf decodes reading this unit
    refs: usize,
    cooldown: u32,
    cooldown_reset: u32,
}

/// Cache of decoded SOG units, keyed by (scene asset, unit file index).
#[derive(Resource, Default)]
pub struct SogUnitCache {
    units: HashMap<(AssetId<GaussianLodScene>, usize), UnitEntry>,
}

impl SogUnitCache {
    /// Number of units currently resident (loading or decoded).
    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }
}

struct PendingSwap {
    /// level actually being decoded (may be coarser than the wish under budget)
    lod: usize,
    /// the desired level this swap was started for; a change here cancels it
    goal: usize,
    file: usize,
    /// (offset, len) reserved in the composite allocator; None in entity mode
    block: Option<(usize, usize)>,
    task: Task<Result<Vec<Gaussian3d>, String>>,
}

/// Per-scene streaming state, inserted once the `GaussianLodScene` asset is
/// available. Indices parallel `GaussianLodScene::leaves`.
#[derive(Component)]
pub struct LodRuntime {
    desired: Vec<Option<usize>>,
    active: Vec<Option<usize>>,
    entities: Vec<Option<Entity>>,
    pending: Vec<Option<PendingSwap>>,
    /// composite mode: (offset, len) block currently displayed per leaf
    blocks: Vec<Option<(usize, usize)>>,
    allocator: Option<BlockAllocator>,
    composite: Option<Handle<PlanarGaussian3d>>,
    last_eval: Option<Vec3>,
    last_params: Option<(Option<usize>, f32, f32)>,
}

impl LodRuntime {
    fn new(leaves: usize) -> Self {
        Self {
            desired: vec![None; leaves],
            active: vec![None; leaves],
            entities: vec![None; leaves],
            pending: (0..leaves).map(|_| None).collect(),
            blocks: vec![None; leaves],
            allocator: None,
            composite: None,
            last_eval: None,
            last_params: None,
        }
    }

    /// Splats currently resident in the composite cloud (None in entity mode).
    pub fn composite_used(&self) -> Option<usize> {
        self.allocator.as_ref().map(BlockAllocator::used)
    }

    /// Handle of the composite cloud asset (None in entity mode).
    pub fn composite_handle(&self) -> Option<&Handle<PlanarGaussian3d>> {
        self.composite.as_ref()
    }

    /// (leaf index, active level) for every currently displayed leaf.
    pub fn active_lods(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(i, lod)| lod.map(|lod| (i, lod)))
    }

    pub fn in_flight(&self) -> usize {
        self.pending.iter().filter(|p| p.is_some()).count()
    }
}

fn cancel_pending(
    pending: &mut Option<PendingSwap>,
    cache: &mut SogUnitCache,
    scene_id: AssetId<GaussianLodScene>,
    allocator: &mut Option<BlockAllocator>,
) {
    if let Some(swap) = pending.take() {
        if let Some(entry) = cache.units.get_mut(&(scene_id, swap.file)) {
            entry.refs = entry.refs.saturating_sub(1);
        }
        // reserved but never written: release without clearing
        if let (Some(allocator), Some((offset, len))) = (allocator.as_mut(), swap.block) {
            allocator.free(offset, len);
        }
    }
}

#[derive(Default)]
pub struct SogLodStreamingPlugin;

impl Plugin for SogLodStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LodSettings>();
        app.init_resource::<SogUnitCache>();
        app.add_plugins(crate::stream::composite::CompositeCloudPlugin);

        app.add_systems(
            Update,
            (
                init_lod_runtime,
                evaluate_lod,
                manage_units,
                apply_swaps,
                evict_units,
            )
                .chain(),
        );
    }
}

fn init_lod_runtime(
    mut commands: Commands,
    scenes: Res<Assets<GaussianLodScene>>,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    handles: Query<(Entity, &GaussianLodSceneHandle, &LodSettings), Without<LodRuntime>>,
) {
    for (entity, handle, settings) in &handles {
        let Some(scene) = scenes.get(&handle.0) else {
            continue;
        };

        let mut runtime = LodRuntime::new(scene.leaves.len());

        if settings.composite {
            let capacity = settings.splat_budget;
            let cloud_handle = clouds.add(PlanarGaussian3d::from_interleaved(vec![
                Gaussian3d::default();
                capacity
            ]));

            // frustum-culling bounds: the union of all leaves — the composite
            // asset's own positions are meaningless (default until streamed)
            let mut min = Vec3::MAX;
            let mut max = Vec3::MIN;
            for leaf in &scene.leaves {
                min = min.min(leaf.min);
                max = max.max(leaf.max);
            }
            if scene.leaves.is_empty() {
                min = Vec3::ZERO;
                max = Vec3::ZERO;
            }

            commands.entity(entity).with_children(|builder| {
                builder.spawn((
                    PlanarGaussian3dHandle(cloud_handle.clone()),
                    CloudSettings::default(),
                    Name::new("lod_composite"),
                    Transform::default(),
                    Visibility::default(),
                    Aabb::from_min_max(min, max),
                ));
            });

            runtime.allocator = Some(BlockAllocator::new(capacity));
            runtime.composite = Some(cloud_handle);
        }

        commands.entity(entity).insert(runtime);
    }
}

fn evaluate_lod(
    cameras: Query<(&GlobalTransform, &Projection), With<GaussianCamera>>,
    scenes: Res<Assets<GaussianLodScene>>,
    mut runtimes: Query<(
        &GaussianLodSceneHandle,
        &GlobalTransform,
        &LodSettings,
        &mut LodRuntime,
    )>,
) {
    let Some((camera_transform, projection)) = cameras.iter().next() else {
        return;
    };

    // narrower FOV magnifies the scene, so scale distances down to keep
    // on-screen detail roughly constant (reference FOV: 60 degrees)
    let fov = match projection {
        Projection::Perspective(perspective) => perspective.fov,
        _ => std::f32::consts::FRAC_PI_3,
    };
    let fov_scale = (fov * 0.5).tan() / std::f32::consts::FRAC_PI_6.tan();

    for (handle, scene_transform, settings, mut runtime) in &mut runtimes {
        let Some(scene) = scenes.get(&handle.0) else {
            continue;
        };

        // leaf bounds are scene-local; move the camera into that space
        // (assumes roughly uniform scene scale)
        let local_camera = scene_transform
            .affine()
            .inverse()
            .transform_point3(camera_transform.translation());

        let params = (
            settings.forced_lod,
            settings.base_distance,
            settings.multiplier,
        );
        let moved = runtime
            .last_eval
            .is_none_or(|last| last.distance(local_camera) > settings.update_distance);
        if !moved && runtime.last_params == Some(params) {
            continue;
        }

        for (i, leaf) in scene.leaves.iter().enumerate() {
            let level = settings.forced_lod.unwrap_or_else(|| {
                let distance = leaf.distance_squared(local_camera).sqrt() * fov_scale;
                select_lod_level(
                    distance,
                    settings.base_distance,
                    settings.multiplier,
                    scene.lod_levels,
                )
            });
            runtime.desired[i] = leaf.nearest_available(level);
        }

        runtime.last_eval = Some(local_camera);
        runtime.last_params = Some(params);
    }
}

fn start_unit_load(asset_server: &AssetServer, unit_path: String) -> Task<Result<DecodedUnit, String>> {
    let server = asset_server.clone();

    AsyncComputeTaskPool::get().spawn(async move {
        let read = |relative: String| {
            let server = &server;
            async move {
                let path = AssetPath::try_parse(&relative)
                    .map_err(|e| format!("invalid path '{relative}': {e}"))?;
                let source = server
                    .get_source(path.source().clone())
                    .map_err(|e| format!("{relative}: {e}"))?;
                let mut reader = source
                    .reader()
                    .read(path.path())
                    .await
                    .map_err(|e| format!("{relative}: {e}"))?;
                let mut bytes = Vec::new();
                reader
                    .read_to_end(&mut bytes)
                    .await
                    .map_err(|e| format!("{relative}: {e}"))?;
                Ok::<_, String>(bytes)
            }
        };

        let meta_bytes = read(unit_path.clone()).await?;
        let meta = SogMeta::from_json(&meta_bytes).map_err(|e| e.to_string())?;

        let unit_dir = unit_path.rsplit_once('/').map(|(dir, _)| dir.to_owned());
        let names: Vec<String> = meta
            .texture_files()
            .into_iter()
            .map(str::to_owned)
            .collect();

        let mut files: HashMap<String, Vec<u8>> = HashMap::new();
        for name in names {
            let relative = match &unit_dir {
                Some(dir) => format!("{dir}/{name}"),
                None => name.clone(),
            };
            files.insert(name, read(relative).await?);
        }

        let textures = meta
            .load_textures(|name| {
                files.get(name).cloned().ok_or_else(|| {
                    std::io::Error::other(format!("missing texture {name}"))
                })
            })
            .map_err(|e| e.to_string())?;

        Ok(DecodedUnit { meta, textures })
    })
}

fn manage_units(
    mut cache: ResMut<SogUnitCache>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianLodScene>>,
    mut runtimes: Query<(&GaussianLodSceneHandle, &LodSettings, &mut LodRuntime)>,
) {
    for (handle, settings, mut runtime) in &mut runtimes {
        let Some(scene) = scenes.get(&handle.0) else {
            continue;
        };
        let scene_id = handle.0.id();

        let mut budget = settings
            .max_decodes_in_flight
            .saturating_sub(runtime.in_flight());

        let LodRuntime {
            desired,
            active,
            pending,
            allocator,
            ..
        } = &mut *runtime;

        for i in 0..scene.leaves.len() {
            let Some(wish) = desired[i] else {
                continue;
            };

            // pending decode toward a stale goal: cancel it (a budget-degraded
            // level is fine as long as the wish itself hasn't changed)
            if pending[i].as_ref().is_some_and(|swap| swap.goal != wish) {
                cancel_pending(&mut pending[i], &mut cache, scene_id, allocator);
                budget += 1;
            }

            if active[i] == Some(wish) || pending[i].is_some() {
                continue;
            }

            // composite: reserve a block up front; when the desired level does
            // not fit the budget, degrade to the nearest coarser level that
            // does (stopping at a level already displayed)
            let mut target = wish;
            let mut block = None;
            if let Some(allocator) = allocator.as_mut() {
                let mut reserved = None;
                for level in wish..scene.leaves[i].lods.len() {
                    let Some(interval) = &scene.leaves[i].lods[level] else {
                        continue;
                    };
                    if active[i] == Some(level) {
                        break; // already showing this or an acceptable coarser level
                    }
                    if let Some(offset) = allocator.alloc(interval.count) {
                        reserved = Some((level, (offset, interval.count)));
                        break;
                    }
                }
                let Some((level, reservation)) = reserved else {
                    continue; // nothing fits (or already showing the fallback)
                };
                target = level;
                block = Some(reservation);
            }

            let Some(interval) = scene.leaves[i].lods[target].clone() else {
                continue;
            };

            let entry = cache
                .units
                .entry((scene_id, interval.file))
                .or_insert_with(|| UnitEntry {
                    state: UnitState::Loading(start_unit_load(
                        &asset_server,
                        scene.unit_path(&scene.filenames[interval.file]),
                    )),
                    refs: 0,
                    cooldown: settings.cooldown_frames,
                    cooldown_reset: settings.cooldown_frames,
                });

            let ready = matches!(entry.state, UnitState::Ready(_));
            if !ready || budget == 0 {
                // unit still loading/failed, or decode budget exhausted:
                // release the reservation and retry on a later frame
                if let (Some(allocator), Some((offset, len))) = (allocator.as_mut(), block) {
                    allocator.free(offset, len);
                }
                continue;
            }
            let UnitState::Ready(unit) = &entry.state else {
                unreachable!();
            };
            budget -= 1;

            let unit = unit.clone();
            let pad = !settings.composite;
            let task = AsyncComputeTaskPool::get().spawn(async move {
                decode_range(&unit.meta, &unit.textures, interval.offset, interval.count)
                    .map(|gaussians| {
                        if pad {
                            pad_to_multiple_of_32(gaussians)
                        } else {
                            gaussians
                        }
                    })
                    .map_err(|e| e.to_string())
            });

            entry.refs += 1;
            entry.cooldown = entry.cooldown_reset;
            pending[i] = Some(PendingSwap {
                lod: target,
                goal: wish,
                file: interval.file,
                block,
                task,
            });
        }
    }

    for ((_, file), entry) in cache.units.iter_mut() {
        if let UnitState::Loading(task) = &mut entry.state
            && let Some(result) = block_on(poll_once(task))
        {
            entry.state = match result {
                Ok(unit) => UnitState::Ready(Arc::new(unit)),
                Err(error) => {
                    warn!("failed to load SOG unit {file}: {error}");
                    UnitState::Failed
                }
            };
        }
    }
}

fn apply_swaps(
    mut commands: Commands,
    mut clouds: ResMut<Assets<PlanarGaussian3d>>,
    mut cache: ResMut<SogUnitCache>,
    write_queue: Res<CompositeWriteQueue>,
    mut runtimes: Query<(Entity, &GaussianLodSceneHandle, &mut LodRuntime)>,
) {
    for (scene_entity, handle, mut runtime) in &mut runtimes {
        let scene_id = handle.0.id();

        let LodRuntime {
            active,
            entities,
            pending,
            blocks,
            allocator,
            composite,
            ..
        } = &mut *runtime;

        for i in 0..pending.len() {
            let Some(swap) = &mut pending[i] else {
                continue;
            };
            let Some(result) = block_on(poll_once(&mut swap.task)) else {
                continue;
            };

            let (lod, file, block) = (swap.lod, swap.file, swap.block);
            pending[i] = None;
            if let Some(entry) = cache.units.get_mut(&(scene_id, file)) {
                entry.refs = entry.refs.saturating_sub(1);
            }

            match result {
                Ok(gaussians) => {
                    if let (Some(allocator), Some(composite), Some((offset, len))) =
                        (allocator.as_mut(), composite.as_ref(), block)
                    {
                        // composite: patch the block into the GPU buffers; the
                        // previous block stays visible until the same queue
                        // drain clears it (underfill, exact swap)
                        debug_assert_eq!(gaussians.len(), len);
                        write_queue.push(CompositeWrite::Block {
                            asset: composite.id(),
                            offset,
                            data: PlanarGaussian3d::from_interleaved(gaussians),
                        });

                        if let Some((old_offset, old_len)) = blocks[i].replace((offset, len)) {
                            allocator.free(old_offset, old_len);
                            write_queue.push(CompositeWrite::Clear {
                                asset: composite.id(),
                                offset: old_offset,
                                len: old_len,
                            });
                        }
                        active[i] = Some(lod);
                    } else {
                        let cloud_handle =
                            clouds.add(PlanarGaussian3d::from_interleaved(gaussians));

                        let mut child = Entity::PLACEHOLDER;
                        commands.entity(scene_entity).with_children(|builder| {
                            child = builder
                                .spawn((
                                    PlanarGaussian3dHandle(cloud_handle),
                                    CloudSettings::default(),
                                    Name::new(format!("lod_leaf_{i}_lod{lod}")),
                                    Transform::default(),
                                    Visibility::default(),
                                ))
                                .id();
                        });

                        // underfill swap: the previous level stays visible until
                        // this exact frame, so the leaf never disappears
                        if let Some(old) = entities[i].replace(child) {
                            commands.entity(old).despawn();
                        }
                        active[i] = Some(lod);
                    }
                }
                Err(error) => {
                    // release the reservation so the space isn't leaked
                    if let (Some(allocator), Some((offset, len))) = (allocator.as_mut(), block) {
                        allocator.free(offset, len);
                    }
                    warn!("leaf {i} interval decode failed: {error}");
                }
            }
        }
    }
}

fn evict_units(mut cache: ResMut<SogUnitCache>) {
    cache.units.retain(|_, entry| {
        if entry.refs > 0 {
            entry.cooldown = entry.cooldown_reset;
            return true;
        }
        if matches!(entry.state, UnitState::Loading(_)) {
            return true; // let the load finish; evicted next once idle
        }
        entry.cooldown = entry.cooldown.saturating_sub(1);
        entry.cooldown > 0
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_selection() {
        // bands with base 5, multiplier 3: [0,5) [5,15) [15,45) [45,inf)
        assert_eq!(select_lod_level(0.0, 5.0, 3.0, 4), 0);
        assert_eq!(select_lod_level(4.9, 5.0, 3.0, 4), 0);
        assert_eq!(select_lod_level(5.0, 5.0, 3.0, 4), 1);
        assert_eq!(select_lod_level(14.9, 5.0, 3.0, 4), 1);
        assert_eq!(select_lod_level(15.0, 5.0, 3.0, 4), 2);
        assert_eq!(select_lod_level(44.9, 5.0, 3.0, 4), 2);
        assert_eq!(select_lod_level(45.0, 5.0, 3.0, 4), 3);
        assert_eq!(select_lod_level(1e9, 5.0, 3.0, 4), 3);
    }

    #[test]
    fn band_selection_clamps_levels() {
        assert_eq!(select_lod_level(1e9, 5.0, 3.0, 1), 0);
        assert_eq!(select_lod_level(1e9, 5.0, 3.0, 2), 1);
        assert_eq!(select_lod_level(0.0, 5.0, 3.0, 0), 0);
    }
}
