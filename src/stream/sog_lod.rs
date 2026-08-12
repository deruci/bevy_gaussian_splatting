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
    lod: usize,
    file: usize,
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
            last_eval: None,
            last_params: None,
        }
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
) {
    if let Some(swap) = pending.take()
        && let Some(entry) = cache.units.get_mut(&(scene_id, swap.file))
    {
        entry.refs = entry.refs.saturating_sub(1);
    }
}

#[derive(Default)]
pub struct SogLodStreamingPlugin;

impl Plugin for SogLodStreamingPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LodSettings>();
        app.init_resource::<SogUnitCache>();

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
    handles: Query<(Entity, &GaussianLodSceneHandle), Without<LodRuntime>>,
) {
    for (entity, handle) in &handles {
        if let Some(scene) = scenes.get(&handle.0) {
            commands
                .entity(entity)
                .insert(LodRuntime::new(scene.leaves.len()));
        }
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

        for i in 0..scene.leaves.len() {
            let Some(desired) = runtime.desired[i] else {
                continue;
            };

            // pending decode toward a stale level: cancel it
            if runtime.pending[i]
                .as_ref()
                .is_some_and(|swap| swap.lod != desired)
            {
                cancel_pending(&mut runtime.pending[i], &mut cache, scene_id);
                budget += 1;
            }

            if runtime.active[i] == Some(desired) || runtime.pending[i].is_some() {
                continue;
            }

            let Some(interval) = scene.leaves[i].lods[desired].clone() else {
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

            let UnitState::Ready(unit) = &entry.state else {
                continue; // still loading (or failed: retried after eviction)
            };

            if budget == 0 {
                continue;
            }
            budget -= 1;

            let unit = unit.clone();
            let task = AsyncComputeTaskPool::get().spawn(async move {
                decode_range(&unit.meta, &unit.textures, interval.offset, interval.count)
                    .map(pad_to_multiple_of_32)
                    .map_err(|e| e.to_string())
            });

            entry.refs += 1;
            entry.cooldown = entry.cooldown_reset;
            runtime.pending[i] = Some(PendingSwap {
                lod: desired,
                file: interval.file,
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
    mut runtimes: Query<(Entity, &GaussianLodSceneHandle, &mut LodRuntime)>,
) {
    for (scene_entity, handle, mut runtime) in &mut runtimes {
        let scene_id = handle.0.id();

        for i in 0..runtime.pending.len() {
            let Some(swap) = &mut runtime.pending[i] else {
                continue;
            };
            let Some(result) = block_on(poll_once(&mut swap.task)) else {
                continue;
            };

            let (lod, file) = (swap.lod, swap.file);
            runtime.pending[i] = None;
            if let Some(entry) = cache.units.get_mut(&(scene_id, file)) {
                entry.refs = entry.refs.saturating_sub(1);
            }

            match result {
                Ok(gaussians) => {
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
                    if let Some(old) = runtime.entities[i].replace(child) {
                        commands.entity(old).despawn();
                    }
                    runtime.active[i] = Some(lod);
                }
                Err(error) => warn!("leaf {i} interval decode failed: {error}"),
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
