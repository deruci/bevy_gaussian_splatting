//! Headless end-to-end test of the SOG LOD streaming runtime: writes a
//! synthetic streamed-SOG scene to disk, loads it through the asset server,
//! and drives the app until distance-based selection has spawned the expected
//! leaf clouds.

#![cfg(feature = "io_sog")]

use bevy::{app::ScheduleRunnerPlugin, asset::AssetPlugin, prelude::*};
use bevy_gaussian_splatting::{
    CompositeWrite, CompositeWriteQueue, GaussianCamera, GaussianLodScene,
    GaussianLodSceneHandle, LodRuntime, LodSettings, SogUnitCache,
    gaussian::formats::planar_3d::PlanarGaussian3d, io::lod::GaussianLodScenePlugin,
};

fn encode_webp(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    image::codecs::webp::WebPEncoder::new_lossless(std::io::Cursor::new(&mut bytes))
        .encode(rgba, width, height, image::ExtendedColorType::Rgba8)
        .unwrap();
    bytes
}

/// Write one SOG unit of `count` gaussians into `dir` (all at texel-grid
/// positions; content is irrelevant to the streaming logic under test).
fn write_unit(dir: &std::path::Path, count: usize) {
    std::fs::create_dir_all(dir).unwrap();
    let w = count as u32;

    let per_gaussian = |values: [u8; 4]| -> Vec<u8> {
        (0..count).flat_map(|_| values).collect::<Vec<u8>>()
    };

    let files = [
        ("means_l.webp", per_gaussian([128, 128, 128, 255])),
        ("means_u.webp", per_gaussian([128, 128, 128, 255])),
        ("quats.webp", per_gaussian([128, 128, 128, 252])),
        ("scales.webp", per_gaussian([0, 0, 0, 255])),
        ("sh0.webp", per_gaussian([0, 0, 0, 255])),
    ];
    for (name, rgba) in files {
        std::fs::write(dir.join(name), encode_webp(&rgba, w, 1)).unwrap();
    }

    let meta = format!(
        r#"{{
            "version": 2,
            "count": {count},
            "means": {{ "mins": [-1, -1, -1], "maxs": [1, 1, 1],
                        "files": ["means_l.webp", "means_u.webp"] }},
            "scales": {{ "codebook": [{scales}], "files": ["scales.webp"] }},
            "quats": {{ "files": ["quats.webp"] }},
            "sh0": {{ "codebook": [{scales}], "files": ["sh0.webp"] }}
        }}"#,
        scales = vec!["-2.0"; 256].join(","),
    );
    std::fs::write(dir.join("meta.json"), meta).unwrap();
}

/// Two leaves, two LOD levels; leaf 0 near the origin, leaf 1 far away.
/// LOD 0 lives in unit `0_0` (leaf 0: rows 0..4, leaf 1: rows 4..8);
/// LOD 1 in unit `1_0` (rows 0..2 and 2..4).
fn write_scene(root: &std::path::Path) {
    write_unit(&root.join("0_0"), 8);
    write_unit(&root.join("1_0"), 4);

    let meta = r#"{
        "version": 1,
        "asset": { "generator": "test" },
        "count": 12,
        "counts": [8, 4],
        "lodLevels": 2,
        "filenames": ["0_0/meta.json", "1_0/meta.json"],
        "tree": {
            "bound": { "min": [-2, -2, -2], "max": [102, 2, 2] },
            "children": [
                {
                    "bound": { "min": [-2, -2, -2], "max": [2, 2, 2] },
                    "lods": {
                        "0": { "file": 0, "offset": 0, "count": 4 },
                        "1": { "file": 1, "offset": 0, "count": 2 }
                    }
                },
                {
                    "bound": { "min": [98, -2, -2], "max": [102, 2, 2] },
                    "lods": {
                        "0": { "file": 0, "offset": 4, "count": 4 },
                        "1": { "file": 1, "offset": 2, "count": 2 }
                    }
                }
            ]
        }
    }"#;
    std::fs::write(root.join("lod-meta.json"), meta).unwrap();
}

fn build_app(assets_root: &std::path::Path, settings: LodSettings) -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins.set(ScheduleRunnerPlugin::default()),
        AssetPlugin {
            file_path: assets_root.to_string_lossy().into_owned(),
            ..Default::default()
        },
    ));
    app.init_asset::<PlanarGaussian3d>();
    app.add_plugins(GaussianLodScenePlugin);

    // camera inside leaf 0, ~99 units from leaf 1
    app.world_mut().spawn((
        GaussianCamera::default(),
        Projection::Perspective(PerspectiveProjection::default()),
        Transform::from_xyz(0.0, 0.0, 0.0),
        GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 0.0)),
    ));

    let handle = app
        .world()
        .resource::<AssetServer>()
        .load::<GaussianLodScene>("lod-meta.json");
    let scene = app
        .world_mut()
        .spawn((GaussianLodSceneHandle(handle), settings))
        .id();

    (app, scene)
}

/// Pump the app until `predicate` holds or `max_updates` elapse.
fn pump(app: &mut App, scene: Entity, max_updates: usize, predicate: impl Fn(&LodRuntime) -> bool) -> bool {
    for _ in 0..max_updates {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(5));
        if let Some(runtime) = app.world().entity(scene).get::<LodRuntime>()
            && predicate(runtime)
        {
            return true;
        }
    }
    false
}

fn scene_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("sog_lod_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_scene(&dir);
    dir
}

#[test]
fn streams_distance_based_lods() {
    let root = scene_dir("distance");
    let (mut app, scene) = build_app(
        &root,
        LodSettings {
            composite: false,
            ..Default::default()
        },
    );

    let done = pump(&mut app, scene, 1000, |runtime| {
        let active: Vec<_> = runtime.active_lods().collect();
        active.len() == 2
    });
    assert!(done, "leaves never became active");

    let runtime = app.world().entity(scene).get::<LodRuntime>().unwrap();
    let active: Vec<_> = runtime.active_lods().collect();
    // camera sits inside leaf 0 (finest) and ~99 units from leaf 1 (coarsest)
    assert!(active.contains(&(0, 0)), "near leaf should be LOD 0: {active:?}");
    assert!(active.contains(&(1, 1)), "far leaf should be LOD 1: {active:?}");

    // both leaves spawned cloud children under the scene entity
    let children = app.world().entity(scene).get::<Children>().unwrap();
    assert_eq!(children.len(), 2);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn forced_lod_and_eviction() {
    let root = scene_dir("forced");
    let (mut app, scene) = build_app(
        &root,
        LodSettings {
            forced_lod: Some(1),
            cooldown_frames: 3,
            composite: false,
            ..Default::default()
        },
    );

    let done = pump(&mut app, scene, 1000, |runtime| {
        runtime.active_lods().collect::<Vec<_>>() == vec![(0, 1), (1, 1)]
    });
    assert!(done, "forced LOD 1 never applied to both leaves");

    // with no pending work, refcounts are zero: the unit cache must drain
    // within cooldown_frames updates
    for _ in 0..20 {
        app.update();
    }
    assert!(
        app.world().resource::<SogUnitCache>().is_empty(),
        "unit cache should evict after cooldown"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn composite_mode_patches_one_cloud() {
    let root = scene_dir("composite");
    let (mut app, scene) = build_app(
        &root,
        LodSettings {
            composite: true,
            splat_budget: 64,
            ..Default::default()
        },
    );

    let done = pump(&mut app, scene, 1000, |runtime| {
        runtime.active_lods().count() == 2 && runtime.in_flight() == 0
    });
    assert!(done, "composite leaves never became active");

    let world = app.world();
    let runtime = world.entity(scene).get::<LodRuntime>().unwrap();
    let active: Vec<_> = runtime.active_lods().collect();
    assert!(active.contains(&(0, 0)), "near leaf should be LOD 0: {active:?}");
    assert!(active.contains(&(1, 1)), "far leaf should be LOD 1: {active:?}");

    // exactly one child: the composite cloud; displayed splats live in blocks
    let children = world.entity(scene).get::<Children>().unwrap();
    assert_eq!(children.len(), 1);
    // leaf 0 at LOD 0 (4 splats) + leaf 1 at LOD 1 (2 splats)
    assert_eq!(runtime.composite_used(), Some(6));

    // without a render app the write queue accumulates: both leaf uploads
    // must be Block writes targeting the composite asset
    let queue = world.resource::<CompositeWriteQueue>();
    let writes = queue.0.lock().unwrap();
    let composite_id = runtime.composite_handle().unwrap().id();
    let blocks = writes
        .iter()
        .filter(|w| matches!(w, CompositeWrite::Block { asset, .. } if *asset == composite_id))
        .count();
    assert_eq!(blocks, 2, "expected one Block write per leaf");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn composite_budget_never_starves_leaves() {
    let root = scene_dir("starve");
    // budget 4 can't give anyone LOD0 (4 splats) without starving the other
    // leaf's coarsest guarantee (2): both leaves must land on LOD1 — no holes
    let (mut app, scene) = build_app(
        &root,
        LodSettings {
            composite: true,
            splat_budget: 4,
            forced_lod: Some(0),
            ..Default::default()
        },
    );

    let done = pump(&mut app, scene, 1000, |runtime| {
        runtime.active_lods().count() == 2 && runtime.in_flight() == 0
    });
    assert!(done, "leaves starved under tight budget");

    let world = app.world();
    let runtime = world.entity(scene).get::<LodRuntime>().unwrap();
    let active: Vec<_> = runtime.active_lods().collect();
    assert_eq!(active, vec![(0, 1), (1, 1)], "both leaves at coarsest, no holes");
    assert!(runtime.composite_used().unwrap() <= 4);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn composite_budget_degrades_to_coarser_lod() {
    let root = scene_dir("budget");
    // both leaves wish LOD0 (4 splats each = 8) but the budget of 6 only fits
    // one at LOD0; the other must degrade to LOD1 (2 splats)
    let (mut app, scene) = build_app(
        &root,
        LodSettings {
            composite: true,
            splat_budget: 6,
            forced_lod: Some(0),
            ..Default::default()
        },
    );

    let done = pump(&mut app, scene, 1000, |runtime| {
        runtime.active_lods().count() == 2 && runtime.in_flight() == 0
    });
    assert!(done, "budget-degraded leaves never became active");

    let world = app.world();
    let runtime = world.entity(scene).get::<LodRuntime>().unwrap();
    let used = runtime.composite_used().unwrap();
    assert!(used <= 6, "budget exceeded: {used}");
    // one leaf gets its wish (LOD0, 4 splats), the other degrades to LOD1
    let active: Vec<_> = runtime.active_lods().collect();
    let lods: Vec<usize> = active.iter().map(|(_, lod)| *lod).collect();
    assert!(
        lods.contains(&0) && lods.contains(&1),
        "expected one degraded leaf: {active:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}
