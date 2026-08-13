//! End-to-end validation of the LOD streaming runtime against a real
//! streamed-SOG scene (splat-transform output). Ignored by default:
//!
//! ```sh
//! SOG_LOD_SCENE_DIR=/path/to/dir_containing_lod_meta \
//!   cargo test --release --test sog_lod_real_scene -- --ignored --nocapture
//! ```

#![cfg(feature = "io_sog")]

use bevy::{app::ScheduleRunnerPlugin, asset::AssetPlugin, prelude::*};
use bevy_gaussian_splatting::{
    GaussianCamera, GaussianLodScene, GaussianLodSceneHandle, LodRuntime, LodSettings,
    gaussian::formats::planar_3d::PlanarGaussian3d, io::lod::GaussianLodScenePlugin,
};

fn pump(
    app: &mut App,
    scene: Entity,
    max_updates: usize,
    predicate: impl Fn(&LodRuntime, &GaussianLodScene) -> bool,
) -> bool {
    for iteration in 0..max_updates {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let world = app.world();
        let Some(runtime) = world.entity(scene).get::<LodRuntime>() else {
            continue;
        };
        if iteration % 500 == 499 {
            println!(
                "pump {iteration}: active {} in_flight {} used {:?}",
                runtime.active_lods().count(),
                runtime.in_flight(),
                runtime.composite_used(),
            );
        }
        let handle = world.entity(scene).get::<GaussianLodSceneHandle>().unwrap();
        let Some(scene_asset) = world.resource::<Assets<GaussianLodScene>>().get(&handle.0)
        else {
            continue;
        };
        if predicate(runtime, scene_asset) {
            return true;
        }
    }
    false
}

fn lod_histogram(runtime: &LodRuntime, levels: usize) -> Vec<usize> {
    let mut histogram = vec![0usize; levels];
    for (_, lod) in runtime.active_lods() {
        histogram[lod] += 1;
    }
    histogram
}

#[test]
#[ignore = "requires SOG_LOD_SCENE_DIR pointing at a streamed SOG directory"]
fn streams_real_scene() {
    let root = std::env::var("SOG_LOD_SCENE_DIR")
        .expect("set SOG_LOD_SCENE_DIR to a directory containing lod-meta.json");

    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins.set(ScheduleRunnerPlugin::default()),
        AssetPlugin {
            file_path: root,
            ..Default::default()
        },
    ));
    app.init_asset::<PlanarGaussian3d>();
    app.add_plugins(GaussianLodScenePlugin);

    app.world_mut().spawn((
        GaussianCamera::default(),
        Projection::Perspective(PerspectiveProjection::default()),
        Transform::default(),
        GlobalTransform::default(),
    ));

    // composite unless SOG_LOD_ENTITY_MODE is set; budget covers full detail
    // (5M) plus transient blocks during swaps
    let settings = if std::env::var("SOG_LOD_ENTITY_MODE").is_ok() {
        LodSettings {
            composite: false,
            ..Default::default()
        }
    } else {
        LodSettings {
            composite: true,
            splat_budget: std::env::var("SOG_LOD_BUDGET")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(6_500_000),
            ..Default::default()
        }
    };

    let handle = app
        .world()
        .resource::<AssetServer>()
        .load::<GaussianLodScene>("lod-meta.json");
    let scene = app
        .world_mut()
        .spawn((GaussianLodSceneHandle(handle), settings))
        .id();

    let start = std::time::Instant::now();

    // settle: every leaf active at its desired level, nothing in flight
    let settled = pump(&mut app, scene, 6000, |runtime, asset| {
        runtime.in_flight() == 0
            && runtime.active_lods().count() == asset.leaves.len()
    });
    assert!(settled, "initial streaming never settled");

    let world = app.world();
    let runtime = world.entity(scene).get::<LodRuntime>().unwrap();
    let handle = world.entity(scene).get::<GaussianLodSceneHandle>().unwrap();
    let asset = world
        .resource::<Assets<GaussianLodScene>>()
        .get(&handle.0)
        .unwrap()
        .clone();

    let histogram = lod_histogram(runtime, asset.lod_levels);
    let displayed: usize = runtime
        .active_lods()
        .filter_map(|(i, lod)| asset.leaves[i].lods[lod].as_ref())
        .map(|interval| interval.count)
        .sum();
    let clouds = world.resource::<Assets<PlanarGaussian3d>>().len();

    println!(
        "settled in {:.2?}: {} leaves, {} lod levels, histogram {histogram:?}",
        start.elapsed(),
        asset.leaves.len(),
        asset.lod_levels,
    );
    println!(
        "displayed splats: {displayed} (full-detail scene: {}), cloud assets: {clouds}",
        asset.counts.first().copied().unwrap_or(0),
    );

    if let Some(used) = runtime.composite_used() {
        println!("composite blocks used: {used}");
        assert_eq!(used, displayed, "allocator must track displayed splats");
        assert_eq!(clouds, 1, "composite mode renders through one cloud");
    } else {
        assert_eq!(clouds, asset.leaves.len());
    }
    if asset.lod_levels > 1 {
        assert!(
            histogram.iter().filter(|&&n| n > 0).count() >= 2,
            "camera inside the scene should see mixed LOD levels: {histogram:?}"
        );
    }

    // teleport far away: all leaves must converge to the coarsest available
    let far = Vec3::new(10_000.0, 0.0, 0.0);
    let mut cameras = app
        .world_mut()
        .query_filtered::<&mut GlobalTransform, With<GaussianCamera>>();
    *cameras.single_mut(app.world_mut()).unwrap() =
        GlobalTransform::from(Transform::from_translation(far));

    let coarse = pump(&mut app, scene, 6000, |runtime, asset| {
        runtime.in_flight() == 0
            && runtime.active_lods().count() == asset.leaves.len()
            && runtime
                .active_lods()
                .all(|(i, lod)| Some(lod) == asset.leaves[i].nearest_available(usize::MAX))
    });
    assert!(coarse, "far camera never converged to coarsest LODs");

    let world = app.world();
    let runtime = world.entity(scene).get::<LodRuntime>().unwrap();
    println!(
        "after teleport: histogram {:?}",
        lod_histogram(runtime, asset.lod_levels)
    );
}
