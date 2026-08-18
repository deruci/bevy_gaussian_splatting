// TODO: move to editor crate
use std::path::PathBuf;

use bevy::{
    app::AppExit,
    camera::primitives::Aabb,
    color::palettes::css::GOLD,
    core_pipeline::{prepass::MotionVectorPrepass, tonemapping::Tonemapping},
    diagnostic::{DiagnosticsStore, FrameCount, FrameTimeDiagnosticsPlugin},
    gizmos::config::GizmoConfigStore,
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};

#[cfg(all(feature = "file_asset", not(target_arch = "wasm32")))]
use bevy::asset::{
    AssetApp,
    io::{AssetSourceBuilder, file::FileAssetReader},
};

#[cfg(feature = "web_asset")]
use bevy::asset::io::web::WebAssetPlugin;
use bevy_args::{BevyArgsPlugin, parse_args};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[cfg(feature = "web_asset")]
use base64::{Engine as _, engine::general_purpose::URL_SAFE};
use bevy_gaussian_splatting::{
    CloudSettings, GaussianCamera, GaussianMode, GaussianPrimitiveMetadata, GaussianScene,
    GaussianSceneHandle, GaussianSplattingPlugin, PlanarGaussian3d, PlanarGaussian3dHandle,
    PlanarGaussian4d, PlanarGaussian4dHandle,
    gaussian::interface::TestCloud,
    io::scene::GaussianSceneLoaded,
    random_gaussians_3d, random_gaussians_3d_seeded, random_gaussians_4d,
    random_gaussians_4d_seeded,
    utils::{GaussianSplattingViewer, log, setup_hooks},
};

#[cfg(not(target_arch = "wasm32"))]
use bevy_gaussian_splatting::{SceneExportCamera, SceneExportCloud, write_khr_gaussian_scene_glb};

#[cfg(feature = "morph_interpolate")]
use bevy_gaussian_splatting::{Gaussian3d, morph::interpolate::GaussianInterpolate};

#[cfg(feature = "material_noise")]
use bevy_gaussian_splatting::material::noise::NoiseMaterial;

#[cfg(feature = "morph_particles")]
use bevy_gaussian_splatting::morph::particle::{
    ParticleBehaviors, ParticleBehaviorsHandle, random_particle_behaviors,
};

#[cfg(feature = "query_select")]
use bevy_gaussian_splatting::query::select::{InvertSelectionEvent, SaveSelectionEvent};

#[cfg(feature = "query_sparse")]
use bevy_gaussian_splatting::query::sparse::SparseSelect;

#[derive(Component, Debug, Default)]
struct ViewerMainCamera;

#[derive(Component, Debug, Default)]
struct SceneCameraApplied;

#[derive(Component, Debug, Default)]
struct SceneRenderModeApplied;

#[cfg(not(target_arch = "wasm32"))]
type ExportCloudQuery = (
    &'static PlanarGaussian3dHandle,
    &'static GlobalTransform,
    Option<&'static Name>,
    Option<&'static CloudSettings>,
    Option<&'static GaussianPrimitiveMetadata>,
);

#[cfg(not(target_arch = "wasm32"))]
type ExportCameraQuery = (&'static GlobalTransform, Option<&'static Name>);
type SceneCameraApplyQuery = (Entity, &'static mut Transform, &'static mut PanOrbitCamera);
type SceneRenderModeQuery = (Entity, &'static Children);
type SceneRenderModeFilter = (With<GaussianSceneLoaded>, Without<SceneRenderModeApplied>);

fn parse_input_file(input_file: &str) -> String {
    #[cfg(feature = "web_asset")]
    let input_uri = match URL_SAFE.decode(input_file.as_bytes()) {
        Ok(data) => match String::from_utf8(data) {
            Ok(decoded) => decoded,
            Err(_) => input_file.to_string(),
        },
        Err(err) => {
            if let Some(decoded) = decode_percent_encoded(input_file) {
                return decoded;
            }

            // Leave as-is for regular relative paths and already-decoded URLs.
            debug!("failed to decode base64 input: {:?}", err);
            input_file.to_string()
        }
    };

    #[cfg(not(feature = "web_asset"))]
    let input_uri = input_file.to_string();

    input_uri
}

#[cfg(feature = "web_asset")]
fn decode_percent_encoded(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    let mut changed = false;

    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return None;
            }

            let high = decode_hex(bytes[index + 1])?;
            let low = decode_hex(bytes[index + 2])?;
            decoded.push((high << 4) | low);
            index += 3;
            changed = true;
            continue;
        }

        decoded.push(bytes[index]);
        index += 1;
    }

    if !changed {
        return None;
    }

    String::from_utf8(decoded).ok()
}

#[cfg(feature = "web_asset")]
fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn setup_gaussian_cloud(
    mut commands: Commands,
    args: Res<GaussianSplattingViewer>,
    asset_server: Res<AssetServer>,
    mut gaussian_3d_assets: ResMut<Assets<PlanarGaussian3d>>,
    mut gaussian_4d_assets: ResMut<Assets<PlanarGaussian4d>>,
) {
    debug!("spawning camera...");
    let cloud_transform = args.cloud_transform();

    let mut pan_orbit = PanOrbitCamera {
        allow_upside_down: true,
        orbit_smoothness: 0.1,
        pan_smoothness: 0.1,
        zoom_smoothness: 0.1,
        ..default()
    };
    let mut camera_transform = Transform::from_translation(Vec3::new(0.0, 1.5, 5.0));

    if let Some((position, focus)) = args
        .camera_pose
        .as_deref()
        .and_then(bevy_gaussian_splatting::utils::parse_camera_pose)
    {
        // invert the orbit parameterization (mirrors bevy_panorbit_camera's
        // internal calculate_from_translation_and_focus, which is private)
        let delta = position - focus;
        let radius = delta.length().max(0.05);
        let yaw = delta.x.atan2(delta.z);
        let pitch = (delta.y / radius).asin();

        pan_orbit.focus = focus;
        pan_orbit.target_focus = focus;
        pan_orbit.yaw = Some(yaw);
        pan_orbit.target_yaw = yaw;
        pan_orbit.pitch = Some(pitch);
        pan_orbit.target_pitch = pitch;
        pan_orbit.radius = Some(radius);
        pan_orbit.target_radius = radius;

        camera_transform = Transform::from_translation(position).looking_at(focus, Vec3::Y);
    }

    commands
        .spawn(Camera3d::default())
        .insert(camera_transform)
        .insert(Tonemapping::None)
        .insert(MotionVectorPrepass)
        .insert(pan_orbit)
        .insert(ViewerMainCamera)
        .insert(GaussianCamera::default());

    if let Some(input_scene) = &args.input_scene {
        let input_uri = parse_input_file(input_scene.as_str());
        log(&format!("loading {input_uri}"));
        let scene: Handle<GaussianScene> = asset_server.load(&input_uri);
        commands.spawn((
            GaussianSceneHandle(scene),
            Name::new("gaussian_scene"),
            cloud_transform,
            CloudRoot {
                initial_rotation: cloud_transform.rotation,
            },
        ));
        return;
    }

    #[cfg(feature = "io_sog")]
    if let Some(input_lod) = &args.input_lod {
        use bevy_gaussian_splatting::{GaussianLodScene, GaussianLodSceneHandle, LodSettings};

        let input_uri = parse_input_file(input_lod.as_str());
        log(&format!("loading streamed SOG scene {input_uri}"));
        let scene: Handle<GaussianLodScene> = asset_server.load(&input_uri);

        let mut lod_settings = LodSettings::default();
        if args.splat_budget > 0 {
            lod_settings.splat_budget = args.splat_budget;
        }
        lod_settings.composite = args.composite;

        commands.spawn((
            GaussianLodSceneHandle(scene),
            lod_settings,
            Name::new("gaussian_lod_scene"),
            cloud_transform,
            CloudRoot {
                initial_rotation: cloud_transform.rotation,
            },
        ));
        return;
    }

    match args.gaussian_mode {
        GaussianMode::Gaussian2d | GaussianMode::Gaussian3d => {
            let cloud: Handle<PlanarGaussian3d>;
            if args.gaussian_count > 0 {
                log(&format!("generating {} gaussians", args.gaussian_count));
                cloud = if let Some(seed) = args.gaussian_seed {
                    gaussian_3d_assets.add(random_gaussians_3d_seeded(args.gaussian_count, seed))
                } else {
                    gaussian_3d_assets.add(random_gaussians_3d(args.gaussian_count))
                };
            } else if let Some(input_cloud) = &args.input_cloud {
                let input_uri = parse_input_file(input_cloud.as_str());
                log(&format!("loading {input_uri}"));
                cloud = asset_server.load(&input_uri);
            } else {
                cloud = gaussian_3d_assets.add(PlanarGaussian3d::test_model());
            }

            #[cfg(feature = "morph_interpolate")]
            {
                if let Some(input_cloud_target) = &args.input_cloud_target {
                    let input_uri = parse_input_file(input_cloud_target.as_str());
                    log(&format!("loading {input_uri}"));
                    let binary_cloud: Handle<PlanarGaussian3d> = asset_server.load(&input_uri);

                    commands.spawn((
                        CloudSettings {
                            gaussian_mode: args.gaussian_mode,
                            playback_mode: args.playback_mode,
                            rasterize_mode: args.rasterization_mode,
                            radix_sort_depth_bits: args.radix_sort_depth_bits,
                            ..default()
                        },
                        GaussianInterpolate::<Gaussian3d> {
                            lhs: PlanarGaussian3dHandle(cloud),
                            rhs: PlanarGaussian3dHandle(binary_cloud),
                        },
                        Name::new("gaussian_cloud_3d_binary"),
                        ShowAxes,
                        cloud_transform,
                        CloudRoot {
                            initial_rotation: cloud_transform.rotation,
                        },
                    ));
                } else {
                    commands.spawn((
                        CloudSettings {
                            gaussian_mode: args.gaussian_mode,
                            playback_mode: args.playback_mode,
                            rasterize_mode: args.rasterization_mode,
                            radix_sort_depth_bits: args.radix_sort_depth_bits,
                            ..default()
                        },
                        PlanarGaussian3dHandle(cloud.clone()),
                        Name::new("gaussian_cloud_3d"),
                        ShowAxes,
                        cloud_transform,
                        CloudRoot {
                            initial_rotation: cloud_transform.rotation,
                        },
                    ));
                }
            }

            #[cfg(not(feature = "morph_interpolate"))]
            {
                commands.spawn((
                    CloudSettings {
                        gaussian_mode: args.gaussian_mode,
                        playback_mode: args.playback_mode,
                        rasterize_mode: args.rasterization_mode,
                        radix_sort_depth_bits: args.radix_sort_depth_bits,
                        ..default()
                    },
                    PlanarGaussian3dHandle(cloud.clone()),
                    Name::new("gaussian_cloud_3d"),
                    ShowAxes,
                    cloud_transform,
                    CloudRoot {
                        initial_rotation: cloud_transform.rotation,
                    },
                ));
            }
        }
        GaussianMode::Gaussian4d => {
            let cloud: Handle<PlanarGaussian4d>;
            if args.gaussian_count > 0 {
                log(&format!("generating {} gaussians", args.gaussian_count));
                cloud = if let Some(seed) = args.gaussian_seed {
                    gaussian_4d_assets.add(random_gaussians_4d_seeded(args.gaussian_count, seed))
                } else {
                    gaussian_4d_assets.add(random_gaussians_4d(args.gaussian_count))
                };
            } else if let Some(input_cloud) = &args.input_cloud {
                let input_uri = parse_input_file(input_cloud.as_str());
                log(&format!("loading {input_uri}"));
                cloud = asset_server.load(&input_uri);
            } else {
                cloud = gaussian_4d_assets.add(PlanarGaussian4d::test_model());
            }

            commands.spawn((
                PlanarGaussian4dHandle(cloud),
                CloudSettings {
                    gaussian_mode: args.gaussian_mode,
                    playback_mode: args.playback_mode,
                    rasterize_mode: args.rasterization_mode,
                    radix_sort_depth_bits: args.radix_sort_depth_bits,
                    ..default()
                },
                Name::new("gaussian_cloud_4d"),
                ShowAxes,
                cloud_transform,
                CloudRoot {
                    initial_rotation: cloud_transform.rotation,
                },
            ));
        }
    }
}

fn apply_scene_camera_spawn(
    mut commands: Commands,
    scene_handles: Query<(Entity, &GaussianSceneHandle), Without<SceneCameraApplied>>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianScene>>,
    mut cameras: Query<SceneCameraApplyQuery, (With<GaussianCamera>, With<ViewerMainCamera>)>,
) {
    for (entity, scene_handle) in scene_handles.iter() {
        if let Some(load_state) = asset_server.get_load_state(&scene_handle.0)
            && !load_state.is_loaded()
        {
            continue;
        }

        let Some(scene) = scenes.get(&scene_handle.0) else {
            continue;
        };

        if let Some(scene_camera) = scene.cameras.first()
            && let Ok((camera_entity, mut camera_transform, mut pan_orbit_camera)) =
                cameras.single_mut()
        {
            let orbit_radius = pan_orbit_camera
                .target_radius
                .max(pan_orbit_camera.zoom_lower_limit);
            let scene_translation = scene_camera.transform.translation;
            let scene_forward = scene_camera.transform.forward().as_vec3();
            let world_up = pan_orbit_camera.axis[1];
            let mut corrected_rotation = scene_camera.transform.rotation;

            // Imported camera can legitimately be upside-down (roll ~= PI) which makes orbit input
            // feel inverted. Flip it upright while keeping the same look direction.
            if scene_camera.transform.up().dot(world_up) < 0.0 {
                corrected_rotation =
                    Quat::from_axis_angle(scene_forward, std::f32::consts::PI) * corrected_rotation;
            }

            let corrected_transform = Transform {
                translation: scene_translation,
                rotation: corrected_rotation,
                scale: Vec3::ONE,
            };
            *camera_transform = corrected_transform;

            let focus = scene_translation + camera_transform.forward() * orbit_radius;

            let (yaw, pitch, radius) = orbit_from_translation_and_focus(
                camera_transform.translation,
                focus,
                pan_orbit_camera.axis,
            );

            pan_orbit_camera.focus = focus;
            pan_orbit_camera.target_focus = focus;
            pan_orbit_camera.yaw = Some(yaw);
            pan_orbit_camera.pitch = Some(pitch);
            pan_orbit_camera.radius = Some(radius);
            pan_orbit_camera.target_yaw = yaw;
            pan_orbit_camera.target_pitch = pitch;
            pan_orbit_camera.target_radius = radius;
            pan_orbit_camera.allow_upside_down = false;
            pan_orbit_camera.initialized = true;
            pan_orbit_camera.force_update = true;
            let _ = camera_entity;
        }

        commands.entity(entity).insert(SceneCameraApplied);
    }
}

fn apply_scene_render_mode_override(
    mut commands: Commands,
    args: Res<GaussianSplattingViewer>,
    scenes: Query<SceneRenderModeQuery, SceneRenderModeFilter>,
    mut cloud_settings: Query<&mut CloudSettings>,
) {
    if args.input_scene.is_none() {
        return;
    }

    for (entity, children) in scenes.iter() {
        for child in children.iter() {
            let child: Entity = child;
            if let Ok(mut settings) = cloud_settings.get_mut(child) {
                settings.rasterize_mode = args.rasterization_mode;
                settings.radix_sort_depth_bits = args.radix_sort_depth_bits;
            }
        }

        commands.entity(entity).insert(SceneRenderModeApplied);
    }
}

fn orbit_from_translation_and_focus(
    translation: Vec3,
    focus: Vec3,
    axis: [Vec3; 3],
) -> (f32, f32, f32) {
    let axis = Mat3::from_cols(axis[0], axis[1], axis[2]);
    let offset = translation - focus;

    // Radius of exactly zero creates unstable orbit behavior.
    let mut radius = offset.length();
    if radius <= f32::EPSILON {
        radius = 0.05;
    }

    let offset = axis * offset;
    let yaw = offset.x.atan2(offset.z);
    let pitch = (offset.y / radius).asin();
    (yaw, pitch, radius)
}

#[cfg(feature = "morph_particles")]
fn setup_particle_behavior(
    mut commands: Commands,
    gaussian_splatting_viewer: Res<GaussianSplattingViewer>,
    mut particle_behavior_assets: ResMut<Assets<ParticleBehaviors>>,
    gaussian_cloud: Query<(Entity, &PlanarGaussian3dHandle), Without<ParticleBehaviorsHandle>>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    let mut particle_behaviors = None;
    if gaussian_splatting_viewer.particle_count > 0 {
        log(&format!(
            "generating {} particle behaviors",
            gaussian_splatting_viewer.particle_count
        ));
        particle_behaviors = particle_behavior_assets
            .add(random_particle_behaviors(
                gaussian_splatting_viewer.particle_count,
            ))
            .into();
    }

    if let Some(particle_behaviors) = particle_behaviors
        && let Ok((entity, _)) = gaussian_cloud.single()
    {
        commands
            .entity(entity)
            .insert(ParticleBehaviorsHandle(particle_behaviors));
    }
}

#[cfg(feature = "material_noise")]
fn setup_noise_material(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gaussian_clouds: Query<(Entity, &PlanarGaussian3dHandle), Without<NoiseMaterial>>,
) {
    if gaussian_clouds.is_empty() {
        return;
    }

    for (entity, cloud_handle) in gaussian_clouds.iter() {
        if let Some(load_state) = asset_server.get_load_state(cloud_handle.0.id())
            && load_state.is_loading()
        {
            continue;
        }

        commands.entity(entity).insert(NoiseMaterial::default());
    }
}

#[cfg(feature = "query_sparse")]
fn setup_sparse_select(
    mut commands: Commands,
    gaussian_cloud: Query<(Entity, &PlanarGaussian3dHandle), Without<SparseSelect>>,
) {
    if gaussian_cloud.is_empty() {
        return;
    }

    if let Ok((entity, _)) = gaussian_cloud.single() {
        commands.entity(entity).insert(SparseSelect {
            completed: true,
            ..default()
        });
    }
}

fn viewer_app() {
    let config = parse_args::<GaussianSplattingViewer>();
    log(&format!("{config:?}"));

    #[cfg(not(feature = "morph_interpolate"))]
    if config.input_cloud_target.is_some() {
        panic!("`--input-cloud-target` requires the `morph_interpolate` feature");
    }

    let mut app = App::new();
    app.register_type::<GizmoConfigStore>();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: config.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: bevy::window::WindowResolution::new(config.width as u32, config.height as u32),
        title: config.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(all(feature = "file_asset", not(target_arch = "wasm32")))]
    app.register_asset_source(
        "file",
        AssetSourceBuilder::new(|| Box::new(FileAssetReader::new("")))
            .with_processed_reader(|| Box::new(FileAssetReader::new(""))),
    );

    // setup for gaussian viewer app
    app.insert_resource(ClearColor(Color::srgb_u8(0, 0, 0)));
    let default_plugins = DefaultPlugins
        .set(AssetPlugin {
            meta_check: bevy::asset::AssetMetaCheck::Never,
            unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
            ..default()
        })
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    #[cfg(feature = "web_asset")]
    let default_plugins = default_plugins.set(WebAssetPlugin {
        silence_startup_warning: true,
    });

    app.add_plugins(default_plugins);
    app.add_plugins(BevyArgsPlugin::<GaussianSplattingViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    if config.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    if config.press_s_screenshot {
        app.add_systems(Update, press_s_screenshot);
    }

    if config.show_axes {
        app.add_systems(Update, draw_axes);
    }

    if config.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    // setup for gaussian splatting
    app.add_plugins(GaussianSplattingPlugin);
    app.add_systems(Startup, setup_gaussian_cloud);
    app.add_systems(Update, apply_scene_camera_spawn);
    app.add_systems(Update, apply_scene_render_mode_override);
    app.add_systems(Update, press_g_save_gltf_scene);

    app.add_message::<ToggleFlyRequest>();
    #[cfg(target_arch = "wasm32")]
    app.add_systems(Update, update_loading_overlay);
    app.add_systems(Startup, (control_panel_setup, mode_hint_setup));
    app.add_systems(
        Update,
        (
            press_p_save_pose,
            panel_click_system,
            panel_move_system,
            press_f_toggle_fly,
            apply_fly_toggle,
            fly_camera_update,
            mode_hint_update,
        )
            .chain(),
    );

    #[cfg(feature = "material_noise")]
    app.add_systems(Update, setup_noise_material);

    #[cfg(feature = "morph_particles")]
    app.add_systems(Update, setup_particle_behavior);

    #[cfg(feature = "query_select")]
    {
        app.add_systems(Update, press_i_invert_selection);
        app.add_systems(Update, press_o_save_selection);
    }

    #[cfg(feature = "query_sparse")]
    app.add_systems(Update, setup_sparse_select);

    app.run();
}

/// Root entity of the displayed cloud/scene; the control panel rotates this.
#[derive(Component)]
struct CloudRoot {
    initial_rotation: Quat,
}

#[derive(Component, Clone, Copy)]
enum PanelAction {
    ToggleFly,
    Rotate { axis: usize, degrees: f32 },
    ResetRotation,
    SaveView,
}

/// Camera-relative move direction (x: right, y: world up, z: forward),
/// applied every frame while the button is held.
#[derive(Component, Clone, Copy)]
struct MoveAction(Vec3);

#[derive(bevy::ecs::message::Message)]
struct ToggleFlyRequest;

#[derive(Component)]
struct ModeHintText;

const ORBIT_HINT: &str = "orbit mode  ·  F: fly  ·  drag: rotate  ·  wheel: zoom  ·  P: save view";
const FLY_HINT: &str =
    "fly mode  ·  F: orbit  ·  WASD + Q/E: move  ·  drag: look  ·  wheel: speed  ·  Shift: boost";

fn mode_hint_setup(mut commands: Commands) {
    commands.spawn((
        ModeHintText,
        Text(ORBIT_HINT.to_string()),
        TextFont {
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.75)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(14.0),
            right: Val::Px(12.0),
            ..default()
        },
    ));
}

fn mode_hint_update(
    flying: Query<(), With<FlyMode>>,
    mut hints: Query<&mut Text, With<ModeHintText>>,
) {
    let hint = if flying.is_empty() { ORBIT_HINT } else { FLY_HINT };
    for mut text in &mut hints {
        if text.0 != hint {
            text.0 = hint.to_string();
        }
    }
}

/// WASD/mouse-drag fly navigation; toggled with F (disables the orbit
/// controller while active and re-seeds it on exit).
#[derive(Component)]
struct FlyMode {
    yaw: f32,
    pitch: f32,
    speed: f32,
}

fn press_f_toggle_fly(
    keys: Res<ButtonInput<KeyCode>>,
    mut requests: MessageWriter<ToggleFlyRequest>,
) {
    if keys.just_pressed(KeyCode::KeyF) {
        requests.write(ToggleFlyRequest);
    }
}

fn apply_fly_toggle(
    mut commands: Commands,
    mut requests: MessageReader<ToggleFlyRequest>,
    mut cameras: Query<
        (Entity, &Transform, &mut PanOrbitCamera, Option<&FlyMode>),
        With<ViewerMainCamera>,
    >,
) {
    if requests.read().count() == 0 {
        return;
    }
    let Ok((entity, transform, mut pan_orbit, fly)) = cameras.single_mut() else {
        return;
    };

    if let Some(fly) = fly {
        // back to orbit: focus a point ahead of the camera and re-seed the
        // orbit parameterization from the flown pose
        let radius = pan_orbit.radius.unwrap_or(4.0).clamp(0.5, 20.0);
        let focus = transform.translation + transform.forward() * radius;
        pan_orbit.focus = focus;
        pan_orbit.target_focus = focus;
        pan_orbit.yaw = Some(fly.yaw);
        pan_orbit.target_yaw = fly.yaw;
        pan_orbit.pitch = Some(fly.pitch);
        pan_orbit.target_pitch = fly.pitch;
        pan_orbit.radius = Some(radius);
        pan_orbit.target_radius = radius;
        pan_orbit.enabled = true;

        commands.entity(entity).remove::<FlyMode>();
        log("orbit mode");
    } else {
        // seed fly yaw/pitch from the current view direction; note the fly
        // convention looks along -Z, mirroring the orbit inversion
        let forward = transform.forward();
        let yaw = (-forward.x).atan2(-forward.z);
        let pitch = forward.y.asin();

        pan_orbit.enabled = false;
        commands.entity(entity).insert(FlyMode {
            yaw,
            pitch,
            speed: 3.0,
        });
        log("fly mode: WASD move, Q/E down/up, drag to look, scroll for speed, Shift to boost");
    }
}

fn fly_camera_update(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
    mut mouse_wheel: MessageReader<MouseWheel>,
    mut cameras: Query<(&mut Transform, &mut FlyMode), With<ViewerMainCamera>>,
) {
    let Ok((mut transform, mut fly)) = cameras.single_mut() else {
        mouse_motion.clear();
        mouse_wheel.clear();
        return;
    };

    // look: drag with either mouse button
    let mut look = Vec2::ZERO;
    for motion in mouse_motion.read() {
        look += motion.delta;
    }
    if buttons.pressed(MouseButton::Left) || buttons.pressed(MouseButton::Right) {
        fly.yaw -= look.x * 0.003;
        fly.pitch = (fly.pitch - look.y * 0.003).clamp(-1.54, 1.54);
    }

    // speed: scroll wheel, multiplicative
    for wheel in mouse_wheel.read() {
        let steps = match wheel.unit {
            MouseScrollUnit::Line => wheel.y,
            MouseScrollUnit::Pixel => wheel.y / 60.0,
        };
        fly.speed = (fly.speed * 1.15f32.powf(steps)).clamp(0.1, 100.0);
    }

    transform.rotation = Quat::from_euler(EulerRot::YXZ, fly.yaw, fly.pitch, 0.0);

    let mut movement = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        movement += *transform.forward();
    }
    if keys.pressed(KeyCode::KeyS) {
        movement += *transform.back();
    }
    if keys.pressed(KeyCode::KeyA) {
        movement += *transform.left();
    }
    if keys.pressed(KeyCode::KeyD) {
        movement += *transform.right();
    }
    if keys.pressed(KeyCode::KeyE) {
        movement += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyQ) {
        movement -= Vec3::Y;
    }

    let boost = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        4.0
    } else {
        1.0
    };
    if movement != Vec3::ZERO {
        let translation =
            movement.normalize() * fly.speed * boost * time.delta_secs();
        transform.translation += translation;
    }
}

/// Serialize the current orbit pose and persist it: on the web the URL query
/// gains a `camera_pose` parameter (the address bar becomes a shareable
/// default-view link, SuperSplat-style); natively the CLI flag is logged.
/// Serialize the current orbit pose (and scene rotation) and persist them: on
/// the web the URL query gains `camera_pose` + `cloud_rotation` parameters
/// (the address bar becomes a shareable default-view link); natively the CLI
/// flags are logged.
fn save_camera_pose(
    cameras: &Query<
        (&Transform, &PanOrbitCamera, Option<&FlyMode>),
        (With<ViewerMainCamera>, Without<CloudRoot>),
    >,
    cloud_rotation: Option<Quat>,
) {
    let Ok((transform, pan_orbit, fly)) = cameras.single() else {
        return;
    };

    let position = transform.translation;
    // in fly mode the orbit focus is stale: focus a point ahead instead
    let focus = if fly.is_some() {
        position + transform.forward() * 4.0
    } else {
        pan_orbit.focus
    };
    let pose = format!(
        "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
        position.x, position.y, position.z, focus.x, focus.y, focus.z,
    );

    let rotation = cloud_rotation.map(|quat| {
        let (x, y, z) = quat.to_euler(EulerRot::XYZ);
        format!(
            "{:.1},{:.1},{:.1}",
            x.to_degrees(),
            y.to_degrees(),
            z.to_degrees()
        )
    });

    match &rotation {
        Some(rotation) => log(&format!(
            "view saved: --camera-pose {pose} --cloud-rotation {rotation}"
        )),
        None => log(&format!("view saved: --camera-pose {pose}")),
    }

    #[cfg(target_arch = "wasm32")]
    {
        let Some(window) = web_sys::window() else {
            return;
        };
        let location = window.location();
        let (Ok(pathname), Ok(search)) = (location.pathname(), location.search()) else {
            return;
        };

        // rebuild the query string with camera_pose / cloud_rotation replaced
        let mut params: Vec<String> = search
            .trim_start_matches('?')
            .split('&')
            .filter(|param| {
                !param.is_empty()
                    && !param.starts_with("camera_pose=")
                    && !(rotation.is_some() && param.starts_with("cloud_rotation="))
            })
            .map(str::to_owned)
            .collect();
        params.push(format!("camera_pose={pose}"));
        if let Some(rotation) = rotation {
            params.push(format!("cloud_rotation={rotation}"));
        }
        let url = format!("{pathname}?{}", params.join("&"));

        if let Ok(history) = window.history() {
            let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&url));
        }
    }
}

#[allow(clippy::type_complexity)]
fn press_p_save_pose(
    keys: Res<ButtonInput<KeyCode>>,
    cameras: Query<
        (&Transform, &PanOrbitCamera, Option<&FlyMode>),
        (With<ViewerMainCamera>, Without<CloudRoot>),
    >,
    clouds: Query<&Transform, With<CloudRoot>>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        let rotation = clouds.iter().next().map(|transform| transform.rotation);
        save_camera_pose(&cameras, rotation);
    }
}

/// Drive the index.html loading overlay through scene streaming: report leaf
/// coverage while SOG units download/decode, and hide the overlay on the
/// first fully-covered frame (or immediately when no streamed scene is used).
#[cfg(target_arch = "wasm32")]
#[allow(clippy::type_complexity)]
fn update_loading_overlay(
    mut done: Local<bool>,
    frames: Res<FrameCount>,
    #[cfg(feature = "io_sog")] scenes: Res<
        Assets<bevy_gaussian_splatting::GaussianLodScene>,
    >,
    #[cfg(feature = "io_sog")] runtimes: Query<(
        &bevy_gaussian_splatting::GaussianLodSceneHandle,
        Option<&bevy_gaussian_splatting::LodRuntime>,
    )>,
) {
    if *done {
        return;
    }

    let document = web_sys::window().and_then(|window| window.document());
    let Some(document) = document else {
        *done = true;
        return;
    };
    let overlay = document.get_element_by_id("loading");
    let status = document.get_element_by_id("loading-status");
    let fill = document.get_element_by_id("loading-fill");
    let (Some(overlay), Some(status), Some(fill)) = (overlay, status, fill) else {
        *done = true;
        return;
    };

    let mut finish = || {
        overlay.set_class_name("done");
        *done = true;
    };

    #[cfg(feature = "io_sog")]
    {
        if let Some((handle, runtime)) = runtimes.iter().next() {
            let Some(scene) = scenes.get(&handle.0) else {
                status.set_text_content(Some("loading scene index…"));
                return;
            };
            let total = scene.leaves.len().max(1);
            let active = runtime
                .map(|runtime| runtime.active_lods().count())
                .unwrap_or(0);

            let percent = (active as f32 / total as f32) * 100.0;
            let _ = fill.set_attribute("style", &format!("width:{percent:.0}%"));
            status.set_text_content(Some(&format!(
                "loading scene… {active} / {total} chunks"
            )));

            if active >= total {
                finish();
            }
            return;
        }
    }

    // no streamed scene (plain cloud / test scene): reveal once the renderer
    // has produced a few frames
    if frames.0 > 10 {
        finish();
    }
}

const PANEL_BUTTON_BG: Color = Color::srgba(0.0, 0.0, 0.0, 0.6);
const PANEL_BUTTON_HOVER: Color = Color::srgba(0.25, 0.25, 0.25, 0.7);
const PANEL_BUTTON_ACTIVE: Color = Color::srgba(0.2, 0.5, 0.2, 0.8);

fn panel_button(builder: &mut ChildSpawnerCommands, label: &str, bundle: impl Bundle) {
    builder
        .spawn((
            Button,
            bundle,
            Node {
                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                margin: UiRect::all(Val::Px(1.0)),
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(PANEL_BUTTON_BG),
        ))
        .with_children(|button| {
            button.spawn((
                Text(label.to_string()),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
        });
}

fn panel_label(builder: &mut ChildSpawnerCommands, label: &str) {
    builder.spawn((
        Text(label.to_string()),
        TextFont {
            font_size: FontSize::Px(12.0),
            ..default()
        },
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.6)),
        Node {
            margin: UiRect::top(Val::Px(6.0)),
            ..default()
        },
    ));
}

fn panel_row(builder: &mut ChildSpawnerCommands, spawn: impl FnOnce(&mut ChildSpawnerCommands)) {
    builder
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            ..default()
        })
        .with_children(spawn);
}

fn control_panel_setup(mut commands: Commands) {
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .with_children(|panel| {
            panel_label(panel, "scene rotation");
            for (axis, name) in ["X", "Y", "Z"].iter().enumerate() {
                panel_row(panel, |row| {
                    panel_label(row, name);
                    for degrees in [-90.0, -5.0, 5.0, 90.0] {
                        let label = if degrees > 0.0 {
                            format!("+{degrees}")
                        } else {
                            format!("{degrees}")
                        };
                        panel_button(row, &label, PanelAction::Rotate { axis, degrees });
                    }
                });
            }
            panel_row(panel, |row| {
                panel_button(row, "reset rotation", PanelAction::ResetRotation);
            });

            panel_label(panel, "move (hold)");
            panel_row(panel, |row| {
                panel_button(row, "forward", MoveAction(Vec3::new(0.0, 0.0, 1.0)));
                panel_button(row, "back", MoveAction(Vec3::new(0.0, 0.0, -1.0)));
                panel_button(row, "left", MoveAction(Vec3::new(-1.0, 0.0, 0.0)));
                panel_button(row, "right", MoveAction(Vec3::new(1.0, 0.0, 0.0)));
            });
            panel_row(panel, |row| {
                panel_button(row, "up", MoveAction(Vec3::new(0.0, 1.0, 0.0)));
                panel_button(row, "down", MoveAction(Vec3::new(0.0, -1.0, 0.0)));
            });

            panel_row(panel, |row| {
                panel_button(row, "orbit/fly (F)", PanelAction::ToggleFly);
                panel_button(row, "save view (P)", PanelAction::SaveView);
            });
        });
}

#[allow(clippy::type_complexity)]
fn panel_click_system(
    mut interactions: Query<
        (&Interaction, &PanelAction, &mut BackgroundColor),
        Changed<Interaction>,
    >,
    mut toggle_requests: MessageWriter<ToggleFlyRequest>,
    mut cloud_roots: Query<(&mut Transform, &CloudRoot)>,
    cameras: Query<(&Transform, &PanOrbitCamera, Option<&FlyMode>), (With<ViewerMainCamera>, Without<CloudRoot>)>,
) {
    for (interaction, action, mut background) in &mut interactions {
        match interaction {
            Interaction::Pressed => {
                *background = BackgroundColor(PANEL_BUTTON_ACTIVE);
                match action {
                    PanelAction::ToggleFly => {
                        toggle_requests.write(ToggleFlyRequest);
                    }
                    PanelAction::Rotate { axis, degrees } => {
                        let world_axis = [Vec3::X, Vec3::Y, Vec3::Z][*axis];
                        let step = Quat::from_axis_angle(world_axis, degrees.to_radians());
                        for (mut transform, _) in &mut cloud_roots {
                            transform.rotation = step * transform.rotation;
                        }
                    }
                    PanelAction::ResetRotation => {
                        for (mut transform, root) in &mut cloud_roots {
                            transform.rotation = root.initial_rotation;
                        }
                    }
                    PanelAction::SaveView => {
                        let rotation = cloud_roots
                            .iter()
                            .next()
                            .map(|(transform, _)| transform.rotation);
                        save_camera_pose(&cameras, rotation);
                    }
                }
            }
            Interaction::Hovered => {
                *background = BackgroundColor(PANEL_BUTTON_HOVER);
            }
            Interaction::None => {
                *background = BackgroundColor(PANEL_BUTTON_BG);
            }
        }
    }
}

/// hold-to-move: applies while the button stays pressed
#[allow(clippy::type_complexity)]
fn panel_move_system(
    time: Res<Time>,
    buttons: Query<(&Interaction, &MoveAction)>,
    mut cameras: Query<
        (&mut Transform, &mut PanOrbitCamera, Option<&FlyMode>),
        With<ViewerMainCamera>,
    >,
) {
    let mut direction = Vec3::ZERO;
    for (interaction, action) in &buttons {
        if *interaction == Interaction::Pressed {
            direction += action.0;
        }
    }
    if direction == Vec3::ZERO {
        return;
    }

    let Ok((mut transform, mut pan_orbit, fly)) = cameras.single_mut() else {
        return;
    };

    let speed = fly.map(|fly| fly.speed).unwrap_or(3.0);
    let forward = *transform.forward();
    let right = *transform.right();
    let delta = (right * direction.x + Vec3::Y * direction.y + forward * direction.z)
        .normalize_or_zero()
        * speed
        * time.delta_secs();

    transform.translation += delta;
    if fly.is_none() {
        // orbit mode: pan the focus along with the camera
        let focus = pan_orbit.focus + delta;
        pan_orbit.focus = focus;
        pan_orbit.target_focus = focus;
    }
}

pub fn press_s_screenshot(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    current_frame: Res<FrameCount>,
    flying: Query<(), With<FlyMode>>,
) {
    // S is "move backward" while flying
    if !flying.is_empty() {
        return;
    }

    if keys.just_pressed(KeyCode::KeyS) {
        let images_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("screenshots");
        std::fs::create_dir_all(&images_dir).unwrap();
        let output_path = images_dir.join(format!("output_{}.png", current_frame.0));

        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(output_path));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn press_g_save_gltf_scene(
    keys: Res<ButtonInput<KeyCode>>,
    current_frame: Res<FrameCount>,
    gaussian_cloud_assets: Res<Assets<PlanarGaussian3d>>,
    gaussian_clouds: Query<ExportCloudQuery>,
    cameras: Query<ExportCameraQuery, (With<GaussianCamera>, With<ViewerMainCamera>)>,
) {
    if !keys.just_pressed(KeyCode::KeyG) {
        return;
    }

    let mut export_clouds = Vec::new();
    for (index, (cloud_handle, global_transform, name, settings, metadata)) in
        gaussian_clouds.iter().enumerate()
    {
        let Some(cloud) = gaussian_cloud_assets.get(&cloud_handle.0) else {
            continue;
        };

        export_clouds.push(SceneExportCloud {
            cloud: cloud.clone(),
            name: name
                .map(|value| value.as_str().to_owned())
                .unwrap_or_else(|| format!("gaussian_cloud_{index}")),
            settings: settings.cloned().unwrap_or_default(),
            transform: Transform::from_matrix(global_transform.to_matrix()),
            metadata: metadata.cloned().unwrap_or_default(),
        });
    }

    if export_clouds.is_empty() {
        log("no gaussian clouds available to export");
        return;
    }

    let export_camera = cameras
        .iter()
        .next()
        .map(|(global_transform, name)| SceneExportCamera {
            name: name
                .map(|value| value.as_str().to_owned())
                .unwrap_or_else(|| "viewer_camera".to_owned()),
            transform: Transform::from_matrix(global_transform.to_matrix()),
            ..default()
        });

    let output_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("exports");
    if let Err(err) = std::fs::create_dir_all(&output_dir) {
        log(&format!(
            "failed to create export directory '{}': {err}",
            output_dir.display()
        ));
        return;
    }

    let output_path = output_dir.join(format!("gaussian_scene_{}.glb", current_frame.0));
    match write_khr_gaussian_scene_glb(&output_path, &export_clouds, export_camera.as_ref()) {
        Ok(()) => log(&format!(
            "saved gaussian scene to {}",
            output_path.display()
        )),
        Err(err) => log(&format!(
            "failed to save gaussian scene '{}': {err}",
            output_path.display()
        )),
    }
}

#[cfg(target_arch = "wasm32")]
fn press_g_save_gltf_scene(keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::KeyG) {
        log("GLB scene export is not supported on wasm32");
    }
}

#[derive(Component, Debug, Default, Reflect)]
pub struct ShowAxes;

fn draw_axes(mut gizmos: Gizmos, query: Query<(&Transform, &Aabb), With<ShowAxes>>) {
    for (&transform, aabb) in &query {
        let length = aabb.half_extents.length();
        gizmos.axes(transform, length);
    }
}

pub fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

#[cfg(feature = "query_select")]
fn press_i_invert_selection(
    keys: Res<ButtonInput<KeyCode>>,
    mut select_inverse_events: MessageWriter<InvertSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::KeyI) {
        log("inverting selection");
        select_inverse_events.write(InvertSelectionEvent);
    }
}

#[cfg(feature = "query_select")]
fn press_o_save_selection(
    keys: Res<ButtonInput<KeyCode>>,
    mut select_inverse_events: MessageWriter<SaveSelectionEvent>,
) {
    if keys.just_pressed(KeyCode::KeyO) {
        log("saving selection");
        select_inverse_events.write(SaveSelectionEvent);
    }
}

fn fps_display_setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn((
            Text("fps: ".to_string()),
            TextFont {
                font: FontSource::Handle(asset_server.load("fonts/Caveat-Bold.ttf")),
                font_size: FontSize::Px(60.0),
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(5.0),
                left: Val::Px(15.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            FpsText,
            TextColor(Color::Srgba(GOLD)),
            TextFont {
                font: FontSource::Handle(asset_server.load("fonts/Caveat-Bold.ttf")),
                font_size: FontSize::Px(60.0),
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct FpsText;

#[derive(Default)]
struct FpsDisplayState {
    smoothed_fps: Option<f64>,
    update_elapsed_secs: f32,
}

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    mut state: Local<FpsDisplayState>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) else {
        return;
    };
    let Some(value) = fps.smoothed() else {
        return;
    };

    const SMOOTHING_ALPHA: f64 = 0.08;
    const DISPLAY_UPDATE_INTERVAL_SECS: f32 = 0.5;

    let smoothed_fps = state.smoothed_fps.map_or(value, |current| {
        current + (value - current) * SMOOTHING_ALPHA
    });
    state.smoothed_fps = Some(smoothed_fps);

    state.update_elapsed_secs += time.delta_secs();
    if state.update_elapsed_secs < DISPLAY_UPDATE_INTERVAL_SECS {
        return;
    }
    state.update_elapsed_secs = 0.0;

    let display_fps = smoothed_fps.round() as u32;
    for mut text in &mut query {
        **text = display_fps.to_string();
    }
}

#[cfg(all(test, feature = "web_asset"))]
mod tests {
    use super::parse_input_file;

    #[test]
    fn decodes_percent_encoded_input_url() {
        let encoded = "https%3A%2F%2Fmitchell.mosure.me%2Ftrellis.glb";
        let decoded = parse_input_file(encoded);
        assert_eq!(decoded, "https://mitchell.mosure.me/trellis.glb");
    }

    #[test]
    fn keeps_plain_relative_path() {
        let input = "trellis.glb";
        let parsed = parse_input_file(input);
        assert_eq!(parsed, "trellis.glb");
    }
}

pub fn main() {
    setup_hooks();
    viewer_app();
}
