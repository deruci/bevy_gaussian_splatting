//! Streamed SOG LOD scene (`lod-meta.json`) loader.
//!
//! Parses the `splat-transform` LOD output (see `docs/lod-design.md`): a binary
//! kd-tree whose leaves reference contiguous `(file, offset, count)` intervals
//! inside per-LOD SOG units. This milestone loads one fixed LOD level — every
//! unit of that level is decoded whole and spawned as a child cloud. Distance-
//! based per-leaf selection and streaming build on the leaf table kept in the
//! asset.

use std::collections::{BTreeSet, HashMap};
use std::io::{Error, ErrorKind};

use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::{
    gaussian::formats::planar_3d::{PlanarGaussian3d, PlanarGaussian3dHandle},
    io::{
        scene::CloudBundle,
        sog::{SogMeta, decode_range, pad_to_multiple_of_32},
    },
};

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

#[derive(Clone, Debug, Deserialize)]
struct RawBound {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Clone, Debug, Deserialize)]
struct RawLod {
    file: usize,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RawNode {
    bound: RawBound,
    children: Option<Vec<RawNode>>,
    lods: Option<HashMap<String, RawLod>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawLodMeta {
    version: u32,
    #[allow(dead_code)]
    count: usize,
    counts: Vec<usize>,
    lod_levels: usize,
    /// optional skydome unit path; not loaded yet (see docs/lod-design.md)
    #[serde(default)]
    #[allow(dead_code)]
    environment: Option<String>,
    filenames: Vec<String>,
    tree: RawNode,
}

/// A leaf's contiguous row range inside one SOG unit (`filenames[file]`).
#[derive(Clone, Debug, Reflect)]
pub struct LodInterval {
    pub file: usize,
    pub offset: usize,
    pub count: usize,
}

/// Flattened kd-tree leaf: world-space bounds plus one optional interval per
/// LOD level. This is the unit of runtime LOD selection.
#[derive(Clone, Debug, Reflect)]
pub struct LodLeaf {
    pub min: Vec3,
    pub max: Vec3,
    pub lods: Vec<Option<LodInterval>>,
}

#[derive(Asset, Clone, Debug, Default, Reflect)]
pub struct GaussianLodScene {
    pub lod_levels: usize,
    /// splat count per LOD level
    pub counts: Vec<usize>,
    /// SOG unit paths relative to lod-meta.json, named `{lod}_{index}/meta.json`
    pub filenames: Vec<String>,
    pub leaves: Vec<LodLeaf>,
    /// LOD level the bundles below were decoded at
    pub loaded_lod: usize,
    pub bundles: Vec<CloudBundle>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LodLoadSettings {
    /// LOD level to load (0 = finest); clamped to the scene's available levels
    pub lod: usize,
}

fn flatten_leaves(node: &RawNode, lod_levels: usize, leaves: &mut Vec<LodLeaf>) {
    if let Some(children) = &node.children {
        for child in children {
            flatten_leaves(child, lod_levels, leaves);
        }
        return;
    }

    let mut lods = vec![None; lod_levels];
    if let Some(raw_lods) = &node.lods {
        for (key, raw) in raw_lods {
            if let Ok(level) = key.parse::<usize>()
                && level < lod_levels
                && raw.count > 0
            {
                lods[level] = Some(LodInterval {
                    file: raw.file,
                    offset: raw.offset,
                    count: raw.count,
                });
            }
        }
    }

    leaves.push(LodLeaf {
        min: Vec3::from_array(node.bound.min),
        max: Vec3::from_array(node.bound.max),
        lods,
    });
}

fn parse_lod_meta(bytes: &[u8]) -> Result<(RawLodMeta, Vec<LodLeaf>), Error> {
    let raw: RawLodMeta = serde_json::from_slice(bytes)
        .map_err(|e| invalid(format!("failed to parse lod-meta.json: {e}")))?;
    if raw.version != 1 {
        return Err(invalid(format!(
            "unsupported lod-meta version: {}",
            raw.version
        )));
    }

    let mut leaves = Vec::new();
    flatten_leaves(&raw.tree, raw.lod_levels, &mut leaves);

    Ok((raw, leaves))
}

#[derive(Default, TypePath)]
pub struct GaussianLodSceneLoader;

impl AssetLoader for GaussianLodSceneLoader {
    type Asset = GaussianLodScene;
    type Settings = LodLoadSettings;
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        let (raw, leaves) = parse_lod_meta(&bytes)?;

        let lod = settings.lod.min(raw.lod_levels.saturating_sub(1));

        // every unit referenced by any leaf at the chosen level; a fixed level's
        // leaf intervals tile their units exactly, so whole-unit decode is
        // equivalent to per-leaf interval decode and avoids repeated texture work
        let unit_indices: BTreeSet<usize> = leaves
            .iter()
            .filter_map(|leaf| leaf.lods.get(lod).and_then(|l| l.as_ref()))
            .map(|interval| interval.file)
            .collect();

        async fn read_relative(
            load_context: &mut LoadContext<'_>,
            relative: &str,
        ) -> Result<Vec<u8>, Error> {
            let path = load_context
                .path()
                .resolve_embed_str(relative)
                .map_err(|e| invalid(format!("invalid unit path '{relative}': {e}")))?;
            load_context
                .read_asset_bytes(path)
                .await
                .map_err(|e| Error::new(ErrorKind::NotFound, format!("{relative}: {e}")))
        }

        let mut bundles = Vec::new();

        for unit_index in unit_indices {
            let unit_path = raw
                .filenames
                .get(unit_index)
                .ok_or_else(|| invalid(format!("leaf references missing file {unit_index}")))?
                .clone();
            let unit_dir = unit_path.rsplit_once('/').map(|(dir, _)| dir);

            let meta_bytes = read_relative(load_context, &unit_path).await?;
            let meta = SogMeta::from_json(&meta_bytes)?;

            let mut files: HashMap<String, Vec<u8>> = HashMap::new();
            for name in meta.texture_files() {
                let relative = match unit_dir {
                    Some(dir) => format!("{dir}/{name}"),
                    None => name.to_owned(),
                };
                let texture_bytes = read_relative(load_context, &relative).await?;
                files.insert(name.to_owned(), texture_bytes);
            }

            let textures = meta.load_textures(|name| {
                files
                    .get(name)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing texture {name}")))
            })?;
            let gaussians = decode_range(&meta, &textures, 0, meta.count)?;
            let cloud: PlanarGaussian3d = pad_to_multiple_of_32(gaussians).into();

            let name = format!("lod{lod}_unit{unit_index}");
            let cloud_handle = load_context.add_labeled_asset(name.clone(), cloud);

            bundles.push(CloudBundle {
                cloud: cloud_handle,
                name,
                ..Default::default()
            });
        }

        Ok(GaussianLodScene {
            lod_levels: raw.lod_levels,
            counts: raw.counts,
            filenames: raw.filenames,
            leaves,
            loaded_lod: lod,
            bundles,
        })
    }

    fn extensions(&self) -> &[&str] {
        // splat-transform writes literally `lod-meta.json` (full extension
        // "json"); `*.lod-meta.json` also matches for apps that rename
        &["lod-meta.json", "json"]
    }
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[require(Transform, Visibility)]
pub struct GaussianLodSceneHandle(pub Handle<GaussianLodScene>);

#[derive(Component, Clone, Debug, Default, Reflect)]
pub struct GaussianLodSceneLoaded;

#[derive(Default)]
pub struct GaussianLodScenePlugin;

impl Plugin for GaussianLodScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LodInterval>();
        app.register_type::<LodLeaf>();
        app.register_type::<GaussianLodScene>();
        app.register_type::<GaussianLodSceneHandle>();
        app.register_type::<GaussianLodSceneLoaded>();

        app.init_asset::<GaussianLodScene>();
        app.init_asset_loader::<GaussianLodSceneLoader>();

        app.add_systems(Update, spawn_lod_scene);
    }
}

fn spawn_lod_scene(
    mut commands: Commands,
    scene_handles: Query<(Entity, &GaussianLodSceneHandle), Without<GaussianLodSceneLoaded>>,
    asset_server: Res<AssetServer>,
    scenes: Res<Assets<GaussianLodScene>>,
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

        let bundles = scene.bundles.clone();

        commands
            .entity(entity)
            .with_children(move |builder| {
                for bundle in bundles {
                    builder.spawn((
                        PlanarGaussian3dHandle(bundle.cloud.clone()),
                        Name::new(bundle.name.clone()),
                        bundle.settings.clone(),
                        bundle.transform,
                        bundle.metadata.clone(),
                    ));
                }
            })
            .insert(GaussianLodSceneLoaded);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const META: &str = r#"{
        "version": 1,
        "asset": { "generator": "splat-transform v3.0.0" },
        "count": 300,
        "counts": [200, 100],
        "lodLevels": 2,
        "filenames": ["0_0/meta.json", "1_0/meta.json"],
        "tree": {
            "bound": { "min": [-2, 0, -2], "max": [2, 2, 2] },
            "children": [
                {
                    "bound": { "min": [-2, 0, -2], "max": [0, 2, 2] },
                    "lods": {
                        "0": { "file": 0, "offset": 0, "count": 120 },
                        "1": { "file": 1, "offset": 0, "count": 60 }
                    }
                },
                {
                    "bound": { "min": [0, 0, -2], "max": [2, 2, 2] },
                    "lods": {
                        "0": { "file": 0, "offset": 120, "count": 80 },
                        "1": { "file": 1, "offset": 60, "count": 40 }
                    }
                }
            ]
        }
    }"#;

    #[test]
    fn flattens_leaves() {
        let (raw, leaves) = parse_lod_meta(META.as_bytes()).unwrap();

        assert_eq!(raw.lod_levels, 2);
        assert_eq!(raw.counts, vec![200, 100]);
        assert_eq!(raw.filenames.len(), 2);
        assert_eq!(leaves.len(), 2);

        let leaf = &leaves[0];
        assert_eq!(leaf.min, Vec3::new(-2.0, 0.0, -2.0));
        assert_eq!(leaf.max, Vec3::new(0.0, 2.0, 2.0));

        let lod0 = leaf.lods[0].as_ref().unwrap();
        assert_eq!((lod0.file, lod0.offset, lod0.count), (0, 0, 120));
        let lod1 = leaf.lods[1].as_ref().unwrap();
        assert_eq!((lod1.file, lod1.offset, lod1.count), (1, 0, 60));

        let second_lod0 = leaves[1].lods[0].as_ref().unwrap();
        assert_eq!(
            (second_lod0.file, second_lod0.offset, second_lod0.count),
            (0, 120, 80)
        );
    }

    #[test]
    fn intervals_tile_units() {
        let (raw, leaves) = parse_lod_meta(META.as_bytes()).unwrap();

        for lod in 0..raw.lod_levels {
            let total: usize = leaves
                .iter()
                .filter_map(|leaf| leaf.lods[lod].as_ref())
                .map(|interval| interval.count)
                .sum();
            assert_eq!(total, raw.counts[lod]);
        }
    }

    #[test]
    fn rejects_unknown_version() {
        let bad = META.replace("\"version\": 1", "\"version\": 9");
        assert!(parse_lod_meta(bad.as_bytes()).is_err());
    }
}
