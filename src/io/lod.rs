//! Streamed SOG LOD scene (`lod-meta.json`) format.
//!
//! Parses the `splat-transform` LOD output (see `docs/lod-design.md`): a binary
//! kd-tree whose leaves reference contiguous `(file, offset, count)` intervals
//! inside per-LOD SOG units. The loader is parse-only; unit loading, distance-
//! based LOD selection and eviction live in `crate::stream::sog_lod`.

use std::collections::HashMap;
use std::io::{Error, ErrorKind};

use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    prelude::*,
};
use serde::Deserialize;

use crate::stream::sog_lod::{LodSettings, SogLodStreamingPlugin};

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

/// Flattened kd-tree leaf: scene-local bounds plus one optional interval per
/// LOD level. This is the unit of runtime LOD selection.
#[derive(Clone, Debug, Reflect)]
pub struct LodLeaf {
    pub min: Vec3,
    pub max: Vec3,
    pub lods: Vec<Option<LodInterval>>,
}

impl LodLeaf {
    /// Squared distance from `point` to this leaf's AABB (0 inside).
    pub fn distance_squared(&self, point: Vec3) -> f32 {
        point.clamp(self.min, self.max).distance_squared(point)
    }

    /// The interval to display for a desired level: exact if present, else the
    /// nearest coarser level, else the nearest finer one.
    pub fn nearest_available(&self, desired: usize) -> Option<usize> {
        let desired = desired.min(self.lods.len().saturating_sub(1));
        if self.lods.get(desired)?.is_some() {
            return Some(desired);
        }
        for coarser in desired + 1..self.lods.len() {
            if self.lods[coarser].is_some() {
                return Some(coarser);
            }
        }
        (0..desired)
            .rev()
            .find(|&finer| self.lods[finer].is_some())
    }
}

#[derive(Asset, Clone, Debug, Default, Reflect)]
pub struct GaussianLodScene {
    pub lod_levels: usize,
    /// splat count per LOD level
    pub counts: Vec<usize>,
    /// SOG unit paths relative to lod-meta.json, named `{lod}_{index}/meta.json`
    pub filenames: Vec<String>,
    pub leaves: Vec<LodLeaf>,
    /// directory of lod-meta.json as a full asset path string (including
    /// source), used by the streaming runtime to resolve unit files
    pub base_dir: String,
}

impl GaussianLodScene {
    /// Full asset path of a unit file relative to the scene.
    pub fn unit_path(&self, relative: &str) -> String {
        if self.base_dir.is_empty() {
            relative.to_owned()
        } else {
            format!("{}/{}", self.base_dir, relative)
        }
    }
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

pub(crate) fn parse_lod_meta(bytes: &[u8]) -> Result<(GaussianLodScene, usize), Error> {
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

    Ok((
        GaussianLodScene {
            lod_levels: raw.lod_levels,
            counts: raw.counts,
            filenames: raw.filenames,
            leaves,
            base_dir: String::new(),
        },
        raw.count,
    ))
}

#[derive(Default, TypePath)]
pub struct GaussianLodSceneLoader;

impl AssetLoader for GaussianLodSceneLoader {
    type Asset = GaussianLodScene;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        let (mut scene, _) = parse_lod_meta(&bytes)?;

        let full_path = load_context.path().to_string();
        scene.base_dir = match full_path.rsplit_once('/') {
            Some((dir, _)) => dir.to_owned(),
            None => String::new(),
        };

        Ok(scene)
    }

    fn extensions(&self) -> &[&str] {
        // splat-transform writes literally `lod-meta.json` (full extension
        // "json"); `*.lod-meta.json` also matches for apps that rename
        &["lod-meta.json", "json"]
    }
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[require(Transform, Visibility, LodSettings)]
pub struct GaussianLodSceneHandle(pub Handle<GaussianLodScene>);

#[derive(Default)]
pub struct GaussianLodScenePlugin;

impl Plugin for GaussianLodScenePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<LodInterval>();
        app.register_type::<LodLeaf>();
        app.register_type::<GaussianLodScene>();
        app.register_type::<GaussianLodSceneHandle>();

        app.init_asset::<GaussianLodScene>();
        app.init_asset_loader::<GaussianLodSceneLoader>();

        app.add_plugins(SogLodStreamingPlugin);
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
        let (scene, count) = parse_lod_meta(META.as_bytes()).unwrap();

        assert_eq!(count, 300);
        assert_eq!(scene.lod_levels, 2);
        assert_eq!(scene.counts, vec![200, 100]);
        assert_eq!(scene.filenames.len(), 2);
        assert_eq!(scene.leaves.len(), 2);

        let leaf = &scene.leaves[0];
        assert_eq!(leaf.min, Vec3::new(-2.0, 0.0, -2.0));
        assert_eq!(leaf.max, Vec3::new(0.0, 2.0, 2.0));

        let lod0 = leaf.lods[0].as_ref().unwrap();
        assert_eq!((lod0.file, lod0.offset, lod0.count), (0, 0, 120));
        let lod1 = leaf.lods[1].as_ref().unwrap();
        assert_eq!((lod1.file, lod1.offset, lod1.count), (1, 0, 60));

        let second_lod0 = scene.leaves[1].lods[0].as_ref().unwrap();
        assert_eq!(
            (second_lod0.file, second_lod0.offset, second_lod0.count),
            (0, 120, 80)
        );
    }

    #[test]
    fn intervals_tile_units() {
        let (scene, _) = parse_lod_meta(META.as_bytes()).unwrap();

        for lod in 0..scene.lod_levels {
            let total: usize = scene
                .leaves
                .iter()
                .filter_map(|leaf| leaf.lods[lod].as_ref())
                .map(|interval| interval.count)
                .sum();
            assert_eq!(total, scene.counts[lod]);
        }
    }

    #[test]
    fn rejects_unknown_version() {
        let bad = META.replace("\"version\": 1", "\"version\": 9");
        assert!(parse_lod_meta(bad.as_bytes()).is_err());
    }

    #[test]
    fn nearest_available_falls_back() {
        let leaf = LodLeaf {
            min: Vec3::ZERO,
            max: Vec3::ONE,
            lods: vec![
                None,
                Some(LodInterval {
                    file: 0,
                    offset: 0,
                    count: 1,
                }),
                None,
            ],
        };

        assert_eq!(leaf.nearest_available(1), Some(1)); // exact
        assert_eq!(leaf.nearest_available(0), Some(1)); // coarser fallback
        assert_eq!(leaf.nearest_available(2), Some(1)); // finer fallback
        assert_eq!(leaf.nearest_available(9), Some(1)); // clamped

        let empty = LodLeaf {
            min: Vec3::ZERO,
            max: Vec3::ONE,
            lods: vec![None, None],
        };
        assert_eq!(empty.nearest_available(0), None);
    }

    #[test]
    fn distance_to_aabb() {
        let leaf = LodLeaf {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
            lods: vec![],
        };

        assert_eq!(leaf.distance_squared(Vec3::ZERO), 0.0); // inside
        assert_eq!(leaf.distance_squared(Vec3::new(3.0, 0.0, 0.0)), 4.0);
        assert_eq!(leaf.distance_squared(Vec3::new(2.0, 2.0, 0.0)), 2.0);
    }
}
