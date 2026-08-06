//! SOG v2 reader (PlayCanvas compressed gaussian splat format).
//!
//! A SOG is a `meta.json` plus WebP textures, optionally bundled as a single
//! `.sog` ZIP. Decoding follows `splat-transform`'s `read-sog.ts` (MIT); see
//! `docs/lod-design.md` for the texel layout and the mapping onto this crate's
//! gaussian conventions.

use std::io::{Error, ErrorKind, Read};

use serde::Deserialize;

use crate::{
    gaussian::formats::planar_3d::{Gaussian3d, PlanarGaussian3d},
    material::spherical_harmonics::{SH_CHANNELS, SH_COEFF_COUNT},
};
use bevy_interleave::prelude::Planar;

// same clamp as the PLY path (io/ply.rs), kept local so io_sog builds without io_ply
const MAX_SIZE_VARIANCE: f32 = 4.0;

#[derive(Clone, Debug, Deserialize)]
pub struct SogMeans {
    pub mins: [f32; 3],
    pub maxs: [f32; 3],
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SogCodebook {
    pub codebook: Vec<f32>,
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SogQuats {
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SogShN {
    pub count: usize,
    pub bands: usize,
    pub codebook: Vec<f32>,
    pub files: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SogMeta {
    pub version: u32,
    pub count: usize,
    pub means: SogMeans,
    pub scales: SogCodebook,
    pub quats: SogQuats,
    pub sh0: SogCodebook,
    #[serde(rename = "shN")]
    pub sh_n: Option<SogShN>,
}

/// Decoded RGBA8 textures backing one SOG unit. Texel `g` of each texture is
/// the flat index `g * 4` — width/height only matter for capacity.
pub struct SogTextures {
    pub means_l: Vec<u8>,
    pub means_u: Vec<u8>,
    pub quats: Vec<u8>,
    pub scales: Vec<u8>,
    pub sh0: Vec<u8>,
    /// (rgba, width) — centroid lookups are 2d, unlike the per-gaussian textures
    pub sh_n_centroids: Option<(Vec<u8>, usize)>,
    pub sh_n_labels: Option<Vec<u8>>,
}

const REST_COEFFS_PER_BAND: [usize; 4] = [0, 3, 8, 15];

// component order of the three packed values per maxComp row (indices into [w, x, y, z])
const QUAT_IDX: [usize; 12] = [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2];

// inverse of logTransform(x) = sign(x) * ln(|x| + 1)
fn inv_log_transform(v: f32) -> f32 {
    v.signum() * (v.abs().exp() - 1.0)
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, msg.into())
}

fn decode_webp(name: &str, bytes: &[u8]) -> Result<(Vec<u8>, usize), Error> {
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::WebP)
        .map_err(|e| invalid(format!("failed to decode {name}: {e}")))?;
    let rgba = img.to_rgba8();
    let width = rgba.width() as usize;
    Ok((rgba.into_raw(), width))
}

impl SogMeta {
    /// All texture filenames referenced by this meta, in load order.
    pub fn texture_files(&self) -> Vec<&str> {
        let mut files: Vec<&str> = Vec::new();
        files.extend(self.means.files.iter().map(String::as_str));
        files.extend(self.quats.files.iter().map(String::as_str));
        files.extend(self.scales.files.iter().map(String::as_str));
        files.extend(self.sh0.files.iter().map(String::as_str));
        if let Some(sh_n) = &self.sh_n {
            files.extend(sh_n.files.iter().map(String::as_str));
        }
        files
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        let meta: SogMeta = serde_json::from_slice(bytes)
            .map_err(|e| invalid(format!("failed to parse SOG meta.json: {e}")))?;
        if meta.version != 2 {
            return Err(invalid(format!(
                "unsupported SOG meta version: {} (only v2 supported)",
                meta.version
            )));
        }
        Ok(meta)
    }

    /// Load and decode all textures referenced by this meta. `fetch` resolves a
    /// filename from meta.json (relative to its directory) to file bytes.
    pub fn load_textures(
        &self,
        mut fetch: impl FnMut(&str) -> Result<Vec<u8>, Error>,
    ) -> Result<SogTextures, Error> {
        let mut load = |name: &str| -> Result<(Vec<u8>, usize), Error> {
            decode_webp(name, &fetch(name)?)
        };

        if self.means.files.len() < 2 {
            return Err(invalid("SOG means requires two files (low + high bytes)"));
        }

        let means_l = load(&self.means.files[0])?;
        let means_u = load(&self.means.files[1])?;
        let quats = load(&self.quats.files[0])?;
        let scales = load(&self.scales.files[0])?;
        let sh0 = load(&self.sh0.files[0])?;

        for (name, (rgba, _)) in [
            (&self.means.files[0], &means_l),
            (&self.means.files[1], &means_u),
            (&self.quats.files[0], &quats),
            (&self.scales.files[0], &scales),
            (&self.sh0.files[0], &sh0),
        ] {
            if rgba.len() / 4 < self.count {
                return Err(invalid(format!("SOG texture {name} too small for count")));
            }
        }

        let (sh_n_centroids, sh_n_labels) = match &self.sh_n {
            Some(sh_n) if REST_COEFFS_PER_BAND.get(sh_n.bands).copied().unwrap_or(0) > 0 => {
                let coeffs = REST_COEFFS_PER_BAND[sh_n.bands];
                let centroids = load(&sh_n.files[0])?;
                let labels = load(&sh_n.files[1])?;
                if labels.0.len() / 4 < self.count {
                    return Err(invalid("SOG shN labels texture too small for count"));
                }
                if centroids.1 != 64 * coeffs {
                    return Err(invalid(format!(
                        "SOG shN centroids width {} does not match expected {} for {}-band palette",
                        centroids.1,
                        64 * coeffs,
                        sh_n.bands
                    )));
                }
                (Some(centroids), Some(labels.0))
            }
            _ => (None, None),
        };

        Ok(SogTextures {
            means_l: means_l.0,
            means_u: means_u.0,
            quats: quats.0,
            scales: scales.0,
            sh0: sh0.0,
            sh_n_centroids,
            sh_n_labels,
        })
    }
}

/// Decode gaussians `offset..offset + count` from a SOG unit. Sub-range decode
/// is what lets the LOD loader expand a leaf's `(file, offset, count)` interval
/// without touching the rest of the unit.
pub fn decode_range(
    meta: &SogMeta,
    textures: &SogTextures,
    offset: usize,
    count: usize,
) -> Result<Vec<Gaussian3d>, Error> {
    if offset + count > meta.count {
        return Err(invalid(format!(
            "SOG range {}..{} out of bounds (count {})",
            offset,
            offset + count,
            meta.count
        )));
    }

    let scale_codebook = &meta.scales.codebook;
    let sh0_codebook = &meta.sh0.codebook;

    let (rest_coeffs, palette_count, sh_codebook) = match (&meta.sh_n, &textures.sh_n_centroids) {
        (Some(sh_n), Some(_)) => (
            REST_COEFFS_PER_BAND[sh_n.bands],
            sh_n.count,
            sh_n.codebook.as_slice(),
        ),
        _ => (0, 0, [].as_slice()),
    };

    let [x_min, y_min, z_min] = meta.means.mins;
    let x_scale = if meta.means.maxs[0] - x_min == 0.0 { 1.0 } else { meta.means.maxs[0] - x_min };
    let y_scale = if meta.means.maxs[1] - y_min == 0.0 { 1.0 } else { meta.means.maxs[1] - y_min };
    let z_scale = if meta.means.maxs[2] - z_min == 0.0 { 1.0 } else { meta.means.maxs[2] - z_min };

    let mut gaussians = Vec::with_capacity(count);

    for g in offset..offset + count {
        let o4 = g * 4;
        let mut gaussian = Gaussian3d::default();

        // position: 16-bit lerp between mins/maxs in log space
        let xv = (textures.means_l[o4] as u32 | ((textures.means_u[o4] as u32) << 8)) as f32;
        let yv = (textures.means_l[o4 + 1] as u32 | ((textures.means_u[o4 + 1] as u32) << 8)) as f32;
        let zv = (textures.means_l[o4 + 2] as u32 | ((textures.means_u[o4 + 2] as u32) << 8)) as f32;
        gaussian.position_visibility.position = [
            inv_log_transform(x_min + x_scale * (xv / 65535.0)),
            inv_log_transform(y_min + y_scale * (yv / 65535.0)),
            inv_log_transform(z_min + z_scale * (zv / 65535.0)),
        ];
        gaussian.position_visibility.visibility = 1.0;

        // rotation: smallest-three, alpha tag 252 + maxComp, output [w, x, y, z]
        let tag = textures.quats[o4 + 3] as usize;
        if (252..=255).contains(&tag) {
            let max_comp = tag - 252;
            let a = (textures.quats[o4] as f32 / 255.0 * 2.0 - 1.0) / std::f32::consts::SQRT_2;
            let b = (textures.quats[o4 + 1] as f32 / 255.0 * 2.0 - 1.0) / std::f32::consts::SQRT_2;
            let c = (textures.quats[o4 + 2] as f32 / 255.0 * 2.0 - 1.0) / std::f32::consts::SQRT_2;

            let mut quat = [0.0f32; 4];
            let base = max_comp * 3;
            quat[QUAT_IDX[base]] = a;
            quat[QUAT_IDX[base + 1]] = b;
            quat[QUAT_IDX[base + 2]] = c;
            quat[max_comp] = (1.0 - (a * a + b * b + c * c)).max(0.0).sqrt();

            let norm = quat.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut quat {
                    *v /= norm;
                }
            }
            gaussian.rotation.rotation = quat;
        } else {
            gaussian.rotation.rotation = [1.0, 0.0, 0.0, 0.0];
        }

        // scale: codebook holds log scales; clamp + exp matches the PLY path
        let log_scale = [
            *scale_codebook.get(textures.scales[o4] as usize).unwrap_or(&0.0),
            *scale_codebook.get(textures.scales[o4 + 1] as usize).unwrap_or(&0.0),
            *scale_codebook.get(textures.scales[o4 + 2] as usize).unwrap_or(&0.0),
        ];
        let mean_scale = (log_scale[0] + log_scale[1] + log_scale[2]) / 3.0;
        for (out, log_scale) in gaussian.scale_opacity.scale.iter_mut().zip(log_scale) {
            *out = log_scale
                .max(mean_scale - MAX_SIZE_VARIANCE)
                .min(mean_scale + MAX_SIZE_VARIANCE)
                .exp();
        }

        // opacity: sh0 alpha is the already-sigmoided opacity
        gaussian.scale_opacity.opacity = textures.sh0[o4 + 3] as f32 / 255.0;

        // SH dc: interleaved index of coefficient 0 is the channel itself
        for ch in 0..SH_CHANNELS {
            let v = *sh0_codebook.get(textures.sh0[o4 + ch] as usize).unwrap_or(&0.0);
            gaussian.spherical_harmonic.set(ch, v);
        }

        // SH rest: 16-bit label -> centroid texel row, channel-major per coefficient
        if rest_coeffs > 0 {
            let labels = textures.sh_n_labels.as_ref().unwrap();
            let (centroids, c_width) = textures.sh_n_centroids.as_ref().unwrap();
            let c_height = centroids.len() / 4 / c_width;

            let label = labels[o4] as usize | ((labels[o4 + 1] as usize) << 8);
            if label < palette_count {
                let cy = label / 64;
                let cx_base = (label % 64) * rest_coeffs;
                for j in 0..rest_coeffs {
                    let cx = cx_base + j;
                    let idx = (cy * c_width + cx) * 4;
                    let in_range = cy < c_height && cx < *c_width;
                    for ch in 0..SH_CHANNELS {
                        let code = if in_range { centroids[idx + ch] as usize } else { 0 };
                        let v = *sh_codebook.get(code).unwrap_or(&0.0);
                        let interleaved_idx = (j + 1) * SH_CHANNELS + ch;
                        if interleaved_idx < SH_COEFF_COUNT {
                            gaussian.spherical_harmonic.set(interleaved_idx, v);
                        }
                    }
                }
            }
        }

        gaussians.push(gaussian);
    }

    Ok(gaussians)
}

pub(crate) fn pad_to_multiple_of_32(mut gaussians: Vec<Gaussian3d>) -> Vec<Gaussian3d> {
    let pad = (32 - gaussians.len() % 32) % 32;
    gaussians.extend(std::iter::repeat_n(Gaussian3d::default(), pad));
    gaussians
}

/// Decode a full SOG unit given its meta.json bytes and a texture fetcher.
pub fn parse_sog(
    meta_bytes: &[u8],
    fetch: impl FnMut(&str) -> Result<Vec<u8>, Error>,
) -> Result<PlanarGaussian3d, Error> {
    let meta = SogMeta::from_json(meta_bytes)?;
    let textures = meta.load_textures(fetch)?;
    let gaussians = decode_range(&meta, &textures, 0, meta.count)?;

    Ok(PlanarGaussian3d::from_interleaved(pad_to_multiple_of_32(
        gaussians,
    )))
}

/// Decode a bundled `.sog` (a ZIP holding meta.json + textures, possibly under
/// a common directory prefix).
pub fn parse_sog_bundle(bytes: &[u8]) -> Result<PlanarGaussian3d, Error> {
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| invalid(format!("failed to open .sog bundle: {e}")))?;

    let names: Vec<String> = archive.file_names().map(str::to_owned).collect();
    let resolve = |name: &str| -> Result<String, Error> {
        names
            .iter()
            .find(|n| *n == name || n.ends_with(&format!("/{name}")))
            .cloned()
            .ok_or_else(|| invalid(format!("{name} not found in .sog bundle")))
    };

    let mut read_entry = |name: &str| -> Result<Vec<u8>, Error> {
        let path = resolve(name)?;
        let mut entry = archive
            .by_name(&path)
            .map_err(|e| invalid(format!("failed to read {path} from .sog bundle: {e}")))?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        Ok(bytes)
    };

    let meta_bytes = read_entry("meta.json")?;
    parse_sog(&meta_bytes, read_entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::spherical_harmonics::SH_DEGREE;

    fn encode_webp(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(std::io::Cursor::new(&mut bytes))
            .encode(rgba, width, height, image::ExtendedColorType::Rgba8)
            .unwrap();
        bytes
    }

    fn log_transform(x: f32) -> f32 {
        x.signum() * (x.abs() + 1.0).ln()
    }

    // Build a 2-gaussian SOG: gaussian 0 at `mins` (texel 0), gaussian 1 at
    // `maxs` (texel 65535), identity-ish rotation, known codebooks.
    fn test_sog() -> (SogMeta, std::collections::HashMap<String, Vec<u8>>) {
        let count = 2;
        let mins = [log_transform(-1.0), log_transform(-2.0), log_transform(-4.0)];
        let maxs = [log_transform(1.0), log_transform(2.0), log_transform(4.0)];

        let means_l = vec![
            0, 0, 0, 255, //
            255, 255, 255, 255,
        ];
        let means_u = vec![
            0, 0, 0, 255, //
            255, 255, 255, 255,
        ];

        // maxComp = 0 (w): tag 252, packed (x, y, z) all at midpoint => w = 1
        let quats = vec![
            128, 128, 128, 252, //
            128, 128, 128, 252,
        ];

        let mut scale_codebook = vec![0.0f32; 256];
        scale_codebook[7] = -2.0;
        let scales = vec![
            7, 7, 7, 255, //
            7, 7, 7, 255,
        ];

        let mut sh0_codebook = vec![0.0f32; 256];
        sh0_codebook[3] = 0.5;
        sh0_codebook[9] = -0.25;
        let sh0 = vec![
            3, 3, 3, 255, // opaque
            9, 9, 9, 51, // 0.2 opacity
        ];

        let mut files = std::collections::HashMap::new();
        files.insert("means_l.webp".to_owned(), encode_webp(&means_l, count, 1));
        files.insert("means_u.webp".to_owned(), encode_webp(&means_u, count, 1));
        files.insert("quats.webp".to_owned(), encode_webp(&quats, count, 1));
        files.insert("scales.webp".to_owned(), encode_webp(&scales, count, 1));
        files.insert("sh0.webp".to_owned(), encode_webp(&sh0, count, 1));

        let meta = SogMeta {
            version: 2,
            count: count as usize,
            means: SogMeans {
                mins,
                maxs,
                files: vec!["means_l.webp".to_owned(), "means_u.webp".to_owned()],
            },
            scales: SogCodebook {
                codebook: scale_codebook,
                files: vec!["scales.webp".to_owned()],
            },
            quats: SogQuats {
                files: vec!["quats.webp".to_owned()],
            },
            sh0: SogCodebook {
                codebook: sh0_codebook,
                files: vec!["sh0.webp".to_owned()],
            },
            sh_n: None,
        };

        (meta, files)
    }

    fn fetch_from(
        files: &std::collections::HashMap<String, Vec<u8>>,
    ) -> impl FnMut(&str) -> Result<Vec<u8>, Error> + '_ {
        move |name: &str| {
            files
                .get(name)
                .cloned()
                .ok_or_else(|| invalid(format!("missing {name}")))
        }
    }

    #[test]
    fn decodes_positions_scales_opacity() {
        let (meta, files) = test_sog();
        let textures = meta.load_textures(fetch_from(&files)).unwrap();
        let gaussians = decode_range(&meta, &textures, 0, 2).unwrap();

        let p0 = gaussians[0].position_visibility.position;
        let p1 = gaussians[1].position_visibility.position;
        for (v, expected) in p0.iter().zip([-1.0f32, -2.0, -4.0]) {
            assert!((v - expected).abs() < 1e-3, "{v} != {expected}");
        }
        for (v, expected) in p1.iter().zip([1.0f32, 2.0, 4.0]) {
            assert!((v - expected).abs() < 1e-3, "{v} != {expected}");
        }

        for g in &gaussians {
            for s in g.scale_opacity.scale {
                assert!((s - (-2.0f32).exp()).abs() < 1e-6);
            }
        }

        assert!((gaussians[0].scale_opacity.opacity - 1.0).abs() < 1e-6);
        assert!((gaussians[1].scale_opacity.opacity - 0.2).abs() < 1e-2);

        // rotation decodes to identity [w, x, y, z] within 8-bit precision
        for g in &gaussians {
            assert!((g.rotation.rotation[0] - 1.0).abs() < 1e-2);
            for v in &g.rotation.rotation[1..] {
                assert!(v.abs() < 1e-2);
            }
        }

        // SH dc from codebook
        assert!((gaussians[0].spherical_harmonic.coefficients[0] - 0.5).abs() < 1e-6);
        assert!((gaussians[1].spherical_harmonic.coefficients[0] + 0.25).abs() < 1e-6);
    }

    #[test]
    fn decodes_sub_range() {
        let (meta, files) = test_sog();
        let textures = meta.load_textures(fetch_from(&files)).unwrap();

        let all = decode_range(&meta, &textures, 0, 2).unwrap();
        let tail = decode_range(&meta, &textures, 1, 1).unwrap();

        assert_eq!(tail.len(), 1);
        assert_eq!(
            tail[0].position_visibility.position,
            all[1].position_visibility.position
        );

        assert!(decode_range(&meta, &textures, 2, 1).is_err());
    }

    #[test]
    fn decodes_sh_rest_palette() {
        if SH_DEGREE == 0 {
            return;
        }

        let (mut meta, mut files) = test_sog();

        // 1-band palette: 3 rest coefficients, centroid width 64 * 3
        let coeffs = 3;
        let c_width = 64 * coeffs;
        let mut codebook = vec![0.0f32; 256];
        codebook[5] = 0.75;

        // label 1 occupies columns 3..6 of row 0; every channel points at entry 5
        let mut centroids = vec![0u8; c_width * 4];
        for j in 0..coeffs {
            let idx = (coeffs + j) * 4;
            centroids[idx] = 5;
            centroids[idx + 1] = 5;
            centroids[idx + 2] = 5;
            centroids[idx + 3] = 255;
        }

        let labels = vec![
            1, 0, 0, 255, //
            1, 0, 0, 255,
        ];

        files.insert(
            "shN_centroids.webp".to_owned(),
            encode_webp(&centroids, c_width as u32, 1),
        );
        files.insert("shN_labels.webp".to_owned(), encode_webp(&labels, 2, 1));

        meta.sh_n = Some(SogShN {
            count: 2,
            bands: 1,
            codebook,
            files: vec!["shN_centroids.webp".to_owned(), "shN_labels.webp".to_owned()],
        });

        let textures = meta.load_textures(fetch_from(&files)).unwrap();
        let gaussians = decode_range(&meta, &textures, 0, 2).unwrap();

        // rest coefficient j=0..2, channels interleaved from index 3
        for j in 0..coeffs {
            for ch in 0..SH_CHANNELS {
                let idx = (j + 1) * SH_CHANNELS + ch;
                let v = gaussians[0].spherical_harmonic.coefficients[idx];
                assert!((v - 0.75).abs() < 1e-6, "coefficient {idx}: {v}");
            }
        }
    }

    #[test]
    fn parses_bundle() {
        let (meta, files) = test_sog();

        let mut bytes = Vec::new();
        {
            let mut writer = zip::ZipWriter::new(std::io::Cursor::new(&mut bytes));
            let options: zip::write::SimpleFileOptions = Default::default();

            let meta_json = format!(
                r#"{{
                    "version": 2,
                    "count": {},
                    "means": {{ "mins": {:?}, "maxs": {:?}, "files": ["means_l.webp", "means_u.webp"] }},
                    "scales": {{ "codebook": {:?}, "files": ["scales.webp"] }},
                    "quats": {{ "files": ["quats.webp"] }},
                    "sh0": {{ "codebook": {:?}, "files": ["sh0.webp"] }}
                }}"#,
                meta.count, meta.means.mins, meta.means.maxs, meta.scales.codebook, meta.sh0.codebook,
            );

            use std::io::Write;
            writer.start_file("scene/meta.json", options).unwrap();
            writer.write_all(meta_json.as_bytes()).unwrap();
            for (name, data) in &files {
                writer.start_file(format!("scene/{name}"), options).unwrap();
                writer.write_all(data).unwrap();
            }
            writer.finish().unwrap();
        }

        let cloud = parse_sog_bundle(&bytes).unwrap();
        assert_eq!(cloud.len(), 32); // 2 gaussians padded to 32
    }
}
