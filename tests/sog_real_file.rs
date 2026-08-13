//! Validation against a real SOG bundle (e.g. LichtFeld Studio / splat-transform
//! output). Ignored by default; point `SOG_VALIDATION_PATH` at a `.sog` and run:
//!
//! ```sh
//! SOG_VALIDATION_PATH=/path/to/scene.sog cargo test --test sog_real_file -- --ignored --nocapture
//! ```

#![cfg(feature = "io_sog")]

use bevy_gaussian_splatting::io::sog::parse_sog_bundle;
use bevy_interleave::prelude::Planar;

#[test]
#[ignore = "requires SOG_VALIDATION_PATH pointing at a real .sog bundle"]
fn decodes_real_sog_bundle() {
    let path = std::env::var("SOG_VALIDATION_PATH")
        .expect("set SOG_VALIDATION_PATH to a .sog bundle");
    let bytes = std::fs::read(&path).expect("failed to read .sog");

    let start = std::time::Instant::now();
    let cloud = parse_sog_bundle(&bytes).expect("failed to decode .sog bundle");
    let elapsed = start.elapsed();

    let count = cloud.len();
    assert!(count > 0);
    assert_eq!(count % 32, 0, "cloud must be padded to a multiple of 32");

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut opacity_sum = 0.0f64;
    let mut opacity_min = f32::INFINITY;
    let mut opacity_max = f32::NEG_INFINITY;
    let mut scale_sum = [0.0f64; 3];
    let mut dc_sum = [0.0f64; 3];
    let mut max_quat_norm_err = 0.0f32;
    let mut real = 0usize;
    let mut rest_sum = 0.0f64;
    let mut rest_sq_sum = 0.0f64;
    let mut rest_n = 0usize;

    for gaussian in cloud.iter() {
        // skip padding (visibility 0, default-constructed)
        if gaussian.position_visibility.visibility == 0.0 {
            continue;
        }
        real += 1;

        let p = gaussian.position_visibility.position;
        for i in 0..3 {
            assert!(p[i].is_finite(), "non-finite position");
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }

        let o = gaussian.scale_opacity.opacity;
        assert!((0.0..=1.0).contains(&o), "opacity out of range: {o}");
        opacity_sum += o as f64;
        opacity_min = opacity_min.min(o);
        opacity_max = opacity_max.max(o);

        for i in 0..3 {
            let s = gaussian.scale_opacity.scale[i];
            assert!(s.is_finite() && s > 0.0, "bad scale: {s}");
            scale_sum[i] += s as f64;
        }

        let q = gaussian.rotation.rotation;
        let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        max_quat_norm_err = max_quat_norm_err.max((norm - 1.0).abs());

        for (i, dc) in dc_sum.iter_mut().enumerate() {
            *dc += gaussian.spherical_harmonic.coefficients[i] as f64;
        }

        for coefficient in &gaussian.spherical_harmonic.coefficients[3..] {
            rest_sum += *coefficient as f64;
            rest_sq_sum += (*coefficient as f64) * (*coefficient as f64);
            rest_n += 1;
        }
    }

    let n = real as f64;
    println!("decoded {count} rows ({real} real gaussians) in {elapsed:.2?}");
    println!("position bbox min: {min:?}");
    println!("position bbox max: {max:?}");
    println!(
        "opacity mean/min/max: {:.6} / {opacity_min:.6} / {opacity_max:.6}",
        opacity_sum / n
    );
    println!(
        "scale mean: [{:.6}, {:.6}, {:.6}]",
        scale_sum[0] / n,
        scale_sum[1] / n,
        scale_sum[2] / n
    );
    println!(
        "sh dc mean: [{:.6}, {:.6}, {:.6}]",
        dc_sum[0] / n,
        dc_sum[1] / n,
        dc_sum[2] / n
    );
    let rest_mean = rest_sum / rest_n as f64;
    let rest_std = (rest_sq_sum / rest_n as f64 - rest_mean * rest_mean).sqrt();
    println!("sh rest mean/std ({rest_n} values): {rest_mean:.6} / {rest_std:.6}");
    println!("max quaternion norm error: {max_quat_norm_err:.6}");

    assert!(max_quat_norm_err < 1e-3, "quaternions must be normalized");
}
