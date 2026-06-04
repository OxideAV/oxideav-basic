//! Integration test for the `filter::tone` (Reinhard 2002 simple global)
//! primitive — exercises the public surface from outside the crate.
//!
//! Source: `docs/image/filter/tone-mapping-operators.md` §2.2.

use oxideav_basic::filter::{DisplayLuminance, SceneLuminance};

/// `scene → display → scene` reproduces the input to within f64 round-off
/// across a representative span of scene luminances. This is the
/// public-surface mirror of the same test inside `src/filter.rs`.
#[test]
fn reinhard_simple_scene_roundtrip_public_surface() {
    let probes: &[f64] = &[
        0.0, 1.0e-6, 0.018, // perceptual mid-grey
        0.18,  // photographic 18 % grey
        1.0,   // the curve's symmetric point (Ld == 0.5)
        4.0, 100.0, 1.0e5,
    ];
    for &l in probes {
        let scene = SceneLuminance::new(l).expect("non-negative finite probe");
        let display = scene.to_display();
        let back = display.to_scene().get();
        let denom = if l > 0.0 { l } else { 1.0 };
        let rel = (back - l).abs() / denom;
        assert!(
            rel < 1e-9,
            "round-trip drift at L={l}: got {back}, rel err {rel}",
        );
    }
}

/// The forward curve `Ld = L/(1+L)` matches §2.2's three reference values
/// (L=0 → 0, L=1 → 0.5, L=3 → 0.75) bit-exactly.
#[test]
fn reinhard_simple_known_values_public_surface() {
    assert_eq!(SceneLuminance::new(0.0).unwrap().to_display().get(), 0.0);
    assert_eq!(SceneLuminance::new(1.0).unwrap().to_display().get(), 0.5);
    assert_eq!(SceneLuminance::new(3.0).unwrap().to_display().get(), 0.75);
}

/// `DisplayLuminance::new` rejects 1.0 — that value is unreachable by the
/// forward curve and the inverse blows up.
#[test]
fn display_luminance_rejects_unity() {
    assert!(DisplayLuminance::new(1.0).is_none());
    assert!(DisplayLuminance::new(0.0).is_some());
    assert!(DisplayLuminance::new(0.999_999_999).is_some());
}
