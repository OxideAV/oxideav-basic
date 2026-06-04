//! Small image-filter primitives, grouped here while there are too few to
//! deserve their own crate.
//!
//! Currently exposes a single typed scalar tone-mapping operator: the
//! Reinhard 2002 simple global form `Ld = L / (1 + L)`. The math is a
//! published uncopyrightable fact (Reinhard et al., ACM TOG SIGGRAPH 2002),
//! transcribed in `docs/image/filter/tone-mapping-operators.md` §2.2. This
//! crate re-states the formula and pins it behind a typed wrapper so
//! callers can't accidentally mix scene and display luminance domains.
//!
//! The operator has a closed-form inverse on `Ld ∈ [0, 1)` (see
//! [`DisplayLuminance::to_scene`] below), which is what the round-trip
//! unit test exercises. There is no implicit clamping: the caller decides
//! whether to saturate `Ld` to a smaller range (e.g. for an 8-bit output).
//!
//! # Why a typed scalar
//!
//! The wider image-filter pipeline carries planar pixel data through other
//! crates. The point of this module is *not* to be a fast batch-luminance
//! tone-mapper — it is the operator's reference scalar form, suitable for
//! per-pixel use, LUT building, or unit-testing other compressors against
//! a known-good identity round-trip.

/// Scene (world) luminance: a non-negative scalar in arbitrary linear units.
///
/// "Scene" here means pre-tone-mapping luminance, before the simple Reinhard
/// curve `Ld = L / (1 + L)` has been applied. The struct does not enforce a
/// specific unit (cd/m², linear-light normalized, …) — that's the caller's
/// contract. It does enforce non-negativity and finiteness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneLuminance(f64);

impl SceneLuminance {
    /// Construct a scene luminance. Returns `None` for negative, NaN, or
    /// infinite inputs — none of those are valid linear-light luminance.
    pub fn new(l: f64) -> Option<Self> {
        if l.is_finite() && l >= 0.0 {
            Some(Self(l))
        } else {
            None
        }
    }

    /// Inner scalar.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// Apply the Reinhard 2002 simple global tone-reproduction curve:
    ///
    /// `Ld = L / (1 + L)`
    ///
    /// The result is always in the half-open range `[0, 1)` for any
    /// finite non-negative `L`, so the conversion is total.
    ///
    /// Source: `docs/image/filter/tone-mapping-operators.md` §2.2
    /// (the simple form, no white point, no key scaling).
    pub fn to_display(self) -> DisplayLuminance {
        let l = self.0;
        let ld = l / (1.0 + l);
        // ld is in [0, 1) by construction for L ∈ [0, ∞), so the
        // unchecked constructor is safe by maths.
        DisplayLuminance(ld)
    }
}

/// Display luminance after tone-mapping: a scalar in `[0, 1)`.
///
/// This is the output domain of [`SceneLuminance::to_display`]. The closed
/// half-open range comes from the asymptote of `L/(1+L)` at `L → ∞`: 1 is
/// approached but never reached, so any displayable pixel value lives
/// strictly below 1. The closed-form inverse below relies on that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayLuminance(f64);

impl DisplayLuminance {
    /// Construct a display luminance. Returns `None` for inputs outside
    /// `[0, 1)`, NaN, or non-finite values. `1.0` is rejected because it
    /// is unreachable by the forward curve (and the inverse blows up).
    pub fn new(ld: f64) -> Option<Self> {
        if ld.is_finite() && (0.0..1.0).contains(&ld) {
            Some(Self(ld))
        } else {
            None
        }
    }

    /// Inner scalar.
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// Inverse of [`SceneLuminance::to_display`]:
    ///
    /// `L = Ld / (1 − Ld)`
    ///
    /// Derived by solving `Ld = L/(1+L)` for `L`. Total on the
    /// half-open construction range `[0, 1)`. The result is non-negative
    /// and finite (it grows without bound as `Ld → 1`, but the
    /// constructor refuses `Ld == 1.0`, so we never hit infinity here).
    pub fn to_scene(self) -> SceneLuminance {
        let ld = self.0;
        // 1.0 - ld is strictly positive because the constructor rejects ld == 1.
        let l = ld / (1.0 - ld);
        SceneLuminance(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_constructor_rejects_invalid() {
        assert!(SceneLuminance::new(-0.1).is_none());
        assert!(SceneLuminance::new(f64::NAN).is_none());
        assert!(SceneLuminance::new(f64::INFINITY).is_none());
        assert!(SceneLuminance::new(0.0).is_some());
        assert!(SceneLuminance::new(1.0e6).is_some());
    }

    #[test]
    fn display_constructor_rejects_invalid() {
        assert!(DisplayLuminance::new(-0.01).is_none());
        assert!(DisplayLuminance::new(1.0).is_none());
        assert!(DisplayLuminance::new(1.001).is_none());
        assert!(DisplayLuminance::new(f64::NAN).is_none());
        assert!(DisplayLuminance::new(0.0).is_some());
        assert!(DisplayLuminance::new(0.99999).is_some());
    }

    #[test]
    fn known_curve_values() {
        // L = 0 → Ld = 0 (no light maps to no light).
        assert_eq!(SceneLuminance::new(0.0).unwrap().to_display().get(), 0.0);
        // L = 1 → Ld = 0.5 (the symmetric point of the curve).
        assert_eq!(SceneLuminance::new(1.0).unwrap().to_display().get(), 0.5);
        // L = 3 → Ld = 0.75; L = 9 → Ld = 0.9.
        assert!((SceneLuminance::new(3.0).unwrap().to_display().get() - 0.75).abs() < 1e-15);
        assert!((SceneLuminance::new(9.0).unwrap().to_display().get() - 0.9).abs() < 1e-15);
    }

    #[test]
    fn forward_output_is_strictly_below_one() {
        // Even at very large L the curve stays strictly < 1.
        let big = SceneLuminance::new(1.0e12).unwrap().to_display().get();
        assert!(big < 1.0, "Ld must remain strictly below 1, got {big}");
    }

    /// Round-trip: `scene → display → scene` reproduces the input over a
    /// representative range of scene luminances.
    #[test]
    fn scene_display_scene_roundtrip() {
        // Test points span dark, mid, and high-light regions. The upper
        // limit stays in a range where 1/(1+L) is representable with
        // many significant bits; well above L≈1e7 the round-trip starts
        // losing ULPs to the `1+L` cancellation and the tolerance below
        // would no longer be meaningful.
        let probes: &[f64] = &[
            0.0, 1.0e-9, 0.001, 0.018, 0.18, 1.0, 2.5, 10.0, 100.0, 1.0e4, 1.0e6,
        ];
        for &l in probes {
            let scene = SceneLuminance::new(l).unwrap();
            let back = scene.to_display().to_scene();
            // f64 has ~15-16 decimal digits of precision, but the
            // `1+L` step costs ULPs that scale with L. A relative
            // tolerance of 1e-9 is comfortably above the worst-case
            // round-off across the probe set and well below any
            // perceptual threshold.
            let denom = if l > 0.0 { l } else { 1.0 };
            let rel = (back.get() - l).abs() / denom;
            assert!(
                rel < 1e-9,
                "round-trip lost precision at L={l}: got {got}, rel err {rel}",
                got = back.get(),
            );
        }
    }

    /// Round-trip the other way too: `display → scene → display` over the
    /// open `[0, 1)` range reproduces the input.
    #[test]
    fn display_scene_display_roundtrip() {
        let probes: &[f64] = &[0.0, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9999];
        for &ld in probes {
            let display = DisplayLuminance::new(ld).unwrap();
            let back = display.to_scene().to_display();
            let denom = if ld > 0.0 { ld } else { 1.0 };
            let rel = (back.get() - ld).abs() / denom;
            assert!(
                rel < 1e-9,
                "round-trip lost precision at Ld={ld}: got {got}, rel err {rel}",
                got = back.get(),
            );
        }
    }
}
