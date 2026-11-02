//! CUTLASS half.h, bfloat16.h, float8.h bit encodings and rounding contracts.
use noodle_cute::numeric::{BF16, F8E4M3, F8E5M2, F16};

#[test]
fn storage_and_known_encodings() {
    assert_eq!((size_of::<F16>(), align_of::<F16>()), (2, 2));
    assert_eq!((size_of::<BF16>(), align_of::<BF16>()), (2, 2));
    assert_eq!((size_of::<F8E4M3>(), size_of::<F8E5M2>()), (1, 1));
    for (v, h, b, e4, e5) in [
        (0.0, 0, 0, 0, 0),
        (-0.0, 0x8000, 0x8000, 0x80, 0x80),
        (1.0, 0x3c00, 0x3f80, 0x38, 0x3c),
        (-2.0, 0xc000, 0xc000, 0xc0, 0xc0),
    ] {
        assert_eq!(F16::from_f32(v).to_bits(), h);
        assert_eq!(BF16::from_f32(v).to_bits(), b);
        assert_eq!(F8E4M3::from_f32(v).to_bits(), e4);
        assert_eq!(F8E5M2::from_f32(v).to_bits(), e5);
    }
    assert_eq!(F16::from_f32(65504.0).to_bits(), 0x7bff);
    assert_eq!(F8E4M3::from_f32(448.0).to_bits(), 0x7e);
    assert_eq!(F8E5M2::from_f32(57344.0).to_bits(), 0x7b);
}

#[test]
fn round_to_nearest_ties_to_even_and_underflow() {
    assert_eq!(F16::from_f32(1.0 + 1.0 / 2048.0).to_bits(), 0x3c00);
    assert_eq!(F16::from_f32(1.0 + 3.0 / 2048.0).to_bits(), 0x3c02);
    assert_eq!(BF16::from_f32(1.0 + 1.0 / 256.0).to_bits(), 0x3f80);
    assert_eq!(BF16::from_f32(1.0 + 3.0 / 256.0).to_bits(), 0x3f82);
    assert_eq!(F8E4M3::from_f32(1.0625).to_bits(), 0x38);
    assert_eq!(F8E4M3::from_f32(1.1875).to_bits(), 0x3a);
    assert_eq!(F8E5M2::from_f32(1.125).to_bits(), 0x3c);
    assert_eq!(F8E5M2::from_f32(1.375).to_bits(), 0x3e);
    assert_eq!(F16::from_f32(2.0f32.powi(-24)).to_bits(), 1);
    assert_eq!(F16::from_f32(2.0f32.powi(-25)).to_bits(), 0);
    assert_eq!(F16::from_f32(3.0 * 2.0f32.powi(-25)).to_bits(), 2);
    assert_eq!(F8E4M3::from_f32(1.0 / 1024.0).to_bits(), 0);
    assert_eq!(F8E4M3::from_f32(3.0 / 1024.0).to_bits(), 2);
    assert_eq!(F8E5M2::from_f32(1.0 / 131072.0).to_bits(), 0);
    assert_eq!(F8E5M2::from_f32(3.0 / 131072.0).to_bits(), 2);
    assert_eq!(BF16::from_f32(f32::from_bits(0x8000)).to_bits(), 0);
    assert_eq!(BF16::from_f32(f32::from_bits(0x18000)).to_bits(), 2);
}

#[test]
fn special_values_and_finite_saturation() {
    for sign in [1.0, -1.0] {
        assert_eq!(F16::from_f32(sign * 65520.0).to_f32(), sign * f32::INFINITY);
        assert_eq!(
            BF16::from_f32(sign * f32::INFINITY).to_f32(),
            sign * f32::INFINITY
        );
        assert_eq!(
            F8E4M3::from_f32(sign * f32::INFINITY).to_f32(),
            sign * 448.0
        );
        assert_eq!(
            F8E5M2::from_f32(sign * f32::INFINITY).to_f32(),
            sign * 57344.0
        );
        assert_eq!(F8E4M3::from_f32(sign * f32::MAX).to_f32(), sign * 448.0);
        assert_eq!(F8E5M2::from_f32(sign * f32::MAX).to_f32(), sign * 57344.0);
    }
    for bits in [0x7f800001, 0x7fc00000, 0xff800001, 0xffffffff] {
        let nan = f32::from_bits(bits);
        assert_eq!(F16::from_f32(nan).to_bits(), 0x7fff);
        assert_eq!(BF16::from_f32(nan).to_bits(), 0x7fff);
        assert_eq!(F8E4M3::from_f32(nan).to_bits(), 0x7f);
        assert_eq!(F8E5M2::from_f32(nan).to_bits(), 0x7f);
    }
    assert!(F8E4M3::from_bits(0xff).to_f32().is_nan());
    assert!(F8E5M2::from_bits(0xfc).to_f32().is_infinite());
    assert_eq!(F16::from_f32(-0.0), F16::default());
    assert_ne!(F16::from_f32(f32::NAN), F16::from_f32(f32::NAN));
}

#[test]
fn all_finite_encodings_widen_exactly_and_round_trip() {
    for bits in 0..=u16::MAX {
        let h = F16::from_bits(bits);
        if h.to_f32().is_finite() {
            assert_eq!(F16::from(h.to_f32()).to_bits(), bits);
        }
        let b = BF16::from_bits(bits);
        if b.to_f32().is_finite() {
            assert_eq!(BF16::from(b.to_f32()).to_bits(), bits);
        }
    }
    for bits in 0..=u8::MAX {
        let a = F8E4M3::from_bits(bits);
        if a.to_f32().is_finite() {
            assert_eq!(F8E4M3::from(a.to_f32()).to_bits(), bits);
        }
        let b = F8E5M2::from_bits(bits);
        if b.to_f32().is_finite() {
            assert_eq!(F8E5M2::from(b.to_f32()).to_bits(), bits);
        }
    }
}

#[test]
fn fp8_midpoints_follow_nearest_even_including_normal_boundary() {
    for code in 0..0x7eu8 {
        let low = F8E4M3::from_bits(code).to_f32();
        let high = F8E4M3::from_bits(code + 1).to_f32();
        let midpoint = (low + high) * 0.5;
        let expected = code + (code & 1);
        assert_eq!(F8E4M3::from_f32(midpoint).to_bits(), expected);
        assert_eq!(F8E4M3::from_f32(-midpoint).to_bits(), expected | 0x80);
    }
    for code in 0..0x7bu8 {
        let low = F8E5M2::from_bits(code).to_f32();
        let high = F8E5M2::from_bits(code + 1).to_f32();
        let midpoint = (low + high) * 0.5;
        let expected = code + (code & 1);
        assert_eq!(F8E5M2::from_f32(midpoint).to_bits(), expected);
        assert_eq!(F8E5M2::from_f32(-midpoint).to_bits(), expected | 0x80);
    }
    assert_eq!(F16::from_bits(1).to_f32(), 2.0f32.powi(-24));
    assert_eq!(F16::from_bits(0x400).to_f32(), 2.0f32.powi(-14));
    assert_eq!(BF16::from_bits(1).to_f32(), 2.0f64.powi(-133) as f32);
    assert_eq!(F8E4M3::from_bits(1).to_f32(), 2.0f32.powi(-9));
    assert_eq!(F8E5M2::from_bits(1).to_f32(), 2.0f32.powi(-16));
}
