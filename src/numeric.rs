//! Portable low-precision storage and numeric conversions.
//!
//! Conversions round to nearest, ties to even, following CUTLASS `half_t`,
//! `bfloat16_t`, `float_e4m3_t`, and `float_e5m2_t`. FP8 conversions saturate
//! overflow and infinity to the largest finite value (`satfinite`). NaNs are
//! canonicalized on narrowing. These types issue no architecture instructions.
//!
//! ```
//! use noodle_cute::numeric::{F16, BF16, F8E4M3, F8E5M2};
//! assert_eq!(F16::from_f32(1.0).to_bits(), 0x3c00);
//! assert_eq!(BF16::from_f32(1.0).to_bits(), 0x3f80);
//! assert_eq!(F8E4M3::from_f32(500.0).to_f32(), 448.0);
//! assert_eq!(F8E5M2::from_f32(f32::INFINITY).to_f32(), 57344.0);
//! ```

const fn round_shift(value: u32, shift: u32) -> u32 {
    if shift >= 32 {
        return 0;
    }
    let retained = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    retained + (remainder > halfway || (remainder == halfway && retained & 1 != 0)) as u32
}

const fn encode(value: f32, mantissa: u32, bias: i32, max: u32, saturate: bool) -> u32 {
    let bits = value.to_bits();
    let sign = (bits >> 31) << (if mantissa == 10 { 15 } else { 7 });
    let exponent = ((bits >> 23) & 255) as i32;
    let fraction = bits & 0x7fffff;
    if exponent == 255 && fraction != 0 {
        return if mantissa == 10 { 0x7fff } else { 0x7f };
    }
    let overflow = if saturate { max } else { 0x7c00 };
    if exponent == 255 {
        return sign | overflow;
    }
    let target_exp = exponent - 127 + bias;
    let magnitude = if target_exp <= 0 {
        round_shift(
            fraction | if exponent == 0 { 0 } else { 1 << 23 },
            (24 - mantissa) + (-target_exp) as u32,
        )
    } else if target_exp > (max >> mantissa) as i32 {
        return sign | overflow;
    } else {
        ((target_exp as u32) << mantissa) + round_shift(fraction, 23 - mantissa)
    };
    sign | if magnitude > max { overflow } else { magnitude }
}

const fn decode(bits: u32, mantissa: u32, bias: i32, exp_bits: u32, finite: bool) -> f32 {
    let sign = (bits >> (mantissa + exp_bits)) << 31;
    let exp_mask = (1 << exp_bits) - 1;
    let exponent = (bits >> mantissa) & exp_mask;
    let mut fraction = bits & ((1 << mantissa) - 1);
    if exponent == exp_mask && (!finite || fraction == (1 << mantissa) - 1) {
        return f32::from_bits(if fraction == 0 && !finite {
            sign | 0x7f800000
        } else {
            0x7fffffff
        });
    }
    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        let mut exp = 1 - bias + 127;
        while fraction & (1 << mantissa) == 0 {
            fraction <<= 1;
            exp -= 1;
        }
        return f32::from_bits(
            sign | ((exp as u32) << 23) | ((fraction & ((1 << mantissa) - 1)) << (23 - mantissa)),
        );
    }
    f32::from_bits(
        sign | (((exponent as i32 - bias + 127) as u32) << 23) | (fraction << (23 - mantissa)),
    )
}

macro_rules! float_type {
    ($name:ident, $storage:ty, $description:literal, $encode:expr, $decode:expr) => {
        #[doc = $description]
        #[derive(Clone, Copy, Default, Debug)]
        #[repr(transparent)]
        pub struct $name($storage);

        impl $name {
            /// Constructs from the exact storage bits, including NaN payloads.
            pub const fn from_bits(bits: $storage) -> Self {
                Self(bits)
            }
            /// Returns the exact storage bits.
            pub const fn to_bits(self) -> $storage {
                self.0
            }
            /// Rounds an FP32 value to this format, ties to even.
            pub const fn from_f32(value: f32) -> Self {
                Self(($encode)(value) as $storage)
            }
            /// Widens to FP32 without rounding finite values.
            pub const fn to_f32(self) -> f32 {
                ($decode)(self.0 as u32)
            }
        }
        impl From<f32> for $name {
            fn from(value: f32) -> Self {
                Self::from_f32(value)
            }
        }
        impl From<$name> for f32 {
            fn from(value: $name) -> Self {
                value.to_f32()
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.to_f32() == other.to_f32()
            }
        }
        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.to_f32().partial_cmp(&other.to_f32())
            }
        }
    };
}

const fn encode_f16(x: f32) -> u32 {
    encode(x, 10, 15, 0x7bff, false)
}
const fn decode_f16(x: u32) -> f32 {
    decode(x, 10, 15, 5, false)
}
const fn encode_bf16(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x7fffffff > 0x7f800000 {
        0x7fff
    } else if bits & 0x7f800000 == 0x7f800000 {
        bits >> 16
    } else {
        (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    }
}
const fn decode_bf16(x: u32) -> f32 {
    f32::from_bits(x << 16)
}
const fn encode_e4m3(x: f32) -> u32 {
    encode(x, 3, 7, 0x7e, true)
}
const fn decode_e4m3(x: u32) -> f32 {
    decode(x, 3, 7, 4, true)
}
const fn encode_e5m2(x: f32) -> u32 {
    encode(x, 2, 15, 0x7b, true)
}
const fn decode_e5m2(x: u32) -> f32 {
    decode(x, 2, 15, 5, false)
}

float_type!(
    F16,
    u16,
    "IEEE binary16 (FP16), with infinity and subnormal support.",
    encode_f16,
    decode_f16
);
float_type!(
    BF16,
    u16,
    "BF16, with the FP32 exponent range and seven fraction bits.",
    encode_bf16,
    decode_bf16
);
float_type!(
    F8E4M3,
    u8,
    "FP8 E4M3, bias 7, maximum 448; 0x7f/0xff encode NaN.",
    encode_e4m3,
    decode_e4m3
);
float_type!(
    F8E5M2,
    u8,
    "FP8 E5M2, bias 15, maximum 57344; widening preserves infinity.",
    encode_e5m2,
    decode_e5m2
);
