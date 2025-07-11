#pragma once
#include <cstdint>

namespace auv {
namespace math {

inline uint16_t to_float16(float value) {
  uint32_t f = *reinterpret_cast<uint32_t*>(&value);
  uint32_t sign = (f >> 31) & 0x1;
  int32_t exponent = ((f >> 23) & 0xFF) - 127;
  uint32_t mantissa = f & 0x7FFFFF;

  if ((f & 0x7FFFFFFF) == 0) {
    // Zero (positive or negative)
    return sign << 15;
  } else if ((f & 0x7F800000) == 0x7F800000) {
    // Inf or NaN
    uint16_t result = (sign << 15) | 0x7C00;
    if (mantissa != 0)
      result |= (mantissa >> 13);  // preserve some payload bits
    return result;
  }

  int32_t half_exp = exponent + 15;
  if (half_exp <= 0) {
    // Subnormal (or underflow to zero)
    if (half_exp < -10) return sign << 15;  // too small, becomes zero
    mantissa |= 1 << 23;                    // add implicit 1
    int32_t shift = 14 - half_exp;
    uint16_t submantissa = mantissa >> shift;
    return (sign << 15) | submantissa;
  } else if (half_exp >= 31) {
    // Overflow to infinity
    return (sign << 15) | 0x7C00;
  } else {
    return (sign << 15) | (half_exp << 10) | (mantissa >> 13);
  }
}

inline float from_float16(uint16_t value) {
  uint32_t sign = (value >> 15) & 0x1;
  uint32_t exponent = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x3FF;

  uint32_t f;

  if (exponent == 0) {
    if (mantissa == 0) {
      f = sign << 31;  // ±0
    } else {
      // Subnormal number
      exponent = 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x3FF;
      exponent = exponent + 127 - 15;
      mantissa <<= 13;
      f = (sign << 31) | (exponent << 23) | mantissa;
    }
  } else if (exponent == 0x1F) {
    // Inf or NaN
    f = (sign << 31) | 0x7F800000 | (mantissa << 13);
  } else {
    exponent = exponent + 127 - 15;
    mantissa <<= 13;
    f = (sign << 31) | (exponent << 23) | mantissa;
  }

  return *reinterpret_cast<float*>(&f);
}

}  // namespace math
}  // namespace auv
