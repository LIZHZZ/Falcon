#pragma once

#include <cstdint>
#include <cstddef>

//ssize_t
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
#ifdef _WIN32
#include <basetsd.h> //SSIZE_T
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h> //Linux/macOS
#endif
#endif

//translated comment
#ifdef __cplusplus
extern "C" {
#endif

//translated comment
ssize_t elf_encode(double *in, ssize_t len, uint8_t **out, double error);
ssize_t elf_decode(uint8_t *in, ssize_t len, double *out, double error);
ssize_t elf_encode_32(float *in, ssize_t len, uint8_t **out, float error);
ssize_t elf_decode_32(uint8_t *in, ssize_t len, float *out, float error);

#ifdef __cplusplus
}
#endif
