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


#ifdef __cplusplus
extern "C" {
#endif
// #include <sys/types.h>

ssize_t elf_star_encode(double *in, ssize_t len, uint8_t **out);
ssize_t elf_star_decode(uint8_t *in, ssize_t len, double *out);
ssize_t elf_star_encode_32(float *in, ssize_t len, uint8_t **out);
ssize_t elf_star_decode_32(uint8_t *in, ssize_t len, float *out);

#ifdef __cplusplus
}
#endif 
