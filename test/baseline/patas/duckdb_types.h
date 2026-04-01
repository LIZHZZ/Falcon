#pragma once

#include <cstdint>
#include <cstring>

//translated comment
typedef uint64_t idx_t;

//DuckDB API
#ifndef DUCKDB_API
#ifdef _WIN32
#define DUCKDB_API __declspec(dllimport)
#else
#define DUCKDB_API
#endif
#endif

//Debug
#ifdef DEBUG
#define D_ASSERT(condition) assert(condition)
#else
#define D_ASSERT(condition)
#endif

namespace alp_bench {

//Load -
template <class T>
inline T Load(const uint8_t* ptr) {
	T ret;
	memcpy(&ret, ptr, sizeof(T));
	return ret;
}

} // namespace alp_bench
