// Host and build metadata used to stamp every benchmark result.
#ifndef PARALLEL_COMPUTE_SYSTEM_SYSTEM_INFO_HPP
#define PARALLEL_COMPUTE_SYSTEM_SYSTEM_INFO_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace parallel_compute::system {

struct CpuInfo {
  std::string model;             ///< e.g. "Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz"
  std::string vendor;
  std::string architecture;      ///< e.g. "x86_64", "aarch64"
  int logical_cores = 0;
  int physical_cores = 0;
  int threads_per_core = 0;
  std::size_t l1d_bytes = 0;
  std::size_t l2_bytes = 0;
  std::size_t l3_bytes = 0;
  std::size_t total_ram_bytes = 0;
  std::string supported_simd;
};

struct OsInfo {
  std::string system;
  std::string release;
  std::string machine;
};

struct RepoInfo {
  std::string revision;   ///< empty when not a git checkout
  bool dirty = false;
  std::string branch;
};

[[nodiscard]] CpuInfo query_cpu_info();
[[nodiscard]] OsInfo query_os_info();

/// Compiler id + version, e.g. "GNU 14.2.0".
[[nodiscard]] std::string compiler_description();

/// Dependency provenance, e.g. {"googletest", "fetched:v1.17.0"}.
[[nodiscard]] std::vector<std::string> dependency_versions();

/// "e1c0ff.. (dirty)" or "" when git is unavailable.
[[nodiscard]] RepoInfo query_repo_info();

/// Run `git` in the given directory (defaults to PCL_SOURCE_DIR baked in at
/// configure time).  Returns false when git is missing or the directory is
/// not a repository.
bool query_repo_info_at(const std::string& directory, RepoInfo& out);

}  // namespace parallel_compute::system

#endif  // PARALLEL_COMPUTE_SYSTEM_SYSTEM_INFO_HPP
