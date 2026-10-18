// Host and toolchain metadata.
//
// Everything here is read from the running system at process start; nothing is
// hard coded, so the metadata attached to a benchmark result always describes
// the machine that actually produced it.
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "parallel_compute/common/types.hpp"
#include "parallel_compute/simd/cpu_features.hpp"
#include "parallel_compute/system/system_info.hpp"

#if defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

#include "parallel_compute/system/build_config.hpp"

namespace parallel_compute::system {
namespace {

std::string trim(std::string s) {
  const auto not_space = [](unsigned char c) { return c != ' ' && c != '\t' && c != '\n' && c != '\r'; };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

/// Contents of a one-line sysfs file (e.g. "32K").
std::string read_value_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) return {};
  std::string value;
  std::getline(in, value);
  return trim(value);
}

std::string read_first_line(const std::string& path, const std::string& key) {
  std::ifstream in(path);
  if (!in) return {};
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(key, 0) == 0) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) continue;
      return trim(line.substr(colon + 1));
    }
  }
  return {};
}

std::size_t parse_cache_size(const std::string& s) {
  if (s.empty()) return 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(s.c_str(), &end, 10);
  if (end == s.c_str()) return 0;
  std::string unit = end;
  unit = trim(unit);
  if (!unit.empty() && (unit[0] == 'K' || unit[0] == 'k')) value *= 1024ULL;
  if (!unit.empty() && (unit[0] == 'M' || unit[0] == 'm')) value *= 1024ULL * 1024ULL;
  return static_cast<std::size_t>(value);
}

std::size_t detect_cache(const std::string& level, const std::string& type) {
  for (int index = 0; index < 10; ++index) {
    const std::string base =
        "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
    std::ifstream probe(base + "size");
    if (!probe) continue;
    const std::string lvl = read_value_file(base + "level");
    const std::string typ = read_value_file(base + "type");
    if (lvl == level && typ == type) {
      return parse_cache_size(read_value_file(base + "size"));
    }
  }
  return 0;
}

/// "7615516 kB" -> bytes
std::size_t parse_mem_total(const std::string& s) {
  if (s.empty()) return 0;
  char* end = nullptr;
  const unsigned long long kb = std::strtoull(s.c_str(), &end, 10);
  if (end == s.c_str()) return 0;
  return static_cast<std::size_t>(kb) * 1024ULL;
}

}  // namespace

CpuInfo query_cpu_info() {
  CpuInfo info;
  info.model = read_first_line("/proc/cpuinfo", "model name");
  if (info.model.empty()) info.model = "unknown";
  info.vendor = read_first_line("/proc/cpuinfo", "vendor_id");
  if (info.vendor.empty()) info.vendor = "unknown";

#if defined(__x86_64__) || defined(__i386__)
  info.architecture = "x86_64";
#elif defined(__aarch64__)
  info.architecture = "aarch64";
#else
  info.architecture = build_info::system_processor;
#endif

  const std::string processors = read_first_line("/proc/cpuinfo", "processor");
  (void)processors;
  long nprocs = ::sysconf(_SC_NPROCESSORS_ONLN);
  info.logical_cores = nprocs > 0 ? static_cast<int>(nprocs) : 1;

  const std::string cores = read_first_line("/proc/cpuinfo", "cpu cores");
  const std::string siblings = read_first_line("/proc/cpuinfo", "siblings");
  if (!cores.empty()) {
    info.physical_cores = std::atoi(cores.c_str());
  }
  if (info.physical_cores <= 0) info.physical_cores = info.logical_cores;
  info.threads_per_core =
      info.physical_cores > 0 ? info.logical_cores / info.physical_cores : 1;
  if (info.threads_per_core < 1) info.threads_per_core = 1;
  (void)siblings;

  info.l1d_bytes = detect_cache("1", "Data");
  info.l2_bytes = detect_cache("2", "Unified");
  if (info.l2_bytes == 0) info.l2_bytes = detect_cache("2", "Data");
  info.l3_bytes = detect_cache("3", "Unified");

  info.total_ram_bytes = parse_mem_total(read_first_line("/proc/meminfo", "MemTotal"));
  info.supported_simd = simd::describe_supported_simd();
  return info;
}

OsInfo query_os_info() {
  OsInfo os;
#if defined(__linux__)
  utsname u{};
  if (uname(&u) == 0) {
    os.system = u.sysname;
    os.release = u.release;
    os.machine = u.machine;
  }
#endif
  if (os.system.empty()) os.system = "unknown";
  if (os.release.empty()) os.release = "unknown";
  if (os.machine.empty()) os.machine = "unknown";
  return os;
}

std::string compiler_description() {
  return std::string(build_info::compiler_id) + " " + build_info::compiler_version;
}

std::vector<std::string> dependency_versions() {
  return {
      std::string("googletest=") + build_info::googletest_source,
      std::string("google_benchmark=") + build_info::benchmark_source,
      std::string("openmp=") + build_info::openmp_status,
      std::string("blas=") + build_info::blas_status,
  };
}

bool query_repo_info_at(const std::string& directory, RepoInfo& out) {
  std::string cmd = "git -C \"" + directory + "\" rev-parse --short HEAD 2>/dev/null";
  std::string rev;
  if (std::FILE* pipe = ::popen(cmd.c_str(), "r")) {
    char buffer[256];
    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) rev = trim(buffer);
    ::pclose(pipe);
  }
  if (rev.empty()) return false;

  out.revision = rev;
  std::string dirty;
  cmd = "git -C \"" + directory + "\" status --porcelain 2>/dev/null";
  if (std::FILE* pipe = ::popen(cmd.c_str(), "r")) {
    char buffer[256];
    const bool any = std::fgets(buffer, sizeof(buffer), pipe) != nullptr;
    ::pclose(pipe);
    out.dirty = any;
  }
  std::string branch;
  cmd = "git -C \"" + directory + "\" rev-parse --abbrev-ref HEAD 2>/dev/null";
  if (std::FILE* pipe = ::popen(cmd.c_str(), "r")) {
    char buffer[256];
    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) branch = trim(buffer);
    ::pclose(pipe);
  }
  out.branch = branch;
  return true;
}

RepoInfo query_repo_info() {
  RepoInfo info;
#ifdef PCL_SOURCE_DIR
  query_repo_info_at(PCL_SOURCE_DIR, info);
#endif
  return info;
}

}  // namespace parallel_compute::system
