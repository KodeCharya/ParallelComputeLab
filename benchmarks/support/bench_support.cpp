#include "bench_support.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

#include "parallel_compute/algorithms/matmul.hpp"
#include "parallel_compute/algorithms/reductions.hpp"
#include "parallel_compute/algorithms/vector_ops.hpp"
#include "parallel_compute/simd/cpu_features.hpp"
#include "parallel_compute/simd/simd_kernels.hpp"
#include "parallel_compute/system/build_config.hpp"

namespace pcl_bench {
namespace {

std::string utc_now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buffer;
}

std::vector<std::size_t> split_sizes(const std::string& value) {
  std::vector<std::size_t> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::strtoull(item.c_str(), nullptr, 10));
  }
  return out;
}

std::vector<int> split_ints(const std::string& value) {
  std::vector<int> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::atoi(item.c_str()));
  }
  return out;
}

std::string value_of(const std::string& arg, const std::string& flag) {
  const std::string prefix = flag + "=";
  if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
  return {};
}

std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string json_string(const std::string& in) { return "\"" + json_escape(in) + "\""; }

template <class T>
std::string json_number_array(const std::vector<T>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

}  // namespace

std::vector<std::size_t> default_vector_sizes() { return {1u << 20, 1u << 24}; }
std::vector<std::size_t> default_matrix_sizes() { return {128, 256, 512}; }

RunContext parse_context(int argc, char** argv) {
  RunContext ctx;
  ctx.vector_sizes = default_vector_sizes();
  ctx.matrix_sizes = default_matrix_sizes();
  ctx.thread_counts = {1, 2, 4};
  ctx.block_sizes = {32, 64};
  ctx.schedule_chunks = {1, 64};

  std::string cmd;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) cmd += " ";
    cmd += argv[i];

    const std::string arg = argv[i];
    if (!value_of(arg, "--pcl_sizes").empty()) {
      ctx.vector_sizes = split_sizes(value_of(arg, "--pcl_sizes"));
    }
    if (!value_of(arg, "--pcl_matrix_sizes").empty()) {
      ctx.matrix_sizes = split_sizes(value_of(arg, "--pcl_matrix_sizes"));
    }
    if (!value_of(arg, "--pcl_threads").empty()) {
      ctx.thread_counts = split_ints(value_of(arg, "--pcl_threads"));
    }
    if (!value_of(arg, "--pcl_blocks").empty()) {
      ctx.block_sizes = split_ints(value_of(arg, "--pcl_blocks"));
    }
    if (!value_of(arg, "--pcl_chunks").empty()) {
      ctx.schedule_chunks = split_ints(value_of(arg, "--pcl_chunks"));
    }
    if (!value_of(arg, "--pcl_seed").empty()) {
      ctx.seed = std::strtoull(value_of(arg, "--pcl_seed").c_str(), nullptr, 10);
    }
    if (!value_of(arg, "--pcl_label").empty()) {
      ctx.label = value_of(arg, "--pcl_label");
    }
    if (!value_of(arg, "--benchmark_out").empty()) {
      ctx.raw_output = value_of(arg, "--benchmark_out");
      if (!ctx.raw_output.empty() && ctx.raw_output.back() == ':') ctx.raw_output.pop_back();
    }
    if (!value_of(arg, "--benchmark_repetitions").empty()) {
      ctx.repetitions = value_of(arg, "--benchmark_repetitions");
    }
    if (!value_of(arg, "--benchmark_min_time").empty()) {
      ctx.min_time = value_of(arg, "--benchmark_min_time");
    }
    if (!value_of(arg, "--benchmark_enable_random_interleaving").empty()) {
      ctx.random_interleaving = true;
    }
  }
  ctx.command_line = cmd;
  if (ctx.thread_counts.empty()) ctx.thread_counts = {1};
  std::sort(ctx.thread_counts.begin(), ctx.thread_counts.end());
  ctx.thread_counts.erase(
      std::unique(ctx.thread_counts.begin(), ctx.thread_counts.end()), ctx.thread_counts.end());
  return ctx;
}

RunContext& context() {
  static RunContext ctx;
  return ctx;
}

const std::vector<std::string>& context_flag_names() {
  static const std::vector<std::string> names = {
      "--pcl_sizes=1048576,16777216   element counts for vector kernels",
      "--pcl_matrix_sizes=   comma separated matrix edge lengths (square)",
      "--pcl_threads=1,2,4   thread counts to sweep (also controls OpenMP teams)",
      "--pcl_blocks=32,64    GEMM cache-block sizes",
      "--pcl_chunks=1,64     scheduling chunk sizes for dynamic/guided",
      "--pcl_seed=1234       seed for the deterministic input generator",
      "--pcl_label=NAME      free-form tag copied into the result metadata",
      "--benchmark_out=FILE  raw JSON results (Google Benchmark)",
      "--benchmark_repetitions=N",
      "--benchmark_min_time=T",
  };
  return names;
}

void write_metadata(const RunContext& ctx, const std::string& path) {
  if (path.empty()) return;
  const auto cpu = parallel_compute::system::query_cpu_info();
  const auto os = parallel_compute::system::query_os_info();
  const auto repo = parallel_compute::system::query_repo_info();

  std::ofstream out(path);
  if (!out) {
    std::fprintf(stderr, "[pcl] could not write metadata to %s\n", path.c_str());
    return;
  }
  out << "{\n";
  out << "  \"schema_version\": " << json_string(PCL_SCHEMA_VERSION) << ",\n";
  out << "  \"pcl_version\": " << json_string(parallel_compute::build_info::version_string) << ",\n";
  out << "  \"generated_utc\": " << json_string(utc_now()) << ",\n";
  out << "  \"label\": " << json_string(ctx.label) << ",\n";
  out << "  \"raw_results\": " << json_string(ctx.raw_output) << ",\n";
  out << "  \"command_line\": " << json_string(ctx.command_line) << ",\n";
  out << "  \"seed\": " << ctx.seed << ",\n";

  out << "  \"config\": {\n";
  out << "    \"vector_sizes\": " << json_number_array(ctx.vector_sizes) << ",\n";
  out << "    \"matrix_sizes\": " << json_number_array(ctx.matrix_sizes) << ",\n";
  out << "    \"thread_counts\": " << json_number_array(ctx.thread_counts) << ",\n";
  out << "    \"block_sizes\": " << json_number_array(ctx.block_sizes) << ",\n";
  out << "    \"schedule_chunks\": " << json_number_array(ctx.schedule_chunks) << "\n";
  out << "  },\n";

  out << "  \"build\": {\n";
  out << "    \"compiler_id\": " << json_string(parallel_compute::build_info::compiler_id) << ",\n";
  out << "    \"compiler_version\": " << json_string(parallel_compute::build_info::compiler_version) << ",\n";
  out << "    \"build_type\": " << json_string(parallel_compute::build_info::build_type) << ",\n";
  out << "    \"cxx_standard\": " << json_string(parallel_compute::build_info::cxx_standard) << ",\n";
  out << "    \"cxx_flags\": " << json_string(parallel_compute::build_info::cxx_flags) << ",\n";
  out << "    \"simd_level\": " << json_string(parallel_compute::build_info::simd_level) << ",\n";
  out << "    \"native_arch\": " << json_string(parallel_compute::build_info::native_arch) << ",\n";
  out << "    \"openmp\": " << json_string(parallel_compute::build_info::openmp_status) << "\n";
  out << "  },\n";

  out << "  \"cpu\": {\n";
  out << "    \"model\": " << json_string(cpu.model) << ",\n";
  out << "    \"vendor\": " << json_string(cpu.vendor) << ",\n";
  out << "    \"architecture\": " << json_string(cpu.architecture) << ",\n";
  out << "    \"logical_cores\": " << cpu.logical_cores << ",\n";
  out << "    \"physical_cores\": " << cpu.physical_cores << ",\n";
  out << "    \"threads_per_core\": " << cpu.threads_per_core << ",\n";
  out << "    \"l1d_bytes\": " << cpu.l1d_bytes << ",\n";
  out << "    \"l2_bytes\": " << cpu.l2_bytes << ",\n";
  out << "    \"l3_bytes\": " << cpu.l3_bytes << ",\n";
  out << "    \"total_ram_bytes\": " << cpu.total_ram_bytes << ",\n";
  out << "    \"supported_simd\": " << json_string(cpu.supported_simd) << ",\n";
  out << "    \"active_simd_kernel\": " << json_string(parallel_compute::simd::kernels::active_name())
      << "\n";
  out << "  },\n";

  out << "  \"os\": {\n";
  out << "    \"system\": " << json_string(os.system) << ",\n";
  out << "    \"release\": " << json_string(os.release) << ",\n";
  out << "    \"machine\": " << json_string(os.machine) << "\n";
  out << "  },\n";

  out << "  \"repo\": {\n";
  out << "    \"revision\": " << json_string(repo.revision) << ",\n";
  out << "    \"branch\": " << json_string(repo.branch) << ",\n";
  out << "    \"dirty\": " << (repo.dirty ? "true" : "false") << "\n";
  out << "  },\n";

  out << "  \"dependencies\": [";
  const auto deps = parallel_compute::system::dependency_versions();
  for (std::size_t i = 0; i < deps.size(); ++i) {
    if (i != 0) out << ", ";
    out << json_string(deps[i]);
  }
  out << "],\n";

  out << "  \"measurement\": {\n";
  out << "    \"repetitions\": " << json_string(ctx.repetitions) << ",\n";
  out << "    \"min_time\": " << json_string(ctx.min_time) << ",\n";
  out << "    \"random_interleaving\": " << (ctx.random_interleaving ? "true" : "false") << ",\n";
  out << "    \"timed_region\": \"kernel only; allocation and input fill are excluded\",\n";
  out << "    \"validation\": \"each variant is checked against the scalar reference before timing\"\n";
  out << "  },\n";

  out << "  \"byte_counting\": {\n";
  out << "    \"elementwise_binary\": \"3*n*sizeof(T) (two reads + one write)\",\n";
  out << "    \"copy_scale\": \"2*n*sizeof(T) (one read + one write)\",\n";
  out << "    \"triad\": \"3*n*sizeof(T)\",\n";
  out << "    \"sequential_strided\": \"1*n*sizeof(T) logical (strided is a lower bound)\",\n";
  out << "    \"gemm\": \"model: 2*(m*n + k*n + m*k)*sizeof(T) bytes for 2*m*n*k flops\"\n";
  out << "  }\n";
  out << "}\n";
  out.close();
  std::fprintf(stderr, "[pcl] metadata written to %s\n", path.c_str());
}

VectorData::VectorData(std::size_t n) {
  a.resize(n);
  b.resize(n);
  y.resize(n);
  ref.resize(n);
  parallel_compute::random::fill_uniform<double>(a, 0xA5A5A5A5ULL, -1.0, 1.0);
  parallel_compute::random::fill_uniform<double>(b, 0x5A5A5A5AULL, -1.0, 1.0);
  parallel_compute::vector_ops::add<double>(a, b, ref, parallel_compute::Backend::reference);
}

MatrixData::MatrixData(std::size_t m_, std::size_t n_, std::size_t k_) : m(m_), n(n_), k(k_) {
  a.resize(m * k);
  b.resize(k * n);
  c.resize(m * n);
  ref.resize(m * n);
  parallel_compute::random::fill_uniform<double>(a, 0x1234567ULL, -1.0, 1.0);
  parallel_compute::random::fill_uniform<double>(b, 0x7654321ULL, -1.0, 1.0);
  parallel_compute::gemm::matmul_reference<double>(
      parallel_compute::MatrixView<const double>(a.data(), m, k),
      parallel_compute::MatrixView<const double>(b.data(), k, n),
      parallel_compute::MutableMatrixView<double>(ref.data(), m, n));
}

void set_common_counters(benchmark::State& state, std::size_t elements, std::size_t bytes,
                         int threads, bool validated, const char* schedule) {
  state.counters["elements"] = static_cast<double>(elements);
  state.counters["bytes_per_iter"] = static_cast<double>(bytes);
  state.counters["threads"] = threads;
  state.counters["validated"] = validated ? 1.0 : 0.0;
  state.counters["simd_lanes"] =
      static_cast<double>(parallel_compute::simd::kernels::lane_width());
  (void)schedule;  // schedule is carried in the run label, which is a string
}

void set_rate_counters(benchmark::State& state, std::size_t bytes_per_iteration,
                       std::size_t flops_per_iteration) {
  const auto iters = static_cast<std::size_t>(state.iterations());
  state.SetBytesProcessed(static_cast<std::int64_t>(bytes_per_iteration) *
                          static_cast<std::int64_t>(iters));
  if (flops_per_iteration != 0) {
    const double gflops = static_cast<double>(flops_per_iteration) *
                          static_cast<double>(iters) / 1.0e9;
    state.counters["gflops_per_sec"] = benchmark::Counter(gflops, benchmark::Counter::kIsRate);
  }
  state.counters["iterations"] = static_cast<double>(iters);
}

std::string make_label(const char* backend, int threads, const char* schedule, std::size_t chunk) {
  std::ostringstream os;
  os << "backend=" << backend << ",threads=" << threads << ",schedule=" << schedule
     << ",chunk=" << chunk;
  return os.str();
}

bool results_agree(const std::vector<double>& want, const std::vector<double>& got,
                   parallel_compute::Tolerance tol) {
  if (want.size() != got.size()) return false;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!parallel_compute::approx_equal<double>(want[i], got[i], tol)) return false;
  }
  return true;
}

}  // namespace pcl_bench
