import std;
import fan;

namespace file = fan::io::file;

static void row(auto n, auto v, auto u) { fan::printf("{:<12}{:>16}{:>16}", n, v, u); }
static void row_float(auto n, f64_t v, auto u) { fan::printf("{:<12}{:>16.3f}{:>16}", n, v, u); }

enum class compression_level_e { fast, normal, high, max };

struct cli_options_t {
  compression_level_e level = compression_level_e::max;
  bool verbose = false;
  bool yes = false;
  std::size_t thread_count = 0;
  std::size_t chunk_mib = 0;
};

struct cli_args_t {
  std::string_view mode;
  std::string in, out;
  cli_options_t options;
};

struct input_stats_t {
  std::uint64_t size = 0, files = 0;
};

struct progress_monitor_t {
  progress_monitor_t() {
    monitor = std::jthread([this](std::stop_token st) {
      while (!st.stop_requested()) {
        auto total = prog.total.load(std::memory_order_relaxed);
        if (total != 0) {
          fan::print_progress(prog.done.load(std::memory_order_relaxed), total);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      auto total = prog.total.load(std::memory_order_relaxed);
      if (total != 0) {
        fan::print_progress(total, total);
      }
      fan::print("");
    });
  }
  fan::progress_t prog;
  std::jthread monitor;
};

static fan::fcs::compress_params_t get_params(const cli_options_t& options) {
  fan::fcs::compress_params_t params;
  switch (options.level) {
    case compression_level_e::fast: params = fan::fcs::params_fast(); break;
    case compression_level_e::normal: params = fan::fcs::params_normal(); break;
    case compression_level_e::high: params = fan::fcs::params_high(); break;
    default: params = fan::fcs::params_max(); break;
  }
  if (options.chunk_mib != 0) {
    constexpr std::size_t max_mib = std::numeric_limits<std::uint32_t>::max() >> 20;
    params.chunk_size = std::min(options.chunk_mib, max_mib) << 20;
  }
  return params;
}

static bool parse_size(std::string_view s, std::size_t& out) {
  auto r = std::from_chars(s.data(), s.data() + s.size(), out);
  return r.ec == std::errc {} && r.ptr == s.data() + s.size() && out != 0;
}

static bool parse_option_value(std::string_view a, std::string_view name, int& i, int argc, char** argv, std::string_view& out) {
  if (a == name) {
    out = ++i < argc ? std::string_view(argv[i]) : std::string_view();
    return true;
  }
  if (a.starts_with(name) && a.size() > name.size() && a[name.size()] == '=') {
    out = a.substr(name.size() + 1);
    return true;
  }
  return false;
}

static std::optional<cli_args_t> parse_args(int argc, char** argv) {
  if (argc < 3) { return {}; }
  cli_args_t args;
  args.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string_view a = argv[i];
    std::string_view v;
    if (a == "--fast") { args.options.level = compression_level_e::fast; }
    else if (a == "--normal") { args.options.level = compression_level_e::normal; }
    else if (a == "--high") { args.options.level = compression_level_e::high; }
    else if (a == "--max") { args.options.level = compression_level_e::max; }
    else if (a == "--verbose") { args.options.verbose = true; }
    else if (a == "-y" || a == "--y") { args.options.yes = true; }
    else if (parse_option_value(a, "--chunk-mib", i, argc, argv, v) || parse_option_value(a, "--dict-mib", i, argc, argv, v)) {
      if (v.empty() || !parse_size(v, args.options.chunk_mib)) { return {}; }
    }
    else if (a.starts_with("-j")) {
      v = a.size() == 2 ? (++i < argc ? argv[i] : "") : a.substr(2);
      if (v.empty() || !parse_size(v, args.options.thread_count)) { return {}; }
    }
    else if (a.starts_with("-")) { return {}; }
    else if (args.in.empty()) { args.in = a; }
    else if (args.out.empty()) { args.out = a; }
    else { return {}; }
  }
  return args.in.empty() ? std::optional<cli_args_t>{} : args;
}

static input_stats_t get_input_stats(const std::filesystem::path& in) {
  input_stats_t stats;
  std::error_code ec;
  if (std::filesystem::is_directory(in, ec)) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(in)) {
      if (!entry.is_regular_file()) { continue; }
      stats.size += entry.file_size();
      ++stats.files;
    }
  }
  else {
    stats.size = std::filesystem::file_size(in);
    stats.files = 1;
  }
  return stats;
}

static bool cmd_compress(const std::string& in, std::string out, const cli_options_t& options) {
  if (out.empty()) {
    std::string_view s = in;
    while (s.ends_with('/') || s.ends_with('\\')) { s.remove_suffix(1); }
    out = std::string(s) + ".fcs";
  }
  if (!options.yes && std::filesystem::exists(out) && !fan::io::ask_override(out)) { return true; }
  auto stats = get_input_stats(in);
  fan::print("compressing", in, "->", out);
  fan::time::timer t;

  {
    progress_monitor_t pm;
    if (!fan::fcs::compress_path_to_file(in, out, get_params(options), &pm.prog, options.verbose, options.thread_count)) {
      fan::print("write failed", out); return false;
    }
  }
  f64_t ms = t.millis(), cs = f64_t(std::filesystem::file_size(out));
  row("files", stats.files, "");
  row("original", stats.size, "bytes");
  row("fcs", cs, "bytes");
  row_float("ratio", stats.size ? cs / stats.size * 100.0 : 0, "%");
  row_float("saved", stats.size ? (1.0 - cs / stats.size) * 100.0 : 0, "%");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(stats.size, ms), "MiB/s");
  fan::print("wrote", out); return true;
}

static bool cmd_decompress(const std::string& in, std::string out_dir, const cli_options_t&) {
  bool default_out = out_dir.empty();
  if (default_out) { out_dir = in.ends_with(".fcs") ? in.substr(0, in.size() - 4) : in + "_ext"; }
  fan::print("decompressing", in, "->", out_dir);
  fan::time::timer t;

  {
    progress_monitor_t pm;
    if (!fan::fcs::decompress_file_to_dir(in, out_dir, default_out, &pm.prog)) {
      fan::print("read failed", in); return false;
    }
  }
  f64_t ms = t.millis();
  auto stats = get_input_stats(out_dir);
  row("files", stats.files, "");
  row("output", stats.size, "bytes");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(stats.size, ms), "MiB/s");
  fan::print("extracted to", out_dir); return true;
}

static int usage(std::string_view exe) {
  fan::printf("usage:\n  {0} c <input_file_or_dir> [output.fcs] [--fast|--normal|--high|--max] [--verbose] [-y|--y] [-jN] [--chunk-mib N]\n  {0} d <input.fcs> [output_dir]\n", exe);
  return 1;
}

struct cmd_t {
  std::string_view name, alias;
  bool (*exec)(const std::string&, std::string, const cli_options_t&);
};

int main(int argc, char** argv) {
  try {
    auto args = parse_args(argc, argv);
    if (!args) { return usage(argv[0]); }
    static const cmd_t commands[] = {{"compress", "c", cmd_compress}, {"decompress", "d", cmd_decompress}};
    for (const auto& c : commands) {
      if (args->mode == c.name || args->mode == c.alias) { return c.exec(args->in, args->out, args->options) ? 0 : 1; }
    }
    return usage(argv[0]);
  }
  catch (const std::exception& e) {
    fan::print("error:", e.what()); return 1;
  }
  catch (...) {
    fan::print("unknown error"); return 1;
  }
}