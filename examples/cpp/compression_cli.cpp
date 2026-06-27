import std;
import fan;
import fan.compression;

namespace file = fan::io::file;

static void row(auto n, auto v, auto u) { fan::printf("{:<12}{:>16}{:>16}", n, v, u); }
static void row_float(auto n, f64_t v, auto u) { fan::printf("{:<12}{:>16.3f}{:>16}", n, v, u); }

enum class compression_level_e { fast, normal, max };

struct cli_options_t {
  compression_level_e level = compression_level_e::max;
  bool verbose = false;
  std::size_t thread_count = 0;
};

struct cli_args_t {
  std::string_view mode;
  std::string in, out;
  cli_options_t options;
};

struct input_stats_t {
  std::uint64_t size = 0, files = 0;
};

static fan::fcs::compress_params_t get_params(compression_level_e level) {
  switch (level) {
    case compression_level_e::fast: return fan::fcs::params_fast();
    case compression_level_e::normal: return fan::fcs::params_normal();
    case compression_level_e::max: return fan::fcs::params_max();
  }
  return fan::fcs::params_max();
}

static bool parse_size(std::string_view s, std::size_t& out) {
  auto r = std::from_chars(s.data(), s.data() + s.size(), out);
  return r.ec == std::errc{} && r.ptr == s.data() + s.size() && out != 0;
}

static std::optional<cli_args_t> parse_args(int argc, char** argv) {
  if (argc < 3) { return {}; }
  cli_args_t args; args.mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string_view a = argv[i];
    if (a == "--fast") { args.options.level = compression_level_e::fast; }
    else if (a == "--normal") { args.options.level = compression_level_e::normal; }
    else if (a == "--max") { args.options.level = compression_level_e::max; }
    else if (a == "--verbose") { args.options.verbose = true; }
    else if (a == "-j" && ++i < argc) {
      if (!parse_size(argv[i], args.options.thread_count)) { return {}; }
    }
    else if (a.starts_with("-j")) {
      if (!parse_size(a.substr(2), args.options.thread_count)) { return {}; }
    }
    else if (a.starts_with("-")) { return {}; }
    else if (args.in.empty()) { args.in = a; }
    else if (args.out.empty()) { args.out = a; }
    else { return {}; }
  }
  return args.in.empty() ? std::optional<cli_args_t>{} : args;
}

static bool ask_overwrite(const std::string& path) {
  if (!std::filesystem::exists(path)) { return true; }
  fan::print(path, "exists. overwrite? [y/N]");
  std::string s; std::getline(std::cin, s);
  return !s.empty() && (s[0] == 'y' || s[0] == 'Y');
}

static input_stats_t get_input_stats(const std::filesystem::path& in) {
  input_stats_t stats; std::error_code ec;
  if (std::filesystem::is_directory(in, ec)) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(in)) {
      if (!entry.is_regular_file()) { continue; }
      stats.size += entry.file_size(); ++stats.files;
    }
  } else { stats.size = std::filesystem::file_size(in); stats.files = 1; }
  return stats;
}

static bool cmd_compress(const std::string& in, std::string out, const cli_options_t& options) {
  if (out.empty()) {
    std::string_view s = in; while (s.ends_with('/') || s.ends_with('\\')) { s.remove_suffix(1); }
    out = std::string(s) + ".fcs";
  }
  if (!ask_overwrite(out)) { return true; }
  auto stats = get_input_stats(in);
  fan::print("compressing", in, "->", out); fan::time::timer t;

  {
    fan::progress_t prog;
    std::jthread monitor([&](std::stop_token st) {
      while (!st.stop_requested()) {
        fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
      fan::print("");
    });
    if (!fan::fcs::compress_path_to_file(in, out, get_params(options.level), &prog, options.verbose, options.thread_count)) {
      fan::print("write failed", out); return false;
    }
  }
  f64_t ms = t.millis(), cs = f64_t(std::filesystem::file_size(out));
  row("files", stats.files, ""); row("original", stats.size, "bytes"); row("fcs", cs, "bytes");
  row_float("ratio", stats.size ? cs / stats.size * 100.0 : 0, "%");
  row_float("saved", stats.size ? (1.0 - cs / stats.size) * 100.0 : 0, "%");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(stats.size, ms), "MiB/s");
  fan::print("wrote", out); return true;
}

static bool cmd_decompress(const std::string& in, std::string out_dir, const cli_options_t&) {
  bool default_out = out_dir.empty();
  if (default_out) { out_dir = in.ends_with(".fcs") ? in.substr(0, in.size() - 4) : in + "_ext"; }
  fan::print("decompressing", in, "->", out_dir); fan::time::timer t;

  {
    fan::progress_t prog;
    std::jthread monitor([&](std::stop_token st) {
      while (!st.stop_requested()) {
        fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
      fan::print("");
    });
    if (!fan::fcs::decompress_file_to_dir(in, out_dir, default_out, &prog)) {
      fan::print("read failed", in); return false;
    }
  }
  row_float("time", t.millis(), "ms");
  fan::print("extracted to", out_dir); return true;
}

static int usage(std::string_view exe) {
  fan::print("usage:");
  fan::printf("  {} c <input_file_or_dir> [output.fcs] [--fast|--normal|--max] [--verbose] [-jN]", exe);
  fan::printf("  {} d <input.fcs> [output_dir]", exe);
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
    static const cmd_t commands[] = { {"compress", "c", cmd_compress}, {"decompress", "d", cmd_decompress} };
    for (const auto& c : commands) { if (args->mode == c.name || args->mode == c.alias) { return c.exec(args->in, args->out, args->options) ? 0 : 1; } }
    return usage(argv[0]);
  } catch (const std::exception& e) {
    fan::print("error:", e.what()); return 1;
  } catch (...) {
    fan::print("unknown error"); return 1;
  }
}