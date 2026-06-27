import std;
import fan;

static void row(auto n, auto v, auto u) { fan::printf("{:<12}{:>16}{:>16}", n, v, u); }
static void row_float(auto n, f64_t v, auto u) { fan::printf("{:<12}{:>16.3f}{:>16}", n, v, u); }

enum class compression_level_e {
  fast,
  normal,
  max
};

struct cli_options_t {
  compression_level_e level = compression_level_e::max;
  bool verbose = false;
};

struct cli_args_t {
  std::string_view mode;
  std::string in;
  std::string out;
  cli_options_t options;
};

static fan::fcs::compress_params_t get_params(compression_level_e level) {
  switch (level) {
    case compression_level_e::fast: return fan::fcs::params_fast();
    case compression_level_e::normal: return fan::fcs::params_normal();
    case compression_level_e::max: return fan::fcs::params_max();
  }
  return fan::fcs::params_max();
}

static std::optional<cli_args_t> parse_args(int argc, char** argv) {
  if (argc < 3) {
    return {};
  }

  cli_args_t args;
  args.mode = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string_view a = argv[i];

    if (a == "--fast") {
      args.options.level = compression_level_e::fast;
    }
    else if (a == "--normal") {
      args.options.level = compression_level_e::normal;
    }
    else if (a == "--max") {
      args.options.level = compression_level_e::max;
    }
    else if (a == "--verbose") {
      args.options.verbose = true;
    }
    else if (a.starts_with("--")) {
      fan::print("unknown option", a);
      return {};
    }
    else if (args.in.empty()) {
      args.in = a;
    }
    else if (args.out.empty()) {
      args.out = a;
    }
    else {
      fan::print("too many arguments", a);
      return {};
    }
  }

  if (args.in.empty()) {
    return {};
  }

  return args;
}

static bool ask_overwrite(const std::string& path) {
  if (!std::filesystem::exists(path)) { return true; }
  fan::print(path, "exists. overwrite? [y/N]");
  std::string s;
  std::getline(std::cin, s);
  return !s.empty() && (s[0] == 'y' || s[0] == 'Y');
}

static void print_progress_until_done(fan::fcs::progress_t& prog, std::stop_token st) {
  while (!st.stop_requested()) {
    fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
    std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }
  fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
  fan::print("");
}

static bool cmd_compress(const std::string& in, std::string out, const cli_options_t& options) {
  if (out.empty()) {
    std::string_view s = in;
    while (s.ends_with('/') || s.ends_with('\\')) { s.remove_suffix(1); }
    out = std::string(s) + ".fcs";
  }
  if (!ask_overwrite(out)) { return true; }

  fan::print("compressing", in, "->", out);
  fan::time::timer t;
  fan::fcs::progress_t prog;

  {
    std::jthread monitor([&](std::stop_token st) { print_progress_until_done(prog, st); });
    if (!fan::fcs::compress_path_to_file(in, out, get_params(options.level), &prog, options.verbose)) {
      fan::print("write failed", out);
      return false;
    }
  }

  f64_t ms = t.millis();
  std::error_code ec;
  std::uint64_t original = prog.total.load(std::memory_order_relaxed);
  std::uint64_t compressed = std::filesystem::file_size(out, ec);

  row("original", original, "bytes");
  row("fcs", ec ? 0 : compressed, "bytes");
  row_float("ratio", !ec && original ? f64_t(compressed) / original * 100.0 : 0, "%");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(original, ms), "MiB/s");
  fan::print("wrote", out);
  return true;
}

static bool cmd_decompress(const std::string& in, std::string out_dir, const cli_options_t&) {
  if (out_dir.empty()) {
    out_dir = in.ends_with(".fcs") ? in.substr(0, in.size() - 4) : in + "_ext";
  }

  fan::print("decompressing", in, "->", out_dir);
  fan::time::timer t;
  fan::fcs::progress_t prog;

  {
    std::jthread monitor([&](std::stop_token st) { print_progress_until_done(prog, st); });
    if (!fan::fcs::decompress_file_to_dir(in, out_dir, &prog)) {
      fan::print("read failed", in);
      return false;
    }
  }

  f64_t ms = t.millis();
  std::uint64_t extracted = prog.total.load(std::memory_order_relaxed);

  row("extracted", extracted, "bytes");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(extracted, ms), "MiB/s");
  fan::print("extracted to", out_dir);
  return true;
}

static int usage(std::string_view exe) {
  fan::print("usage:");
  fan::printf("  {} c <input_file_or_dir> [output.fcs] [--fast|--normal|--max] [--verbose]", exe);
  fan::printf("  {} d <input.fcs> [output_dir]", exe);
  return 1;
}

struct cmd_t {
  std::string_view name;
  std::string_view alias;
  bool (*exec)(const std::string&, std::string, const cli_options_t&);
};

int main(int argc, char** argv) {
  try {
    auto args = parse_args(argc, argv);
    if (!args) {
      return usage(argv[0]);
    }

    static const cmd_t commands[] = {
      {"compress", "c", cmd_compress},
      {"decompress", "d", cmd_decompress}
    };

    for (const auto& c : commands) {
      if (args->mode == c.name || args->mode == c.alias) {
        return c.exec(args->in, args->out, args->options) ? 0 : 1;
      }
    }

    return usage(argv[0]);
  }
  catch (const std::exception& e) {
    fan::print("error:", e.what());
    return 1;
  }
  catch (...) {
    fan::print("unknown error");
    return 1;
  }
}