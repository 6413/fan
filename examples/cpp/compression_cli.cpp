import std;
import fan;

namespace file = fan::io::file;

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

static bool cmd_compress(const std::string& in, std::string out, const cli_options_t& options) {
  if (out.empty()) {
    std::string_view s = in;
    while (s.ends_with('/') || s.ends_with('\\')) { s.remove_suffix(1); }
    out = std::string(s) + ".fcs";
  }
  if (!ask_overwrite(out)) { return true; }
  std::vector<fan::fcs::archive_file_t> files;
  std::size_t total_os = 0;
  std::error_code ec;

  if (std::filesystem::is_directory(in, ec)) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(in)) {
      if (!entry.is_regular_file()) { continue; }
      fan::bytes_t d = file::read_binary(entry.path().string());
      total_os += d.size();
      auto rel = std::filesystem::relative(entry.path(), in);
      if (options.verbose) {
        fan::print("found:", rel.generic_string());
      }
      files.push_back({rel.generic_string(), std::move(d)});
    }
    fan::print("found", files.size(), "files");
  }
  else {
    fan::bytes_t d = file::read_binary(in);
    if (d.empty()) { fan::print("read failed", in); return false; }
    total_os += d.size();
    files.push_back({std::filesystem::path(in).filename().string(), std::move(d)});
  }

  fan::print("compressing", in, "->", out, "(", files.size(), "files)");
  fan::time::timer t;
  fan::bytes_t dst;

  {
    fan::fcs::progress_t prog;
    std::jthread monitor([&](std::stop_token st) {
      while (!st.stop_requested()) {
        fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
      fan::print("");
    });
    dst = fan::fcs::compress(files, get_params(options.level), &prog);
  }

  f64_t ms = t.millis();

  if (!file::write(out, dst)) {
    fan::print("write failed", out);
    return false;
  }

  row("files", files.size(), "");
  row("original", total_os, "bytes");
  row("fcs", dst.size(), "bytes");
  row_float("ratio", total_os ? f64_t(dst.size()) / total_os * 100.0 : 0, "%");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(total_os, ms), "MiB/s");
  fan::print("wrote", out);
  return true;
}

static bool cmd_decompress(const std::string& in, std::string out_dir, const cli_options_t&) {
  if (out_dir.empty()) {
    out_dir = in.ends_with(".fcs") ? in.substr(0, in.size() - 4) : in + "_ext";
  }
  fan::bytes_t src = file::read_binary(in);
  if (src.empty()) {
    fan::print("read failed", in);
    return false;
  }
  fan::print("decompressing", in, "->", out_dir);
  fan::time::timer t;
  std::vector<fan::fcs::archive_file_t> files;

  {
    fan::fcs::progress_t prog;
    std::jthread monitor([&](std::stop_token st) {
      while (!st.stop_requested()) {
        fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
      fan::print("");
    });
    files = fan::fcs::decompress(src, &prog);
  }

  f64_t ms = t.millis();

  std::size_t total_os = 0;
  for (const auto& f : files) {
    std::filesystem::path p = std::filesystem::path(out_dir) / f.path;
    std::filesystem::create_directories(p.parent_path());
    if (!file::write(p.string(), f.data)) {
      fan::print("write failed", p.string());
      return false;
    }
    total_os += f.data.size();
  }

  row("files", files.size(), "");
  row("extracted", total_os, "bytes");
  row_float("time", ms, "ms");
  row_float("speed", fan::bytes_to_mib_per_s(total_os, ms), "MiB/s");
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