import std;
import fan;

namespace file = fan::io::file;

using namespace fan::graphics;

static auto& con() { return gloco()->get_console(); }

static void row(auto n, auto v, auto u) {        con().printfln("{:<12}{:>16}{:>16}", n, v, u); }
static void row_float(auto n, f64_t v, auto u) { con().printfln("{:<12}{:>16.3f}{:>16}", n, v, u); }

enum class compression_level_e { fast, normal, high, max };

struct cli_options_t {
  compression_level_e level = compression_level_e::max;
  bool verbose = true;
  int thread_count = 0;
  int chunk_mib = 0;
};

struct input_stats_t {
  std::uint64_t size = 0, files = 0;
};

struct archive_data {
  std::string in, out;
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
  if (options.chunk_mib > 0) {
    constexpr std::size_t max_mib = std::numeric_limits<std::uint32_t>::max() >> 20;
    params.chunk_size = std::min<std::size_t>(std::size_t(options.chunk_mib), max_mib) << 20;
  }
  return params;
}

static std::string_view level_name(compression_level_e level) {
  switch (level) {
    case compression_level_e::fast: return "fast";
    case compression_level_e::normal: return "normal";
    case compression_level_e::high: return "high";
    default: return "max";
  }
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
    stats.size = file::file_size(in);
    stats.files = 1;
  }
  return stats;
}

static void print_compress_stats(const input_stats_t& stats, std::uint64_t cs, f64_t ms) {
  row("files", stats.files, "");
  row("original", stats.size, "bytes");
  row("fcs", cs, "bytes");
  row_float("ratio", stats.size ? f64_t(cs) / stats.size * 100.0 : 0, "%");
  row_float("saved", stats.size ? (1.0 - f64_t(cs) / stats.size) * 100.0 : 0, "%");
  row_float("compress", ms, "ms");
  row_float("comp speed", fan::bytes_to_mib_per_s(stats.size, ms), "MiB/s");
}

static void print_decompress_stats(const input_stats_t& stats, f64_t ms) {
  row("files", stats.files, "");
  row("output", stats.size, "bytes");
  row_float("decompress", ms, "ms");
  row_float("decomp speed", fan::bytes_to_mib_per_s(stats.size, ms), "MiB/s");
}

fan::event::task_t compression_task(std::string in, std::string out, cli_options_t options) {
  if (out.empty()) {
    std::string_view s = in;
    while (s.ends_with('/') || s.ends_with('\\')) { s.remove_suffix(1); }
    out = std::string(s) + ".fcs";
  }

  con().printfln("compressing {} -> {}", in, out);
  gui::print("Start compression");

  auto stats = get_input_stats(in);
  fan::time::timer t;

  bool ok = co_await fan::event::run_on_thread([in = std::move(in), out, options] {
    progress_monitor_t pm;
    return fan::fcs::compress_path_to_file(
      in,
      out,
      get_params(options),
      &pm.prog,
      options.verbose,
      options.thread_count > 0 ? std::size_t(options.thread_count) : 0
    );
  });

  if (!ok) {
    con().printfln("write failed {}", out);
    gui::print("Compression failed");
    co_return;
  }

  f64_t ms = t.millis();
  print_compress_stats(stats, file::file_size(out), ms);
  con().printfln("wrote {}", out);
  gui::print("Done");
}

fan::event::task_t decompression_task(std::string in, std::string out) {
  bool default_out = out.empty();
  if (default_out) {
    out = in.ends_with(".fcs") ? file::strip_extension(in) : in + "_ext";
  }

  con().printfln("decompressing {} -> {}", in, out);
  gui::print("Start decompression");

  fan::time::timer t;

  bool ok = co_await fan::event::run_on_thread([in = std::move(in), out, default_out] {
    progress_monitor_t pm;
    return fan::fcs::decompress_file_to_dir(in, out, default_out, &pm.prog);
  });

  if (!ok) {
    con().printfln("read failed {}", in);
    gui::print("Decompression failed");
    co_return;
  }

  f64_t ms = t.millis();
  auto stats = get_input_stats(out);
  print_decompress_stats(stats, ms);
  con().printfln("extracted to {}", out);
  gui::print("Done");
}

static void pick(archive_data& d, bool dec) {
  open_file(dec ? "fcs" : "", [&d, dec](std::string_view p) {
    std::string path{p};
    d.in = path;
    d.out = dec ?
      (path.ends_with(".fcs") ? file::strip_extension(path) : path + "_ext") :
      file::replace_extension(path, ".fcs");
  });
}

static void compression_options(cli_options_t& options) {
  gui::text("level:");
  gui::same_line();
  gui::text(level_name(options.level));
  gui::same_line();

  if (gui::button("fast")) { options.level = compression_level_e::fast; }
  gui::same_line();
  if (gui::button("normal")) { options.level = compression_level_e::normal; }
  gui::same_line();
  if (gui::button("high")) { options.level = compression_level_e::high; }
  gui::same_line();
  if (gui::button("max")) { options.level = compression_level_e::max; }

  gui::new_line();
  gui::text(options.verbose ? "verbose: on" : "verbose: off");
  gui::same_line();
  if (gui::button("toggle verbose")) {
    options.verbose = !options.verbose;
  }

  gui::new_line();
  gui::drag("threads", &options.thread_count);
  if (options.thread_count < 0) { options.thread_count = 0; }

  gui::new_line();
  gui::drag("chunk MiB", &options.chunk_mib);
  if (options.chunk_mib < 0) { options.chunk_mib = 0; }
}

static void compression_section(archive_data& d, cli_options_t& options) {
  gui::push_id("compress");
  gui::text("Compress", {.font_size = 32.f});
  gui::text(std::string(40, '-'));

  if (gui::button("Add to archive")) {
    pick(d, false);
  }

  gui::same_line();
  gui::text(d.in);

  if (!d.in.empty()) {
    gui::new_line();
    gui::text("output:");
    gui::input_text(&d.out);
    gui::new_line();

    compression_options(options);
    gui::new_line();

    if (gui::button("Compress")) {
      fan::event::add_awaitable(compression_task(d.in, d.out, options));
    }
  }

  gui::pop_id();
}

static void decompression_section(archive_data& d) {
  gui::push_id("decompress");
  gui::text("Decompress", {.font_size = 32.f});
  gui::text(std::string(40, '-'));

  if (gui::button("Decompress archive")) {
    pick(d, true);
  }

  gui::same_line();
  gui::text(d.in);

  if (!d.in.empty()) {
    gui::new_line();
    gui::text("output:");
    gui::input_text(&d.out);
    gui::new_line();

    if (gui::button("Decompress")) {
      fan::event::add_awaitable(decompression_task(d.in, d.out));
    }
  }

  gui::pop_id();
}

void render_gui() {
  static archive_data c, d;
  static cli_options_t options;

  gui::text("File Compression (fcs)", {.font_size = 48.f});
  gui::new_line();

  compression_section(c, options);

  gui::new_line();
  gui::new_line();

  decompression_section(d);
}

int main() {
  engine_t engine{fan::get_centered_window({626, 573})};

  engine.loop([&] {
    if (auto h = gui::hud_interactive{"##a"}) {
      render_gui();
    }
  });
}