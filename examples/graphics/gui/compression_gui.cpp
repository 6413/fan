import fan;
namespace file = fan::io::file;
import std;

using namespace fan::graphics;

static auto& con() { return gloco()->get_console(); }

static void row(auto n, auto v, auto u) {        con().printfln("{:<12}{:>16}{:>16}", n, v, u); }
static void row_float(auto n, f64_t v, auto u) { con().printfln("{:<12}{:>16.3f}{:>16}", n, v, u); }

static void print_compress_stats(std::size_t os, std::size_t cs, f64_t ms) {
  row("original", os, "bytes");
  row("fcs", cs, "bytes");
  row_float("ratio", os ? f64_t(cs) / os * 100.0 : 0, "%");
  row_float("compress", ms, "ms");
  row_float("comp speed", fan::bytes_to_mib_per_s(os, ms), "MiB/s");
}
static void print_decompress_stats(std::size_t os, f64_t ms) {
  row_float("decompress", ms, "ms");
  row_float("decomp speed", fan::bytes_to_mib_per_s(os, ms), "MiB/s");
}

fan::event::task_t compression_task(std::string in, std::string out) {
  fan::bytes_t src = file::read_binary(in);
  std::size_t os = src.size();

  con().printfln("Start compression");
  gui::print("Start compression");

  fan::time::timer t;
  fan::bytes_t cmp = co_await fan::event::run_on_thread([src = std::move(src), in = std::move(in)]() mutable {
    return fan::fcs::compress(std::move(src), in);
  });

  f64_t ms = t.millis();
  file::write(out, cmp);
  print_compress_stats(os, cmp.size(), ms);
  gui::print("Done");
}

fan::event::task_t decompression_task(std::string in, std::string out) {
  fan::bytes_t src = file::read_binary(in);

  con().printfln("Start decompression");
  gui::print("Start decompression");

  fan::time::timer t;
  auto dec = co_await fan::event::run_on_thread([src = std::move(src)] {
    return fan::fcs::decompress(src);
  });

  f64_t ms = t.millis();
  file::write(out, dec.data);
  print_decompress_stats(dec.data.size(), ms);
  gui::print("Done");
}

struct archive_data {
  std::string in, out;
};

static void pick(archive_data& d, bool dec) {
  open_file(dec ? "fcs" : "", [&d, dec](std::string_view p) {
    std::string path{p};
    d.in = path;
    d.out = dec ? fan::fcs::archive_output_path(path) : file::replace_extension(path, ".fcs");
  });
}

static void section(const char* pick_text, const char* run_text, archive_data& d, bool dec, auto run) {
  gui::push_id(pick_text);
  gui::text(run_text, {.font_size = 32.f});
  gui::text(std::string(40, '-'));

  if (gui::button(pick_text)) {
    pick(d, dec);
  }

  gui::same_line();
  gui::text(d.in);

  if (!d.in.empty()) {
    gui::new_line();
    gui::text("output:");
    gui::input_text(&d.out);
    gui::new_line();

    if (gui::button(run_text)) {
      con().printfln("write to {}", d.out);
      run(d.in, d.out);
    }
  }

  gui::pop_id();
}

void render_gui() {
  static archive_data c, d;

  gui::text("File Compression (fcs)", {.font_size = 48.f});
  gui::new_line();

  section("Add to archive", "Compress", c, false, [](auto& in, auto& out) {
    fan::event::add_awaitable(compression_task(in, out));
  });

  gui::new_line();
  gui::new_line();

  section("Decompress archive", "Decompress", d, true, [](auto& in, auto& out) {
    fan::event::add_awaitable(decompression_task(in, out));
  });
}

int main() {
  engine_t engine{fan::get_centered_window({626, 573})};

  engine.loop([&] {
    if (auto h = gui::hud_interactive{"##a"}) {
      render_gui();
    }
  });
}