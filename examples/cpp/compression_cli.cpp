import fan;
namespace file = fan::io::file;
import std;

static void row(auto n, auto v, auto u) {        fan::printf("{:<12}{:>16}{:>16}", n, v, u); }
static void row_float(auto n, f64_t v, auto u) { fan::printf("{:<12}{:>16.3f}{:>16}", n, v, u); }

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

static bool compress_file(std::string in, std::string out) {
  fan::bytes_t src = file::read_binary(in);
  if (src.empty()) {
    fan::print("read failed", in);
    return false;
  }

  fan::print("Start compression");

  std::size_t os = src.size();
  fan::time::timer t;
  fan::bytes_t cmp = fan::fcs::compress(std::move(src), in);

  f64_t ms = t.millis();
  if (!file::write(out, cmp)) {
    fan::print("write failed", out);
    return false;
  }

  print_compress_stats(os, cmp.size(), ms);
  fan::print("wrote", out);
  return true;
}

static bool decompress_file(std::string in, std::string out) {
  fan::bytes_t src = file::read_binary(in);
  if (src.empty()) {
    fan::print("read failed", in);
    return false;
  }

  fan::print("Start decompression");

  fan::time::timer t;
  auto dec = fan::fcs::decompress(src);

  f64_t ms = t.millis();
  if (!file::write(out, dec.data)) {
    fan::print("write failed", out);
    return false;
  }

  print_decompress_stats(dec.data.size(), ms);
  fan::print("wrote", out);
  return true;
}

static int usage(std::string_view exe) {
  fan::print("usage:");
  fan::printf("  {} c <input> [output]", exe);
  fan::printf("  {} d <input.fcs> [output]", exe);
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    return usage(argv[0]);
  }

  std::string mode = argv[1];
  std::string in = argv[2];
  std::string out;

  if (mode == "c" || mode == "compress") {
    out = argc >= 4 ? argv[3] : file::replace_extension(in, ".fcs");
    return compress_file(in, out) ? 0 : 1;
  }

  if (mode == "d" || mode == "decompress") {
    out = argc >= 4 ? argv[3] : fan::fcs::archive_output_path(in);
    return decompress_file(in, out) ? 0 : 1;
  }

  return usage(argv[0]);
}