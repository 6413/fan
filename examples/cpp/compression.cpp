import fan;
namespace file = fan::io::file;
import std;

static void print_results(const fan::bytes_t& o, const fan::bytes_t& c, f64_t cms, f64_t dms, std::ptrdiff_t m) {
  fan::print(fan::bytes_t(84, '-'));
  auto p = [](auto n, auto s, auto u) { fan::printf("{:<12}{:>16}{:>16}", n, s, u); };
  p("original", o.size(), "bytes"); p("fcs", c.size(), "bytes"); p("zip", file::read_binary("compress.zip").size(), "bytes");
  auto p2 = [](auto&&... args) {fan::printf("{:<12}{:>16.3f}{:>16}", args...);};
  p2("ratio", o.empty() ? 0 : f64_t(c.size()) / o.size() * 100, "%");
  p2("compress", cms, "ms");
  p2("comp speed", fan::bytes_to_mib_per_s(o.size(), cms), "MiB/s");
  p2("decompress", dms, "ms");
  p2("decomp speed", fan::bytes_to_mib_per_s(o.size(), dms), "MiB/s");
  p("valid", m == -1 ? "yes" : "no", "");
  if (m != -1) { p("mismatch", m, "byte"); }
}

int main() {
  fan::bytes_t src = file::read_binary("compress.txt");
  fan::time::timer t;
  fan::bytes_t cmp = fan::fcs::compress(src);
  f64_t cms = t.millis(); t.restart();
  fan::bytes_t dec = fan::fcs::decompress(cmp);
  f64_t dms = t.millis();
  auto [i0, i1] = std::ranges::mismatch(dec, src);
  std::ptrdiff_t m = (i0 != dec.end() || i1 != src.end()) ? i0 - dec.begin() : -1;
  file::write("compress.fcs", cmp);
  print_results(src, cmp, cms, dms, m);
}