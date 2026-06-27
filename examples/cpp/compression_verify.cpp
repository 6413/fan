import std;
import fan;
import fan.compression;

int main() {
  const std::string in_path  = "compress.txt";
  const std::string fcs_path = "compress.txt.fcs";
  const std::string out_dir  = "compress_out";

  std::string original;
  if (fan::io::file::read(in_path, &original)) {
    fan::print("error: cannot read", in_path);
    return 1;
  }
  fan::print("original size:", original.size(), "bytes");

  {
    fan::progress_t prog;
    fan::time::timer t;
    std::jthread monitor([&](std::stop_token st) {
      while (!st.stop_requested()) {
        fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
      fan::print("");
    });
    if (!fan::fcs::compress_path_to_file(in_path, fcs_path, fan::fcs::params_max(), &prog)) {
      fan::print("error: compress failed");
      return 1;
    }
    f64_t ms = t.millis();
    f64_t cs = f64_t(fan::io::file::file_size(fcs_path));
    fan::printf("compressed:  {:.0f} bytes  ({:.2f}%)  {:.1f} ms  {:.2f} MiB/s",
      cs, cs / original.size() * 100.0,
      ms, fan::bytes_to_mib_per_s(original.size(), ms));
  }

  {
    fan::progress_t prog;
    fan::time::timer t;
    std::jthread monitor([&](std::stop_token st) {
      while (!st.stop_requested()) {
        fan::print_progress(prog.done.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
      }
      fan::print_progress(prog.total.load(std::memory_order_relaxed), prog.total.load(std::memory_order_relaxed));
      fan::print("");
    });
    if (!fan::fcs::decompress_file_to_dir(fcs_path, out_dir, false, &prog)) {
      fan::print("error: decompress failed");
      return 1;
    }
    fan::printf("decompressed: {:.1f} ms", t.millis());
  }

  std::string restored;
  std::string restored_path = out_dir + "/" + std::filesystem::path(in_path).filename().string();
  if (fan::io::file::read(restored_path, &restored)) {
    fan::print("error: cannot read restored file:", restored_path);
    return 1;
  }

  if (original.size() != restored.size()) {
    fan::printf("MISMATCH: size {} vs {}", original.size(), restored.size());
    return 1;
  }
  auto diff = std::mismatch(original.begin(), original.end(), restored.begin());
  if (diff.first != original.end()) {
    fan::printf("MISMATCH at byte {}: {:02x} vs {:02x}",
      std::size_t(diff.first - original.begin()),
      std::uint8_t(*diff.first), std::uint8_t(*diff.second));
    return 1;
  }

  fan::print("OK: roundtrip verified,", original.size(), "bytes match");
  return 0;
}