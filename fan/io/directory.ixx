module;

export module fan.io.directory;

import fan.types;
import fan.io.file;
import fan.memory;

import std;

export namespace fan {
  namespace io {

    std::string file_to_directory(const std::string& file);
    bool directory_exists(const std::string& directory);
    void create_directory(const std::string& folders);

    struct iterate_sort_t {
      std::string path;
      std::uint64_t area;
      static bool comp_cb(const iterate_sort_t& a, const iterate_sort_t& b);
    };

    void handle_string_out(std::string& str);
    void handle_string_in(std::string& str);

    bool is_readable_path(const std::string& path);

    std::string exclude_path(const std::string& full_path);

    void iterate_directory(
      const std::string& path,
      const std::function<void(const std::string& path, bool is_directory)>& function
    );

    void iterate_directory(
      const std::filesystem::path& path,
      const std::function<void(const std::filesystem::directory_entry& path)>& function
    );

    void iterate_directory_sorted_by_name(
      const std::filesystem::path& path,
      const std::function<void(const std::filesystem::directory_entry&)>& function
    );

    void iterate_directory_files(
      const std::string& path,
      const std::function<void(const std::string& path)>& function
    );

    void iterate_files_recursive(
      const std::filesystem::path& path,
      const std::function<void(const std::filesystem::path& full, const std::filesystem::path& rel)>& function
    );

    bool is_safe_path(const std::filesystem::path& path);

    struct vfs_provider_t : public data_provider_t {
      struct segment_t {
        std::uint64_t start = 0, size = 0, file_offset = 0;
        fan::bytes_t bytes;
        std::filesystem::path file_path;
      };

      std::uint64_t size() const override { return total_size; }
      std::uint8_t read(std::uint64_t offset) const override { fan::bytes_t b; read_range(offset, 1, b); return b.empty() ? 0 : b[0]; }
      void write(std::uint64_t, std::uint8_t) override {}

      std::uint64_t read_range(std::uint64_t offset, std::uint64_t length, fan::bytes_t& out_buffer) const override {
        if (offset >= total_size) { out_buffer.clear(); return 0; }
        std::uint64_t actual = std::min(length, total_size - offset), end = offset + actual;
        out_buffer.assign(std::size_t(actual), 0);
        for (const auto& s : segments) {
          std::uint64_t s_end = s.start + s.size;
          if (s_end <= offset) { continue; }
          if (s.start >= end) { break; }
          std::uint64_t p = std::max(offset, s.start), n = std::min(end, s_end) - p, local = p - s.start;
          auto* dst = out_buffer.data() + std::size_t(p - offset);
          if (!s.bytes.empty()) {
            std::copy_n(s.bytes.data() + std::size_t(local), std::size_t(n), dst);
          }
          else {
            fan::io::file::file_t* f = nullptr;
            if (!fan::io::file::open(&f, s.file_path.string(), {"rb"})) {
              if (local > 0) { std::vector<std::uint8_t> dump(local); fan::io::file::read(f, dump.data(), 1, dump.size()); }
              if (fan::io::file::read(f, dst, 1, n)) { fan::io::file::close(f); throw std::runtime_error("short read"); }
              fan::io::file::close(f);
            }
            else { throw std::runtime_error("read failed: " + s.file_path.string()); }
          }
        }
        return actual;
      }

      void append_bytes(std::span<const std::uint8_t> bytes) {
        segment_t s; s.start = total_size; s.size = bytes.size(); s.bytes.assign(bytes.begin(), bytes.end());
        segments.push_back(std::move(s)); total_size += bytes.size();
      }
      void append_file(const std::filesystem::path& path, std::uint64_t size) {
        segment_t s; s.start = total_size; s.size = size; s.file_path = path;
        if (size <= 16 * 1024 * 1024) { s.bytes = fan::io::file::read_binary(path.string()); }
        segments.push_back(std::move(s)); total_size += size;
      }

      std::vector<segment_t> segments;
      std::uint64_t total_size = 0;
    };
    
    namespace file {
      struct archive_extractor_t {
        enum class state_e { file_count, path_len, path, size, padding, data, done };
        archive_extractor_t(std::filesystem::path out_dir, bool default_out) : out_dir(std::move(out_dir)), default_out(default_out) { write_buffer.reserve(1 << 20); }
        ~archive_extractor_t() { close_file(); }

        void put(std::uint8_t b) {
          if (state == state_e::data) { write_data(b); return; }
          tmp.push_back(b);
          if (tmp.size() != need) { return; }

          if (state == state_e::file_count) {
            file_count = fan::memory::read_le32(tmp.data()); file_index = 0; total_header += 4;
            set_state(file_count ? state_e::path_len : state_e::done, file_count ? 2 : 0);
          } else if (state == state_e::path_len) {
            path_len = fan::memory::read_le16(tmp.data()); total_header += 2; set_state(state_e::path, path_len);
          } else if (state == state_e::path) {
            archive_path.assign(reinterpret_cast<const char*>(tmp.data()), tmp.size()); total_header += path_len;
            if (default_out && file_count == 1 && out_dir.filename().string() == archive_path) { out_dir = "."; }
            set_state(state_e::size, 8);
          } else if (state == state_e::size) {
            remaining = fan::memory::read_le64(tmp.data()); total_header += 8; open_file();
            if (file_index + 1 == file_count) {
              std::size_t pad = (4 - (total_header & 3)) & 3;
              if (pad) { set_state(state_e::padding, pad); } else { enter_data(); }
            } else { enter_data(); }
          } else if (state == state_e::padding) { enter_data(); }
        }

        void enter_data() { if (remaining == 0) finish_file(); else set_state(state_e::data, 0); }
        void finish() { flush(); close_file(); if (state != state_e::done) { throw std::runtime_error("truncated archive"); } }
        void set_state(state_e s, std::size_t n) { state = s; need = n; tmp.clear(); }
    
        void open_file() {
          std::filesystem::path p = out_dir / archive_path;
          fan::io::create_directory(p.parent_path().string());
          if (fan::io::file::open(&fp, p.string(), {"wb"})) { throw std::runtime_error("write failed"); }
        }
        void write_data(std::uint8_t b) {
          write_buffer.push_back(b);
          if (write_buffer.size() == write_buffer.capacity()) { flush(); }
          if (--remaining == 0) { finish_file(); }
        }
        void finish_file() { flush(); close_file(); ++file_index; set_state(file_index == file_count ? state_e::done : state_e::path_len, file_index == file_count ? 0 : 2); }
        void flush() {
          if (!fp || write_buffer.empty()) { return; }
          if (fan::io::file::write(fp, write_buffer.data(), 1, write_buffer.size())) { throw std::runtime_error("write failed"); }
          write_buffer.clear();
        }
        void close_file() { if (fp) { fan::io::file::close(fp); fp = nullptr; } }
    
        std::filesystem::path out_dir; bool default_out = false; state_e state = state_e::file_count;
        fan::bytes_t tmp, write_buffer; std::string archive_path; file_t* fp = nullptr;
        std::size_t need = 4, total_header = 0, remaining = 0;
        std::uint32_t file_count = 0, file_index = 0; std::uint16_t path_len = 0;
      };
    } // namespace file
  }
}