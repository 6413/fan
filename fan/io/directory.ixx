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
      std::uint8_t read(std::uint64_t offset) const override;
      void write(std::uint64_t, std::uint8_t) override {}

      std::uint64_t read_range(std::uint64_t offset, std::uint64_t length, fan::bytes_t& out_buffer) const override;

      void append_bytes(std::span<const std::uint8_t> bytes);
      void append_file(const std::filesystem::path& path, std::uint64_t size);

      std::vector<segment_t> segments;
      std::uint64_t total_size = 0;
    };
    
    namespace file {
      struct archive_extractor_t {
        enum class state_e { file_count, path_len, path, size, padding, data, done };
        archive_extractor_t(std::filesystem::path out_dir, bool default_out) : out_dir(std::move(out_dir)), default_out(default_out) { write_buffer.reserve(1 << 20); }
        ~archive_extractor_t() { close_file(); }

        void put(std::uint8_t b);
        void enter_data();
        void finish();
        void set_state(state_e s, std::size_t n) { state = s; need = n; tmp.clear(); }
        void open_file();
        void write_data(std::uint8_t b);
        void finish_file() { flush(); close_file(); ++file_index; set_state(file_index == file_count ? state_e::done : state_e::path_len, file_index == file_count ? 0 : 2); }
        void flush();
        void close_file();
    
        std::filesystem::path out_dir; bool default_out = false; state_e state = state_e::file_count;
        fan::bytes_t tmp, write_buffer; std::string archive_path; file_t* fp = nullptr;
        std::size_t need = 4, total_header = 0, remaining = 0;
        std::uint32_t file_count = 0, file_index = 0; std::uint16_t path_len = 0;
      };
    } // namespace file
  }
}