module;

#include <filesystem>
#include <string>
#include <functional>

export module fan.io.directory;

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
  }
}