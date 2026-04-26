// fan.io.types.ixx
module;
#include <cstdint>
#include <vector>
#include <algorithm>
#include <ios>
#include <cstdio>
#include <filesystem>
#include <string_view>
#include <span>
export module fan.io.types;

export namespace fan {
  namespace tmpl {
    template<typename> struct is_std_vector : std::false_type {};
    template<typename T, typename A> struct is_std_vector<std::vector<T, A>> : std::true_type {};
  }

  namespace io {
    struct data_provider_t {
      virtual ~data_provider_t() = default;

      virtual uint64_t size() const = 0;
      virtual uint8_t read(uint64_t offset) const = 0;
      virtual void write(uint64_t offset, uint8_t value) = 0;

      // Fills existing buffer. Returns actual bytes read. Prevents overflow.
      virtual uint64_t read_range(uint64_t offset, uint64_t length, std::vector<uint8_t>& out_buffer) const = 0;

      virtual void write_range(uint64_t offset, std::span<const uint8_t> bytes) {
        for (uint64_t i = 0; i < bytes.size(); ++i)
          write(offset + i, bytes[i]);
      }

      uint64_t read_range_padded(uint64_t offset, uint64_t length, std::vector<uint8_t>& out) const {
        auto r = read_range(offset, length, out);
        while (out.size() < length) out.push_back(0);
        return r;
      }
    };

    struct memory_provider_t : public data_provider_t {
      // Explicitly non-owning view via std::span
      memory_provider_t(std::span<uint8_t> data_view) : data(data_view) {}

      uint64_t size() const override {
        return data.size();
      }

      uint8_t read(uint64_t offset) const override {
        if (offset >= data.size()) return 0x00;
        return data[offset];
      }

      void write(uint64_t offset, uint8_t value) override {
        if (offset >= data.size()) return;
        data[offset] = value;
      }

      uint64_t read_range(uint64_t offset, uint64_t length, std::vector<uint8_t>& out_buffer) const override {
        if (offset >= data.size()) {
          out_buffer.clear();
          return 0;
        }

        // Prevent integer overflow on offset + length
        uint64_t max_len = data.size() - offset;
        uint64_t actual_length = std::min(length, max_len);

        // Retain capacity, avoid reallocation, fast copy
        out_buffer.resize(actual_length);
        std::copy_n(data.begin() + offset, actual_length, out_buffer.begin());

        return actual_length;
      }

      void write_range(uint64_t offset, std::span<const uint8_t> bytes) override {
        if (offset >= data.size()) return;
        uint64_t len = std::min((uint64_t)bytes.size(), data.size() - offset);
        std::copy_n(bytes.begin(), len, data.begin() + offset);
      }

    private:
      std::span<uint8_t> data;
    };

    template <typename T>
    void inspector_write(data_provider_t& data, uint64_t offset, T val) {
      auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(val);
      data.write_range(offset, bytes);
    }

    namespace file {
      using file_t = FILE;
      struct properties_t { const char* mode; };
      using fs_mode = decltype(std::ios_base::binary);

      template <typename T>
      concept path_t =
        std::is_convertible_v<T, std::string_view> ||
        std::is_convertible_v<T, const char*> ||
        std::is_same_v<std::remove_cvref_t<T>, std::filesystem::path>;
    }
  }
}