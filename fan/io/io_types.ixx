// fan.io.types.ixx
module;

#include <fan/utility.h>

export module fan.io.types;

import std;

export namespace fan {
  namespace tmpl {
    template<typename> struct is_std_vector : std::false_type {};
    template<typename T, typename A> struct is_std_vector<std::vector<T, A>> : std::true_type {};
  }

  namespace io {
    struct data_provider_t {
      virtual ~data_provider_t() = default;

      virtual std::uint64_t size() const = 0;
      virtual std::uint8_t read(std::uint64_t offset) const = 0;
      virtual void write(std::uint64_t offset, std::uint8_t value) = 0;

      // Fills existing buffer. Returns actual bytes read. Prevents overflow.
      virtual std::uint64_t read_range(std::uint64_t offset, std::uint64_t length, std::vector<std::uint8_t>& out_buffer) const = 0;

      virtual void write_range(std::uint64_t offset, std::span<const std::uint8_t> bytes) {
        for (std::uint64_t i = 0; i < bytes.size(); ++i)
          write(offset + i, bytes[i]);
      }

      std::uint64_t read_range_padded(std::uint64_t offset, std::uint64_t length, std::vector<std::uint8_t>& out) const {
        auto r = read_range(offset, length, out);
        while (out.size() < length) out.push_back(0);
        return r;
      }
    };

    struct memory_provider_t : public data_provider_t {
      // Explicitly non-owning view via std::span
      memory_provider_t(std::span<std::uint8_t> data_view) : data(data_view) {}

      std::uint64_t size() const override {
        return data.size();
      }

      std::uint8_t read(std::uint64_t offset) const override {
        if (offset >= data.size()) return 0x00;
        return data[offset];
      }

      void write(std::uint64_t offset, std::uint8_t value) override {
        if (offset >= data.size()) return;
        data[offset] = value;
      }

      std::uint64_t read_range(std::uint64_t offset, std::uint64_t length, std::vector<std::uint8_t>& out_buffer) const override {
        if (offset >= data.size()) {
          out_buffer.clear();
          return 0;
        }

        // Prevent integer overflow on offset + length
        std::uint64_t max_len = data.size() - offset;
        std::uint64_t actual_length = std::min(length, max_len);

        // Retain capacity, avoid reallocation, fast copy
        out_buffer.resize(actual_length);
        std::copy_n(data.begin() + offset, actual_length, out_buffer.begin());

        return actual_length;
      }

      void write_range(std::uint64_t offset, std::span<const std::uint8_t> bytes) override {
        if (offset >= data.size()) return;
        std::uint64_t len = std::min((std::uint64_t)bytes.size(), data.size() - offset);
        std::copy_n(bytes.begin(), len, data.begin() + offset);
      }

    private:
      std::span<std::uint8_t> data;
    };

    template <typename T>
    void inspector_write(data_provider_t& data, std::uint64_t offset, T val) {
      auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(val);
      data.write_range(offset, bytes);
    }

    namespace file {
      using file_t = std::FILE;
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