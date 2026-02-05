module;
#include <cstdint>
#include <string_view>
#include <cstring>
#include <string>
#include <functional>
export module fan.types.compile_time_string;

export namespace fan {
  // compile time string
  template<std::size_t N = 256>
  struct ct_string {
    constexpr ct_string() noexcept : buffer{}, ptr(buffer), len(0) {
      buffer[0] = '\0';
    }
    constexpr ct_string(std::string_view sv) {
      set(sv);
    }
    constexpr ct_string(const ct_string& o) {
      assign_from(o);
    }
    constexpr ct_string(ct_string&& o) noexcept {
      assign_from(o);
    }
    constexpr ct_string& operator=(const ct_string& o) {
      if (this != &o) {
        assign_from(o);
      }
      return *this;
    }
    constexpr ct_string& operator=(ct_string&& o) noexcept {
      if (this != &o) {
        assign_from(o);
      }
      return *this;
    }
    constexpr void assign_from_ptr_len(const char* p, std::size_t l) {
      len = l;
      if (l < N) {
        if (std::is_constant_evaluated()) {
          for (std::size_t i = 0; i < l; ++i) {
            buffer[i] = p[i];
          }
        }
        else {
          std::memcpy(buffer, p, l);
        }
        buffer[l] = '\0';
        ptr = buffer;
      }
      else {
        if (std::is_constant_evaluated()) {
          for (std::size_t i = 0; i < N - 1; ++i) {
            buffer[i] = p[i];
          }
          buffer[N - 1] = '\0';
          ptr = buffer;
          len = N - 1;
        }
        else {
          static thread_local std::string fallback;
          fallback.assign(p, l);
          ptr = fallback.c_str();
        }
      }
    }
    constexpr void assign_from(const ct_string& o) {
      if (o.ptr == o.buffer) {
        len = o.len;
        if (std::is_constant_evaluated()) {
          for (std::size_t i = 0; i < o.len; ++i) {
            buffer[i] = o.buffer[i];
          }
        }
        else {
          std::memcpy(buffer, o.buffer, o.len);
        }
        buffer[len] = '\0';
        ptr = buffer;
      }
      else {
        assign_from_ptr_len(o.ptr, o.len);
      }
    }
    constexpr void set(std::string_view sv) {
      assign_from_ptr_len(sv.data(), sv.size());
    }
    constexpr const char* c_str() const noexcept {
      return ptr;
    }
    constexpr operator const char*() const noexcept {
      return ptr;
    }
    constexpr operator std::string_view() const noexcept {
      return std::string_view(ptr, len);
    }
    constexpr std::size_t hash() const noexcept {
      std::size_t hash_value = 14695981039346656037ULL;
      for (std::size_t i = 0; i < len; ++i) {
        hash_value ^= static_cast<std::size_t>(static_cast<unsigned char>(ptr[i]));
        hash_value *= 1099511628211ULL;
      }
      return hash_value;
    }

    char buffer[N];
    const char* ptr;
    std::size_t len = 0;
  };
  export struct ct_string_hash {
    template<std::size_t N>
    constexpr std::size_t operator()(const fan::ct_string<N>& s) const noexcept {
      return s.hash();
    }
  };
}
