module;
#include <cstdint>
#include <string_view>
#include <cstring>
#include <string>
#include <functional>

export module fan.types.compile_time_string;

export namespace fan {

  template<std::size_t N = 256>
  struct ct_string {
    constexpr ct_string() noexcept : buffer {}, ptr(buffer), len(0) {
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

    constexpr operator const char* () const noexcept {
      return ptr;
    }

    constexpr operator std::string_view() const noexcept {
      return std::string_view(ptr, len);
    }

    constexpr std::size_t hash() const noexcept {
      std::size_t h = 14695981039346656037ULL;
      for (std::size_t i = 0; i < len; ++i) {
        h ^= static_cast<unsigned char>(ptr[i]);
        h *= 1099511628211ULL;
      }
      return h;
    }

    char buffer[N];
    const char* ptr;
    std::size_t len = 0;
  };

  struct ct_string_hash {
    using is_transparent = void;

    template<std::size_t N>
    constexpr std::size_t operator()(const ct_string<N>& s) const noexcept {
      return s.hash();
    }

    constexpr std::size_t operator()(std::string_view sv) const noexcept {
      return hash_sv(sv);
    }

    constexpr std::size_t operator()(const char* s) const noexcept {
      return hash_sv(std::string_view(s));
    }

    static constexpr std::size_t hash_sv(std::string_view sv) noexcept {
      std::size_t h = 14695981039346656037ULL;
      for (char c : sv) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ULL;
      }
      return h;
    }
  };

  struct ct_string_equal {
    using is_transparent = void;

    template<std::size_t N>
    constexpr bool operator()(const ct_string<N>& a, const ct_string<N>& b) const noexcept {
      return std::string_view(a.ptr, a.len) == std::string_view(b.ptr, b.len);
    }

    template<std::size_t N>
    constexpr bool operator()(const ct_string<N>& a, std::string_view b) const noexcept {
      return std::string_view(a.ptr, a.len) == b;
    }

    template<std::size_t N>
    constexpr bool operator()(std::string_view a, const ct_string<N>& b) const noexcept {
      return a == std::string_view(b.ptr, b.len);
    }

    template<std::size_t N>
    constexpr bool operator()(const ct_string<N>& a, const char* b) const noexcept {
      return std::string_view(a.ptr, a.len) == std::string_view(b);
    }

    template<std::size_t N>
    constexpr bool operator()(const char* a, const ct_string<N>& b) const noexcept {
      return std::string_view(a) == std::string_view(b.ptr, b.len);
    }
  };

  // Converts snake_case to Title Case in compile time
  template<size_t N = 256>
  constexpr ct_string<N> snake_to_title(const char* s) {
    ct_string<N> o {};
    for (bool c = 1; *s && o.len < N - 1; c = *s == '_', ++s)
      o.buffer[o.len++] = *s == '_' ? ' ' : (c ? (*s & ~32) : *s);
    return o.buffer[o.len] = 0, o.ptr = o.buffer, o;
  }
}