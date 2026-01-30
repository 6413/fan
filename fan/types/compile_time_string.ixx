module;

#include <cstdint>
#include <string_view>
#include <cstring>
#include <string>

export module fan.types.compile_time_string;

export namespace fan {
  template<std::size_t N = 256>
  struct temp_cstr {
    char buffer[N];
    const char* ptr;
    std::size_t len = 0;

    temp_cstr() {
      buffer[0] = '\0';
      ptr = buffer;
    }
    temp_cstr(std::string_view sv) {
      set(sv);
    }

    void assign_from_ptr_len(const char* p, std::size_t l) {
      len = l;
      if (l < N) {
        std::memcpy(buffer, p, l);
        buffer[l] = '\0';
        ptr = buffer;
      }
      else {
        static thread_local std::string fallback;
        fallback.assign(p, l);
        ptr = fallback.c_str();
      }
    }

    void assign_from(const temp_cstr& o) {
      assign_from_ptr_len(o.ptr, o.len);
    }

    temp_cstr(const temp_cstr& o) {
      assign_from(o);
    }
    temp_cstr(temp_cstr&& o) noexcept {
      assign_from(o);
    }
    temp_cstr& operator=(const temp_cstr& o) {
      if (this != &o) assign_from(o);
      return *this;
    }
    temp_cstr& operator=(temp_cstr&& o) noexcept {
      if (this != &o) assign_from(o);
      return *this;
    }

    void set(std::string_view sv) {
      assign_from_ptr_len(sv.data(), sv.size());
    }

    const char* c_str() const { return ptr; }
    operator const char* () const { return ptr; }
    operator std::string_view() const { return std::string_view(ptr, len); }
  };
}