module;

#include <cstdint>
#include <string_view>
#include <cstring>
#include <string>

export module fan.types.compile_time_string;

export namespace fan{
  template<std::size_t N = 256>
  struct temp_cstr {
    char buffer[N];
    const char* ptr;

    temp_cstr() {
      buffer[0] = '\0';
      ptr = buffer;
    }
    temp_cstr(std::string_view sv) {
      set(sv);
    }
    void set(std::string_view sv) {
      if (sv.size() < N) {
        std::memcpy(buffer, sv.data(), sv.size());
        buffer[sv.size()] = '\0';
        ptr = buffer;
      }
      else {
        static thread_local std::string fallback;
        fallback.assign(sv.data(), sv.size());
        ptr = fallback.c_str();
      }
    }
    operator const char* () const { return ptr; }
  };
}