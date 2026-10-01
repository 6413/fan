module;

#include <fan/utils/utility.h>

#if defined(fan_platform_windows)
  #define WIN32_LEAN_AND_MEAN
  #include <Windows.h>
#elif defined(fan_platform_linux)
  #include <dlfcn.h>
#endif

export module fan.system.dynamic_library;

import std;

export struct dynamic_library_t {
  dynamic_library_t() = default;
  dynamic_library_t(dynamic_library_t&& o) : handle(std::exchange(o.handle, nullptr)) {}
  dynamic_library_t& operator=(dynamic_library_t&& o) {

    unload();
    handle = std::exchange(o.handle, nullptr);
    return *this;
  }
  ~dynamic_library_t() {
    unload();
  }
  bool load(const std::string& path) {
    unload();

#if defined(fan_platform_windows)
    handle = (void*)LoadLibraryA(path.c_str());
#elif defined(fan_platform_linux)
    dlerror(); // clear stale errors so last_error() is accurate
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    return handle != nullptr;
  }
  void unload() {
    if (!handle) {
      return;
    }

#if defined(fan_platform_windows)
    FreeLibrary(static_cast<HMODULE>(handle));
#elif defined(fan_platform_linux)
    dlclose(handle);
#endif
    handle = nullptr;
  }
  void* symbol(const char* name) {
#if defined(fan_platform_windows)
    return static_cast<void*>(GetProcAddress(
          static_cast<HMODULE>(handle), name));
#elif defined(fan_platform_linux)
    if (!handle) {
      return nullptr;
    }
    dlerror(); // clear stale errors so last_error() is accurate
    return dlsym(handle, name);
#endif
  }
  bool is_loaded() const { return handle != nullptr; }
  operator bool() const { return handle != nullptr; }

  static std::string last_error() {
#if defined(fan_platform_windows)
    DWORD err = GetLastError();
    if (err == 0) {
      return "unknown error";
    }
    char* msg = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
      nullptr, err, 0, (LPSTR)&msg, 0, nullptr);
    std::string s = msg ? msg : "unknown error";
    if (msg) {
      LocalFree(msg);
    }
    return s;
#elif defined(fan_platform_linux)
    const char* e = dlerror();
    return e ? e : "unknown error";
#else
    return "unsupported platform";
#endif
  }

  void* handle = nullptr;
};

