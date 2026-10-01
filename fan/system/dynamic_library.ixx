module;

#include <fan/utility.h>

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
    return dlsym(handle, name);
#endif
  }

  void* handle = nullptr;
};

