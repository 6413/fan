module;

#if defined(fan_compiler_gcc)
  #define ____mbstate_t_defined
#endif

#include <fan/utility.h>
#include <new>

namespace fan::memory::detail {
  inline void* (*malloc_fn)(std::size_t) = nullptr;
  inline void* (*realloc_fn)(void*, std::size_t) = nullptr;
  inline void (*free_fn)(void*) = nullptr;
}

void* operator new(std::size_t size) {
  return fan::memory::detail::malloc_fn ? fan::memory::detail::malloc_fn(size) : std::malloc(size);
}
void* operator new[](std::size_t size) {
  return fan::memory::detail::malloc_fn ? fan::memory::detail::malloc_fn(size) : std::malloc(size);
}
void operator delete(void* ptr) noexcept {
  fan::memory::detail::free_fn ? fan::memory::detail::free_fn(ptr) : std::free(ptr);
}
void operator delete[](void* ptr) noexcept {
  fan::memory::detail::free_fn ? fan::memory::detail::free_fn(ptr) : std::free(ptr);
}
void operator delete(void* ptr, std::size_t) noexcept {
  fan::memory::detail::free_fn ? fan::memory::detail::free_fn(ptr) : std::free(ptr);
}
void operator delete[](void* ptr, std::size_t) noexcept {
  fan::memory::detail::free_fn ? fan::memory::detail::free_fn(ptr) : std::free(ptr);
}

module fan.memory;

import fan.time;

namespace fan::memory {
  heap_profiler_t::~heap_profiler_t() {
    if (!enabled) {
      return;
    }
    enabled = false;
  #if defined(fan_std23)
    if (!print_leaks) {
      return;
    }

    std::printf("=== Memory Leak Report ===\n");

    std::size_t real_leaks = 0;
    std::size_t real_leak_bytes = 0;
    std::size_t static_ignored = 0;

    for (const auto& pair : memory_map) {
      bool is_from_main = false;

      for (const auto& frame : pair.second.line_data) {
        const char* func = frame.description().c_str();
        if (func && std::strstr(func, "main")) {
          is_from_main = true;
          break;
        }
      }

      if (is_from_main) {
        real_leaks++;
        real_leak_bytes += pair.second.n;
      }
      else {
        static_ignored++;
      }
    }

    std::printf("Total leaked memory: %llu bytes\n", (unsigned long long)real_leak_bytes);
    std::printf("Number of leaked allocations: %zu\n", real_leaks);

    if (real_leaks > 0) {
      std::printf("\nLeak details:\n");
      for (auto it = memory_set.rbegin(); it != memory_set.rend(); ++it) {
        const auto& leak = *it;

        bool is_from_main = false;
        for (const auto& frame : leak.line_data) {
          const char* func = frame.description().c_str();
          if (func && std::strstr(func, "main")) {
            is_from_main = true;
            break;
          }
        }

        if (!is_from_main) {
          continue;
        }

        std::printf("Leaked %zu bytes at address %p\n", leak.n, leak.p);
        std::printf("Stack trace:\n");
        for (const auto& frame : leak.line_data) {
          const char* file = frame.source_file().c_str();
          const char* func = frame.description().c_str();
          auto line = frame.source_line();
          if (file && func) {
            std::printf("  %s:%u in %s\n", file, line, func);
          }
          else if (func) {
            std::printf("  %s\n", func);
          }
          else {
            std::printf("  [unknown frame]\n");
          }
        }
        std::printf("\n");
      }
    }
    else {
      std::printf("No memory leaks detected!\n");
    }

    if (static_ignored > 0) {
      std::printf("Note: %zu static/global allocations ignored\n", static_ignored);
    }

    //std::fflush(stdout);
  #endif
  }

  heap_profiler_t::map_t& heap_profiler_t::get_memory_map() {
    return memory_map;
  }

  heap_profiler_t::set_t& heap_profiler_t::get_memory_set() {
    return memory_set;
  }

  thread_local bool is_inside_allocator = false;

  void heap_profiler_t::track_allocation(void* p, std::size_t n) {
    if (!enabled || is_inside_allocator) {
      return;
    }
    is_inside_allocator = true;

    {
      std::lock_guard<std::mutex> lock(memory_mutex);

      auto check_existing = memory_map.find(p);
      if (check_existing != memory_map.end()) {
        current_allocation_size -= check_existing->second.n;
        memory_set.erase(check_existing->second);
        memory_map.erase(check_existing);
      }

      memory_data_t md;
      md.p = p;
      md.n = n;
      md.alloc_ns = 0; // Or passed via arg if we want, but fine for VMA
    #if defined(fan_std23)
      md.line_data = std::stacktrace::current(0, 20);
    #endif

      memory_map.insert(std::make_pair(p, md));
      memory_set.insert(md);

      current_allocation_size += n;
    }

    is_inside_allocator = false;
  }

  void heap_profiler_t::untrack_allocation(void* p) {
    if (!enabled || is_inside_allocator || !p) {
      return;
    }
    is_inside_allocator = true;

    {
      std::lock_guard<std::mutex> lock(memory_mutex);

      auto found = memory_map.find(p);
      if (found != memory_map.end()) {
        current_allocation_size -= found->second.n;
        memory_set.erase(found->second);
        memory_map.erase(found);
      }
    }

    is_inside_allocator = false;
  }

  void* heap_profiler_t::allocate_memory(std::size_t n) {
    if (!enabled || is_inside_allocator) {
      return std::malloc(n);
    }

    is_inside_allocator = true;

    fan::time::timer timer;
    timer.start();

    void* p = std::malloc(n);

    std::uint64_t elapsed = timer.elapsed();

    is_inside_allocator = false;

    if (!p) {
      throw std::bad_alloc();
    }
    
    track_allocation(p, n);

    // Update alloc_ns
    if (enabled && !is_inside_allocator) {
      is_inside_allocator = true;
      std::lock_guard<std::mutex> lock(memory_mutex);
      auto it = memory_map.find(p);
      if (it != memory_map.end()) {
        memory_set.erase(it->second);
        it->second.alloc_ns = elapsed;
        memory_set.insert(it->second);
      }
      is_inside_allocator = false;
    }

    return p;
  }

  void* heap_profiler_t::reallocate_memory(void* ptr, std::size_t n) {
    if (!enabled || is_inside_allocator) {
      return std::realloc(ptr, n);
    }

    is_inside_allocator = true;

    fan::time::timer timer;
    timer.start();

    void* new_ptr = std::realloc(ptr, n);

    std::uint64_t elapsed = timer.elapsed();

    if (!new_ptr) {
      is_inside_allocator = false;
      return nullptr;
    }

    {
      std::lock_guard<std::mutex> lock(memory_mutex);

      if (ptr) {
        auto found = memory_map.find(ptr);
        if (found != memory_map.end()) {
          current_allocation_size -= found->second.n;
          memory_set.erase(found->second);
          memory_map.erase(found);
        }
      }

      if (new_ptr != ptr) {
        auto check_existing = memory_map.find(new_ptr);
        if (check_existing != memory_map.end()) {
          current_allocation_size -= check_existing->second.n;
          memory_set.erase(check_existing->second);
          memory_map.erase(check_existing);
        }
      }

      current_allocation_size += n;

      memory_data_t md;
      md.p = new_ptr;
      md.n = n;
      md.alloc_ns = elapsed;

    #if defined(fan_std23)
      md.line_data = std::stacktrace::current(0, 20);
    #endif

      memory_map.insert(std::make_pair(new_ptr, md));
      memory_set.insert(md);
    }

    is_inside_allocator = false;
    return new_ptr;
  }

  void heap_profiler_t::deallocate_memory(void* p) {
    if (!p) {
      return;
    }

    if (!enabled || is_inside_allocator) {
      std::free(p);
      return;
    }

    untrack_allocation(p);
    std::free(p);
  }

  bool heap_profiler_t::compare_alloc_size_t::operator()(const memory_data_t& lhs, const memory_data_t& rhs) const {
    if (lhs.n != rhs.n) {
      return lhs.n < rhs.n;
    }
    return lhs.p < rhs.p;
  }

  heap_profiler_t& heap_profiler_t::instance() {
    static heap_profiler_t instance;
    return instance;
  }

  void heap_profiler_t::print_slowest_allocs(int top_count) {
    std::vector<memory_data_t> entries;
    entries.reserve(memory_map.size());

    for (const auto& pair : memory_map) {
      entries.push_back(pair.second);
    }

    std::sort(entries.begin(), entries.end(),
      [](const auto& a, const auto& b) {
      return a.alloc_ns > b.alloc_ns;
    }
    );

    std::size_t count = std::min<std::size_t>(top_count, entries.size());

    std::printf("=== Top %zu Slowest Allocations ===\n", count);

    for (std::size_t i = 0; i < count; ++i) {
      const auto& e = entries[i];

      std::printf("---- #%zu ----\n", i + 1);
      std::printf("Time: %llu ns\n", (unsigned long long)e.alloc_ns);
      std::printf("Size: %zu bytes\n", e.n);
      std::printf("Pointer: %p\n", e.p);

    #if defined(fan_std23)
      std::printf("Stack trace:\n");

      for (const auto& frame : e.line_data) {
        std::string file = frame.source_file();
        std::string func = frame.description();
        auto line = frame.source_line();

        if (!file.empty() && !func.empty()) {
          std::printf("  %s:%u in %s\n", file.c_str(), line, func.c_str());
        }
        else if (!func.empty()) {
          std::printf("  %s\n", func.c_str());
        }
        else {
          std::printf("  [unknown frame]\n");
        }
      }
    #else
      std::printf("  [stacktrace disabled]\n");
    #endif
      std::printf("\n");
    }
  }

}

void* __fan_memory_profile_malloc_cb(std::size_t n) {
  return fan::memory::heap_profiler_t::instance().allocate_memory(n);
}
void* __fan_memory_profile_realloc_cb(void* ptr, std::size_t n) {
  return fan::memory::heap_profiler_t::instance().reallocate_memory(ptr, n);
}
void __fan_memory_profile_free_cb(void* ptr) {
  fan::memory::heap_profiler_t::instance().deallocate_memory(ptr);
}

namespace fan {
  void* memory_profile_malloc_cb(std::size_t n) { return __fan_memory_profile_malloc_cb(n); }
  void* memory_profile_realloc_cb(void* ptr, std::size_t n) { return __fan_memory_profile_realloc_cb(ptr, n); }
  void memory_profile_free_cb(void* ptr) { __fan_memory_profile_free_cb(ptr); }
}