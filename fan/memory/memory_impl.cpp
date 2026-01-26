module;

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <mutex>

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

    printf("=== Memory Leak Report ===\n");

    size_t real_leaks = 0;
    size_t real_leak_bytes = 0;
    size_t static_ignored = 0;

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

    printf("Total leaked memory: %llu bytes\n", (unsigned long long)real_leak_bytes);
    printf("Number of leaked allocations: %zu\n", real_leaks);

    if (real_leaks > 0) {
      printf("\nLeak details:\n");
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

        printf("Leaked %zu bytes at address %p\n", leak.n, leak.p);
        printf("Stack trace:\n");
        for (const auto& frame : leak.line_data) {
          const char* file = frame.source_file().c_str();
          const char* func = frame.description().c_str();
          auto line = frame.source_line();
          if (file && func) {
            printf("  %s:%u in %s\n", file, line, func);
          }
          else if (func) {
            printf("  %s\n", func);
          }
          else {
            printf("  [unknown frame]\n");
          }
        }
        printf("\n");
      }
    }
    else {
      printf("No memory leaks detected!\n");
    }

    if (static_ignored > 0) {
      printf("Note: %zu static/global allocations ignored\n", static_ignored);
    }

    fflush(stdout);
  #endif
  }

  heap_profiler_t::map_t& heap_profiler_t::get_memory_map() {
    return memory_map;
  }

  heap_profiler_t::set_t& heap_profiler_t::get_memory_set() {
    return memory_set;
  }

  void* heap_profiler_t::allocate_memory(std::size_t n) {
    bool was_enabled = enabled;
    if (!was_enabled) {
      return std::malloc(n);
    }

    enabled = false;

    fan::time::timer timer;
    timer.start();

    void* p = std::malloc(n);

    uint64_t elapsed = timer.elapsed();

    if (!p) {
      enabled = was_enabled;
      throw std::bad_alloc();
    }
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
      md.alloc_ns = elapsed;
    #if defined(fan_std23)
      md.line_data = std::stacktrace::current(0, 20);
    #endif

      memory_map.insert(std::make_pair(p, md));
      memory_set.insert(md);

      current_allocation_size += n;
    }

    enabled = was_enabled;
    return p;
  }

  void* heap_profiler_t::reallocate_memory(void* ptr, std::size_t n) {
    bool was_enabled = enabled;

    if (!was_enabled) {
      return std::realloc(ptr, n);
    }

    enabled = false;

    fan::time::timer timer;
    timer.start();

    void* new_ptr = nullptr;

    {
      std::lock_guard<std::mutex> lock(memory_mutex);

      auto found = memory_map.find(ptr);
      if (found == memory_map.end()) {
        new_ptr = std::realloc(ptr, n);
        enabled = was_enabled;
        return new_ptr;
      }

      new_ptr = std::realloc(ptr, n);
      uint64_t elapsed = timer.elapsed();

      if (!new_ptr) {
        enabled = was_enabled;
        return nullptr;
      }

      current_allocation_size -= found->second.n;
      memory_set.erase(found->second);
      memory_map.erase(found);

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

    enabled = was_enabled;
    return new_ptr;
  }

  void heap_profiler_t::deallocate_memory(void* p) {
    if (!p) {
      return;
    }

    bool was_enabled = enabled;

    if (!was_enabled) {
      std::free(p);
      return;
    }

    enabled = false;

    {
      std::lock_guard<std::mutex> lock(memory_mutex);

      auto found = memory_map.find(p);
      if (found != memory_map.end()) {
        current_allocation_size -= found->second.n;
        memory_set.erase(found->second);
        memory_map.erase(found);
      }
    }

    std::free(p);
    enabled = was_enabled;
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

    size_t count = std::min<size_t>(top_count, entries.size());

    printf("=== Top %zu Slowest Allocations ===\n", count);

    for (size_t i = 0; i < count; ++i) {
      const auto& e = entries[i];

      printf("---- #%zu ----\n", i + 1);
      printf("Time: %llu ns\n", (unsigned long long)e.alloc_ns);
      printf("Size: %zu bytes\n", e.n);
      printf("Pointer: %p\n", e.p);

    #if defined(fan_std23)
      printf("Stack trace:\n");

      for (const auto& frame : e.line_data) {
        std::string file = frame.source_file();
        std::string func = frame.description();
        auto line = frame.source_line();

        if (!file.empty() && !func.empty()) {
          printf("  %s:%u in %s\n", file.c_str(), line, func.c_str());
        }
        else if (!func.empty()) {
          printf("  %s\n", func.c_str());
        }
        else {
          printf("  [unknown frame]\n");
        }
      }
    #else
      printf("  [stacktrace disabled]\n");
    #endif
      printf("\n");
    }
  }

}