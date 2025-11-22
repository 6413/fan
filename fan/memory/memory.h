// include before all, because it uses some macros which override default allocation functions
/*
* include these before using
* #include <fan/utility.h>

  #include <source_location>
  #include <set>
  #include <stacktrace>
  #include <map>
  
*/

#if defined(fan_std23)

export namespace fan {
  struct heap_profiler_t {
    ~heap_profiler_t() {
      if (!enabled) {
        return;
      }
      enabled = false;
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
          if (func && strstr(func, "main")) {
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
        // Print only leaks that have "main" in their stack trace
        for (auto it = memory_set.rbegin(); it != memory_set.rend(); ++it) {
          const auto& leak = *it;

          bool is_from_main = false;
          for (const auto& frame : leak.line_data) {
            const char* func = frame.description().c_str();
            if (func && strstr(func, "main")) {
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
    }

    template<typename T>
    struct custom_alloc_t : std::allocator<T> {
      typedef T* pointer;
      typedef typename std::allocator<T>::size_type size_type;

      template<typename U>
      struct rebind {
        typedef custom_alloc_t<U> other;
      };

      custom_alloc_t() {}

      template<typename U>
      custom_alloc_t(custom_alloc_t<U> const& u)
        :std::allocator<T>(u) {}

      pointer allocate(size_type size,
        const void* = 0) {
        void* p = std::malloc(size * sizeof(T));
        if (p == 0) {
          throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
      }

      void deallocate(pointer p, size_type) {
        std::free(p);
      }
    };

    struct memory_data_t {
      std::size_t n = 0;
      void* p = 0;
      std::stacktrace line_data;
    };

    struct compare_alloc_size_t {
      bool operator()(const memory_data_t& lhs, const memory_data_t& rhs) const {
        if (lhs.n != rhs.n) {
          return lhs.n < rhs.n;
        }
        return lhs.p < rhs.p;
      }
    };

    using map_t = std::map<void*, memory_data_t, std::less<void*>, custom_alloc_t<std::pair<void* const, memory_data_t>>>;
    using set_t = std::set<memory_data_t, compare_alloc_size_t, custom_alloc_t<memory_data_t>>;

    static heap_profiler_t& instance() {
      static heap_profiler_t instance;
      return instance;
    }

    map_t& get_memory_map() {
      return memory_map;
    }

    set_t& get_memory_set() {
      return memory_set;
    }

    void* allocate_memory(std::size_t n) {
      bool was_enabled = enabled;
      enabled = false;

      void* p = std::malloc(n);
      if (!p) {
        enabled = was_enabled;
        throw std::bad_alloc();
      }

      if (was_enabled) {
        memory_data_t md;
        md.p = p;
        md.n = n;

        md.line_data = std::stacktrace::current(0, 20);

        auto result_map = memory_map.insert(std::make_pair(p, md));
        if (!result_map.second) {
          printf("duplicate insertion in map for pointer: %p\n", p);
        }

        auto result_set = memory_set.insert(md);
        if (!result_set.second) {
          printf("duplicate insertion in set for pointer: %p\n", p);
        }

        current_allocation_size += n;
        enabled = true;
      }

      return p;
    }

    void* reallocate_memory(void* ptr, std::size_t n) {
      bool was_enabled = enabled;
      enabled = false;

      auto found = memory_map.find(ptr);
      if (found == memory_map.end()) {
        void* result = std::realloc(ptr, n);
        enabled = was_enabled;
        return result;
      }
      else {
        void* new_ptr = std::realloc(ptr, n);
        if (!new_ptr) {
          enabled = was_enabled;
          return nullptr;
        }

        if (was_enabled) {
          current_allocation_size -= found->second.n;
          current_allocation_size += n;

          memory_set.erase(found->second);
          memory_map.erase(found);

          memory_data_t md;
          md.p = new_ptr;
          md.n = n;
          md.line_data = std::stacktrace::current(0, 20);

          auto result_map = memory_map.insert(std::make_pair(new_ptr, md));
          if (!result_map.second) {
            printf("duplicate insertion in map for pointer: %p\n", new_ptr);
          }

          auto result_set = memory_set.insert(md);
          if (!result_set.second) {
            printf("duplicate insertion in set for pointer: %p\n", new_ptr);
          }

          enabled = true;
        }

        return new_ptr;
      }
    }

    void deallocate_memory(void* p) {
      if (enabled) {
        if (p) {
          auto found = memory_map.find(p);
          if (found == memory_map.end()) {
            printf("freeing non-mapped memory: %p\n", p);
          }
          else {
            current_allocation_size -= found->second.n;
            memory_set.erase(found->second);
            memory_map.erase(found);
          }
        }
      }
      std::free(p);
    }

    heap_profiler_t() = default;
    heap_profiler_t(const heap_profiler_t&) = delete;
    heap_profiler_t& operator=(const heap_profiler_t&) = delete;

    map_t memory_map;
    set_t memory_set;
    uint64_t current_allocation_size = 0;
    bool enabled = false;
    bool print_leaks = true;
  };
}


#ifndef __generic_malloc
  #define __generic_malloc(n) fan::heap_profiler_t::instance().allocate_memory(n)
#endif

#ifndef __generic_realloc
  #define __generic_realloc(ptr, n) fan::heap_profiler_t::instance().reallocate_memory(ptr, n)
#endif

#ifndef __generic_free
  #define __generic_free(ptr) fan::heap_profiler_t::instance().deallocate_memory(ptr)
#endif

#define fan_track_allocations() \
  void* operator new(std::size_t n) { return fan::heap_profiler_t::instance().allocate_memory(n); } \
  void operator delete(void* p) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
  \
  \
  void* operator new[](std::size_t n) { return fan::heap_profiler_t::instance().allocate_memory(n); } \
  void operator delete[](void* p) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
  \
  \
  void operator delete(void* p, std::size_t) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
  void operator delete[](void* p, std::size_t) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
  \
  \
  void* operator new(std::size_t n, const std::nothrow_t&) noexcept { \
    try { return fan::heap_profiler_t::instance().allocate_memory(n); } \
    catch(const std::bad_alloc&) { return nullptr; } \
  } \
  void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { \
    try { return fan::heap_profiler_t::instance().allocate_memory(n); } \
    catch(const std::bad_alloc&) { return nullptr; } \
  } \
  void operator delete(void* p, const std::nothrow_t&) noexcept { \
    fan::heap_profiler_t::instance().deallocate_memory(p); \
  } \
  void operator delete[](void* p, const std::nothrow_t&) noexcept { \
    fan::heap_profiler_t::instance().deallocate_memory(p); \
  }

#define fan_track_allocations_start() \
  fan::heap_profiler_t::instance().enabled = true;
#endif