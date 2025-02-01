// include before all, because it uses some macros which override default allocation functions
#pragma once

#include <fan/types/types.h>

#if defined(fan_std23)

#define fan_tracking_allocations

#include <source_location>
#include <set>
#include <map>
#include <stacktrace>

namespace fan {
  struct heap_profiler_t {

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
    using stacktrace_t = std::basic_stacktrace<custom_alloc_t<std::stacktrace_entry>>;
    struct memory_data_t {
      std::size_t n = 0;
      void* p = 0;
      stacktrace_t line_data;
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
      void* p = std::malloc(n);
      if (!p) {
        throw std::bad_alloc();
      }
      if (enabled) {
        memory_data_t md;
        md.p = p;
        md.n = n;
        md.line_data = std::basic_stacktrace<custom_alloc_t<std::stacktrace_entry>>::current(0, 20);

        auto result_map = memory_map.insert(std::make_pair(p, md));
        if (!result_map.second) {
          printf("duplicate insertion in map for pointer: %p\n", p);
        }

        auto result_set = memory_set.insert(md);
        if (!result_set.second) {
          printf("duplicate insertion in set for pointer: %p\n", p);
        }

        current_allocation_size += n;
      }
      return p;
    }
    void* reallocate_memory(void* ptr, std::size_t n) {
      auto found = memory_map.find(ptr);
      if (found == memory_map.end()) {
        return std::realloc(ptr, n);
      }
      else {
        void* new_ptr = std::realloc(ptr, n);
        if (!new_ptr) {
          throw std::bad_alloc();
        }

        current_allocation_size -= found->second.n;
        current_allocation_size += n;

        memory_set.erase(found->second);
        memory_map.erase(found);

        memory_data_t md;
        md.p = new_ptr;
        md.n = n;
        md.line_data = std::basic_stacktrace<custom_alloc_t<std::stacktrace_entry>>::current(0, 20);

        auto result_map = memory_map.insert(std::make_pair(new_ptr, md));
        if (!result_map.second) {
          printf("duplicate insertion in map for pointer: %p\n", new_ptr);
        }

        auto result_set = memory_set.insert(md);
        if (!result_set.second) {
          printf("duplicate insertion in set for pointer: %p\n", new_ptr);
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
  bool fan_heap_profiler_init__ = []{ fan::heap_profiler_t::instance().enabled = true; return 0; }(); \
  void* operator new(std::size_t n) { return fan::heap_profiler_t::instance().allocate_memory(n); } \
  void operator delete(void* p) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
 \
  void* operator new[](std::size_t n) { return fan::heap_profiler_t::instance().allocate_memory(n); } \
  void operator delete[](void* p) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
 \
  void operator delete(void* p, std::size_t) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); } \
  void operator delete[](void* p, std::size_t) noexcept { fan::heap_profiler_t::instance().deallocate_memory(p); }

#endif