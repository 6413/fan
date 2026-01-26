module;

#include <fan/utility.h>

#include <cstddef>
#include <map>
#include <set>
#include <mutex>
#include <memory>
#include <stacktrace>

export module fan.memory;

export namespace fan::memory {
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
      custom_alloc_t(custom_alloc_t<U> const& u) : std::allocator<T>(u) {}
      pointer allocate(size_type size, const void* = 0) {
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
      uint64_t alloc_ns = 0;
#if defined(fan_std23)
      std::stacktrace line_data;
#endif
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
    
    ~heap_profiler_t();
    map_t& get_memory_map();
    set_t& get_memory_set();
    void* allocate_memory(std::size_t n);
    void* reallocate_memory(void* ptr, std::size_t n);
    void deallocate_memory(void* p);
    void print_slowest_allocs(int top_count);
    
    heap_profiler_t() = default;
    heap_profiler_t(const heap_profiler_t&) = delete;
    heap_profiler_t& operator=(const heap_profiler_t&) = delete;
    
    std::mutex memory_mutex;
    map_t memory_map;
    set_t memory_set;
    uint64_t current_allocation_size = 0;
    bool enabled = false;
    bool print_leaks = true;
  };
}