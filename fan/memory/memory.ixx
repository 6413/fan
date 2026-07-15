module;

#include <fan/utility.h>

export module fan.memory;

import std;

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
      std::uint64_t alloc_ns = 0;
#if defined(fan_std23)
      std::stacktrace line_data;
#endif
    };
    
    struct compare_alloc_size_t {
      bool operator()(const memory_data_t& lhs, const memory_data_t& rhs) const;
    };
    
    using map_t = std::map<void*, memory_data_t, std::less<void*>, custom_alloc_t<std::pair<void* const, memory_data_t>>>;
    using set_t = std::set<memory_data_t, compare_alloc_size_t, custom_alloc_t<memory_data_t>>;
    
    static heap_profiler_t& instance();
    
    ~heap_profiler_t();
    map_t& get_memory_map();
    set_t& get_memory_set();
    void* allocate_memory(std::size_t n);
    void* reallocate_memory(void* ptr, std::size_t n);
    void deallocate_memory(void* p);
    void track_allocation(void* p, std::size_t n);
    void untrack_allocation(void* p);
    void print_slowest_allocs(int top_count);
    
    heap_profiler_t() = default;
    heap_profiler_t(const heap_profiler_t&) = delete;
    heap_profiler_t& operator=(const heap_profiler_t&) = delete;
    
    std::mutex memory_mutex;
    map_t memory_map;
    set_t memory_set;
    std::uint64_t current_allocation_size = 0;
    bool enabled = false;
    bool print_leaks = true;
  };
}

export namespace fan::memory {
  constexpr std::uint16_t read_le16(const std::uint8_t* p) { return std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8); }
  constexpr void write_le16(std::uint8_t* p, std::uint16_t v) { p[0] = std::uint8_t(v); p[1] = std::uint8_t(v >> 8); }
  constexpr std::uint32_t read_le32(const std::uint8_t* p) { return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24); }
  constexpr void write_le32(std::uint8_t* p, std::uint32_t v) { p[0] = std::uint8_t(v); p[1] = std::uint8_t(v >> 8); p[2] = std::uint8_t(v >> 16); p[3] = std::uint8_t(v >> 24); }
  constexpr std::uint64_t read_le64(const std::uint8_t* p) { return std::uint64_t(read_le32(p)) | (std::uint64_t(read_le32(p + 4)) << 32); }
  constexpr void write_le64(std::uint8_t* p, std::uint64_t v) { write_le32(p, std::uint32_t(v)); write_le32(p + 4, std::uint32_t(v >> 32)); }
}