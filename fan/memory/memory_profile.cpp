import fan.memory;

import std;

namespace fan {
  void* memory_profile_malloc_cb(std::size_t n) { return fan::memory::heap_profiler_t::instance().allocate_memory(n); }
  void* memory_profile_realloc_cb(void* ptr, std::size_t n) { return fan::memory::heap_profiler_t::instance().reallocate_memory(ptr, n); }
  void memory_profile_free_cb(void* ptr) { fan::memory::heap_profiler_t::instance().deallocate_memory(ptr); }
}