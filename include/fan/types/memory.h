#pragma once

#include _FAN_PATH(types/types.h)
#include _FAN_PATH(types/memory.h)
#include _FAN_PATH(io/file.h)

namespace fan {

  template <typename type_t>
  struct hector_t {

    using value_type = type_t;

    void open() {
      m_size = 0;
      m_capacity = 0;
      ptr = 0;
    }
    void close() {
      resize_buffer(ptr, 0);
    }

    uintptr_t push_back(const type_t& value) {
      m_size++;
      handle_buffer();
      ptr[m_size - 1] = value;
      return m_size - 1;
    }

    void emplace_back(type_t&& value) {
      push_back(value);
    }

    template <typename T>
    void insert(uintptr_t i, T* begin, T* end) {
    #if fan_debug >= fan_debug_low

    #endif
      uintptr_t n = uintptr_t(end - begin);

      m_size += n;
      handle_buffer();

      uintptr_t previous_size = m_size - i - n;

      std::memmove(ptr + i + n, ptr + i, previous_size * sizeof(type_t));
      std::memmove(ptr + i, begin, n * sizeof(T));
    }

    void insert(uintptr_t i, type_t element) {
      uintptr_t n = 1;

      uintptr_t previous_size = m_size - i;

      m_size += n;
      handle_buffer();

      std::memmove(ptr + i + n, ptr + i, previous_size * sizeof(type_t));
      *(ptr + i) = element;
    }

    void erase(uintptr_t i) {
      erase(i, i + 1);
    }

    void erase(uintptr_t begin, uintptr_t end) {
    #if fan_debug >= fan_debug_low

      if (end - begin > m_size) {
        fan::throw_error("invalid erase location 0");
      }

      if (begin > m_size) {
        fan::throw_error("invalid erase location 1");
      }

      if (end > m_size) {
        fan::throw_error("invalid erase location 2");
      }

    #endif
      uintptr_t n = end - begin;

      std::memmove(ptr + begin, ptr + end, (m_size - end) * sizeof(type_t));

      m_size -= n;
      handle_buffer();

      if (m_size == 0) {
        this->clear();
      }
    }

    bool empty() const {
      return !m_size;
    }

    void pop_back() {
      erase(size() - 1);
    }

    void handle_buffer() {

      if (m_size >= m_capacity) {
        m_capacity = m_size + get_buffer_size();
        ptr = (type_t*)resize_buffer(ptr, m_capacity * sizeof(type_t));
      }
      /*if (m_size <= m_capacity) {
        m_capacity = m_size;
        ptr = (type_t*)resize_buffer(ptr, m_capacity * sizeof(type_t));
      }*/
    }

    uintptr_t resize(uintptr_t size) {
      ptr = (type_t*)resize_buffer(ptr, size * sizeof(type_t));
      m_size = size;
      m_capacity = size;
      return m_size - 1;
    }

    void reserve(uintptr_t size) {
      ptr = (type_t*)resize_buffer(ptr, size * sizeof(type_t));
      m_capacity = size;
    }

    type_t operator*() {
      return *ptr;
    }
    const type_t operator*() const {
      return *ptr;
    }

    const type_t& operator[](uintptr_t i) const {
    #if fan_debug >= fan_debug_low
      if (i >= m_size) {
        fan::throw_error("invalid pointer access");
      }
    #endif
      return *(ptr + i);
    }

    type_t& operator[](uintptr_t i) {
    #if fan_debug >= fan_debug_low
      if (i >= m_size) {
        fan::throw_error("invalid pointer access");
      }
    #endif
      return *(ptr + i);
    }

    uintptr_t size() const {
      return m_size;
    }

    uintptr_t capacity() const {
      return m_capacity;
    }

    type_t* data() const {
      return begin();
    }

    type_t* data() {
      return begin();
    }

    type_t* begin() const {
      return ptr;
    }
    type_t* begin() {
      return ptr;
    }

    type_t* end() const {
      return ptr + m_size;
    }
    type_t* end() {
      return ptr + m_size;
    }

    type_t* gb() {
      return ptr;
    }
    type_t* ge() {
      return ptr + m_size - 1;
    }

    void clear() {
      close();
      open();
    }

    static constexpr uintptr_t buffer_increment = 0x100000;

 // protected:

    static type_t* resize_buffer(void* ptr, uintptr_t size) {
      if (ptr) {
        if (size) {
          type_t* rptr = (type_t*)realloc(ptr, size);
        #if fan_debug >= fan_debug_low
          if (rptr == 0) {
            fan::throw_error("realloc failed - ptr:" + std::to_string((uintptr_t)ptr) + " size:" + std::to_string(size));
          }
        #endif
          return rptr;
        }
        else {
          free(ptr);
          return 0;
        }
      }
      else {
        if (size) {
          type_t* rptr = (type_t*)malloc(size);
        #if fan_debug >= fan_debug_low
          if (rptr == 0) {
            fan::throw_error("malloc failed - ptr:" + std::to_string((uintptr_t)ptr) + " size:" + std::to_string(size));
          }
        #endif
          return rptr;
        }
        else {
          return 0;
        }
      }
    }

    constexpr uintptr_t get_buffer_size() {
      constexpr uintptr_t r = buffer_increment / sizeof(type_t);
      if (!r) {
        return 1;
      }
      return r;
    }

    void write_out(fan::io::file::file_t* f) {
      fan::io::file::write(f, &m_size, sizeof(m_size), 1);
      fan::io::file::write(f, ptr, m_size * sizeof(type_t), 1);
    }
    void write_in(fan::io::file::file_t* f) {
      fan::io::file::read(f, &m_size, sizeof(m_size), 1);
      ptr = (type_t*)resize_buffer(ptr, sizeof(type_t) * m_size);
      fan::io::file::read(f, ptr, m_size * sizeof(type_t), 1);
    }

    uintptr_t m_size;
    uintptr_t m_capacity;
    type_t* ptr;
  };

}

namespace fan {

#ifndef A_set_buffer
#define A_set_buffer 512
#endif

  static uint64_t _vector_calculate_buffer(uint64_t size) {
    uint64_t r = A_set_buffer / size;
    if (!r) {
      return 1;
    }
    return r;
  }

  typedef struct {
    fan::hector_t<uint8_t> ptr;
    uint64_t Current, Possible, Type, Buffer;
  }vector_t;
  static void vector_init(vector_t* vec, uint64_t size) {
    vec->Current = 0;
    vec->Possible = 0;
    vec->Type = size;
    vec->Buffer = _vector_calculate_buffer(size);
    vec->ptr.open();
  }

  static void _vector_handle(vector_t* vec) {
    vec->Possible = vec->Current + vec->Buffer;
    vec->ptr.resize(vec->Possible * vec->Type);
  }
  static void VEC_handle(vector_t* vec) {
    if (vec->Current >= vec->Possible) {
      _vector_handle(vec);
    }
  }
  static void vector_handle0(vector_t* vec, uintptr_t amount) {
    vec->Current += amount;
    VEC_handle(vec);
  }

  static void vector_free(vector_t* vec) {
    vec->ptr.close();
    vec->Current = 0;
    vec->Possible = 0;
  }
}