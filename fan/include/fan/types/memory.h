#pragma once

#include <fan/types/types.hpp>

#include <fan/types/memory.h>

namespace fan {

  template <typename type_t>
  struct hector_t{

    void open(){
      m_size = 0;
      m_capacity = 0;
      ptr = 0;
    }
    void close(){
      resize_buffer(ptr, 0);
      open();
    }

    void push_back(const type_t& value) {
      m_size++;
      handle_buffer();
      ptr[m_size - 1] = value;
    }

    void emplace_back(type_t&& value) {
      push_back(value);
    }

    template <typename T>
    void insert(type_t* i, T* begin, T* end) {
    #ifdef fan_debug == fan_debug_soft

    #endif
      uintptr_t n = uintptr_t(end - begin);

      uintptr_t index = uintptr_t(i - ptr);

      uintptr_t previous_size = m_size - index;

      m_size += n;
      handle_buffer();

      std::memmove(ptr + index + n, ptr + index, previous_size * sizeof(type_t));
      std::memmove(ptr + index, begin, n * sizeof(T));
    }

    void insert(type_t* i, type_t element) {
    #ifdef fan_debug == fan_debug_soft
        if (uintptr_t(i - ptr) > m_size) {
          fan::throw_error("invalid erase location 0");
        }
    #endif
      uintptr_t n = 1;

      uintptr_t index = uintptr_t(i - ptr);

      uintptr_t previous_size = m_size - index;

      m_size += n;
      handle_buffer();

      std::memmove(ptr + index + n, ptr + index, previous_size * sizeof(type_t));
      *(ptr + index) = element;
    }

    void erase(uintptr_t i) {
      erase(i, i + 1);
    }

    void erase(uintptr_t begin, uintptr_t end) {
      #ifdef fan_debug == fan_debug_soft

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

      uintptr_t previous_size = m_size - n;

      if (m_size > n) {
        for (int i = 0; i < n; i++) {
          ptr[begin + i] = ptr[n + i];
        }
      }

      m_size -= n;
      handle_buffer();

    }

    bool empty() const {
      return !m_size;
    }

    void pop_back() {
      erase(size() - 1);
    }

    void handle_buffer() {

      if(m_size >= m_capacity){
        m_capacity = m_size + get_buffer_size();
        ptr = (type_t*)resize_buffer(ptr, m_capacity * sizeof(type_t));
      }
    }

    void resize(uintptr_t size) {
      ptr = (type_t*)resize_buffer(ptr, size * sizeof(type_t));
      m_size = size;
      m_capacity = size;
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
      return *(ptr + i);
    }

    type_t& operator[](uintptr_t i) {
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
    
    void clear() {
      close();
    }

    static constexpr uintptr_t buffer_increment = 0xfff;

  protected:

    static uint8_t *resize_buffer(void *ptr, uintptr_t size){
      if(ptr){
        if(size){
          void *rptr = (void *)realloc(ptr, size);
          if(rptr == 0){
            fan::throw_error("realloc failed - ptr:" + std::to_string((uintptr_t)ptr) + " size:" + std::to_string(size));
          }
          return (uint8_t *)rptr;
        }
        else{
          free(ptr);
          return 0;
        }
      }
      else{
        if(size){
          void *rptr = (void *)malloc(size);
          if(rptr == 0){
            fan::throw_error("malloc failed - ptr:" + std::to_string((uintptr_t)ptr) + " size:" + std::to_string(size));
          }
          return (uint8_t *)rptr;
        }
        else{
          return 0;
        }
      }
    }

    constexpr uintptr_t get_buffer_size(){
      constexpr uintptr_t r = buffer_increment / sizeof(type_t);
      if(!r){
        return 1;
      }
      return r;
    }

    uintptr_t m_size;
    uintptr_t m_capacity;
    type_t *ptr;
  };

}

namespace fan {

#ifndef A_set_buffer
#define A_set_buffer 512
#endif

  static uint64_t _vector_calculate_buffer(uint64_t size){
    uint64_t r = A_set_buffer / size;
    if(!r){
      return 1;
    }
    return r;
  }

  typedef struct{
    fan::hector_t<uint8_t> ptr;
    uint64_t Current, Possible, Type, Buffer;
  }vector_t;
  static void vector_init(vector_t *vec, uint64_t size){
    vec->Current = 0;
    vec->Possible = 0;
    vec->Type = size;
    vec->Buffer = _vector_calculate_buffer(size);
    vec->ptr.open();
  }

  static void _vector_handle(vector_t *vec){
    vec->Possible = vec->Current + vec->Buffer;
    vec->ptr.resize(vec->Possible * vec->Type);
  }
  static void VEC_handle(vector_t *vec){
    if(vec->Current >= vec->Possible){
      _vector_handle(vec);
    }
  }
  static void vector_handle0(vector_t *vec, uintptr_t amount){
    vec->Current += amount;
    VEC_handle(vec);
  }

  static void vector_free(vector_t *vec){
    vec->ptr.close();
    vec->Current = 0;
    vec->Possible = 0;
  }
}