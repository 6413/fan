#pragma once

#include <fan/graphics/opengl/gl_defines.h>

#include <fan/math/random.h>
#include <fan/time/time.h>

#if defined(fan_platform_windows)
  #include <Windows.h>
  #pragma comment(lib, "User32.lib")
  #pragma comment(lib, "Gdi32.lib")

#elif defined(fan_platform_unix)
#endif

#include <unordered_map>

#include <fan/window/window.h>

namespace fan {

  namespace opengl {
    struct opengl_t {
      int major = -1;
      int minor = -1;

      static void* get_proc_address_(const char* name)
      {
        return (void*)glfwGetProcAddress(name);
      }


    #if fan_debug >= fan_debug_high
      // todo implement debug
      
      std::unordered_map<void*, fan::string> function_map;
    #else
      //#define get_proc_address(type, name, internal_) type name = (type)get_proc_address_(#name, internal_);
    #endif
      
      #if fan_debug >= fan_debug_high

      fan::time::clock c;

      void open() {
        #define get_proc_address(type, name) \
              name = (type)get_proc_address_(#name)
        #include "opengl_functions.h"
      }

      void execute_before(const fan::string& function_name) {
        c.start();
      }

      // TODO if function empty probably some WGL/GLX function, initialized in bind window
      void execute_after(const fan::string& function_name) {
        glFlush();
        glFinish();
        auto elapsed = c.elapsed();
        if (elapsed > 1000000) {
          fan::print_no_space("slow function call ns:", elapsed, ", function::" + function_name);
        }
      }
      
      //efine call_opengl

      template <typename T, typename... T2>
      constexpr auto call(const T& t, T2&&... args) {
          if constexpr (std::is_same<std::invoke_result_t<T, T2...>, void>::value) {
              t(std::forward<T2>(args)...);
          } else {
              return t(std::forward<T2>(args)...);
          }
      }
      #else

      template <typename T, typename ...T2>
      constexpr auto call(const T& t, T2&&... args) {
        if constexpr (std::is_same<fan::return_type_of_t<T>, void>::value) {
          t(args...);
        }
        else {
          return t(args...);
        }
      }

      #endif

      #define get_proc_address(type, name) \
              type name
      #include "opengl_functions.h"
      
    };
  }

}