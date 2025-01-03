#pragma once

#include <fan/graphics/opengl/gl_defines.h>

//#define debug_glcall_timings
#if defined(debug_glcall_timings)
  #include <unordered_map>
  #include <fan/time/time.h>
#endif

#if defined(fan_platform_windows)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define GLFW_NATIVE_INCLUDE_NONE
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <utility>

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

      #if defined(debug_glcall_timings)
      std::unordered_map<void*, std::string> function_map;
      fan::time::clock c;
      #endif

      void open() {
#if !defined(debug_glcall_timings)
        #define get_proc_address(type, name) \
              name = (type)get_proc_address_(#name)
#else
              #define get_proc_address(type, name) \
                name = (type)get_proc_address_(#name); \
              function_map.insert(std::make_pair((void*)name, #name))
#endif
        
        #include "opengl_functions.h"
      }

      #if defined(debug_glcall_timings)
      void execute_before(const fan::string& function_name) {
        c.start();
      }

      // TODO if function empty probably some WGL/GLX function, initialized in bind window
      void execute_after(const fan::string& function_name) {
        glFlush();
        glFinish();
        auto elapsed = c.elapsed();
        if (elapsed > 1084040) {
          fan::print_no_space("slow function call ns:", elapsed, ", function::" + function_name);
        }
      }
#endif
      //efine call_opengl

      template <typename T, typename... T2>
      constexpr auto call(const T& t, T2&&... args) {
#if defined(debug_glcall_timings)
        execute_before(function_map[(void*)t]);
#endif
        if constexpr (std::is_same<std::invoke_result_t<T, T2...>, void>::value) {
          t(std::forward<T2>(args)...);
#if defined(debug_glcall_timings)
          execute_after(function_map[(void*)t]);
#endif  
        }
        else {
          auto r = t(args...);
#if defined(debug_glcall_timings)
          execute_after(function_map[(void*)t]);
#endif
          return r;
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