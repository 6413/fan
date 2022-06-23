#pragma once

#include _FAN_PATH(graphics/opengl/gl_defines.h)

#include _FAN_PATH(math/random.h)

#if defined(fan_platform_windows)
  #include <Windows.h>
  #pragma comment(lib, "User32.lib")
  #pragma comment(lib, "Gdi32.lib")

#elif defined(fan_platform_unix)
#endif

namespace fan {

  namespace opengl {

    struct internal_t {

      #if defined(fan_platform_windows)
        static constexpr const char* shared_library = "opengl32.lib";
      #elif defined(fan_platform_unix)
        static constexpr const char* shared_library = "libGL.so.1";
      #endif

      

      #if defined(fan_platform_windows)
        
      #elif defined(fan_platform_unix)
        static void open_lib_handle(void** handle) {
          *handle = dlopen(shared_library, RTLD_LAZY);
          #if fan_debug >= fan_debug_low
          if (*handle == nullptr) {
              fan::throw_error(dlerror());
          }
          #endif
        }
        static void close_lib_handle(void** handle) {
          #if fan_debug >= fan_debug_low
          auto error =
          #endif
          dlclose(*handle);
          #if fan_debug >= fan_debug_low
          if (error != 0) {
              fan::throw_error(dlerror());
          }
          #endif
        }

        static void* get_lib_proc(void** handle, const char* name) {
          void* result = dlsym(*handle, name);
          #if fan_debug >= fan_debug_low
          if (result == nullptr) {
              dlerror();
              dlsym(*handle, name);
              auto error = dlerror();
              if (error != nullptr) {
              fan::throw_error(error);
              }
          }
          #endif
          return result;
        }
      #endif

      struct properties_t {
        #if defined(fan_platform_windows)
          HWND hwnd;
          HDC hdc;
          HGLRC context;
        #elif defined(fan_platform_unix)

        #endif
      };

      void open(properties_t* p) {
        #if defined(fan_platform_windows)
          // create dummy window to initialize functions thank u microsoft
          // generate random class name to dont collide with other window classes xd
          auto str = fan::random::string(15);
          WNDCLASSA window_class = {
            .style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
            .lpfnWndProc = DefWindowProcA,
            .hInstance = GetModuleHandle(0),
            .lpszClassName = str.c_str(),
          };

          if (!RegisterClassA(&window_class)) {
            fan::print("failed to register window");
            exit(1);
          }

          p->hwnd = CreateWindowExA(
            0,
            window_class.lpszClassName,
            "temp_window",
            0,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            0,
            0,
            window_class.hInstance,
            0);

          if (!p->hwnd) {
            fan::print("failed to create window");
            exit(1);
          }

          p->hdc = GetDC(p->hwnd);

          PIXELFORMATDESCRIPTOR pfd = {
            sizeof(PIXELFORMATDESCRIPTOR),
            1,
            PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,    // Flags
            PFD_TYPE_RGBA,        // The kind of framebuffer. RGBA or palette.
            32,                   // Colordepth of the framebuffer.
            0, 0, 0, 0, 0, 0,
            0,
            0,
            0,
            0, 0, 0, 0,
            24,                   // Number of bits for the depthbuffer
            8,                    // Number of bits for the stencilbuffer
            0,                    // Number of Aux buffers in the framebuffer.
            PFD_MAIN_PLANE,
            0,
            0, 0, 0
          };

          int pixel_format = ChoosePixelFormat(p->hdc, &pfd);
          if (!pixel_format) {
            fan::throw_error("failed to choose pixel format");
          }
          if (!SetPixelFormat(p->hdc, pixel_format, &pfd)) {
            fan::throw_error("failed to set pixel format");
          }

          p->context = wglCreateContext(p->hdc);
          if (!p->context) {
            fan::throw_error("failed to create context");
          }

          if (!wglMakeCurrent(p->hdc, p->context)) {
            fan::throw_error("failed to make current");
          }

          wglCreateContextAttribsARB = (decltype(wglCreateContextAttribsARB))wglGetProcAddress("wglCreateContextAttribsARB");
          wglChoosePixelFormatARB = (decltype(wglChoosePixelFormatARB))wglGetProcAddress("wglChoosePixelFormatARB");
          wglSwapIntervalEXT = (decltype(wglSwapIntervalEXT))wglGetProcAddress("wglSwapIntervalEXT");

        #elif defined(fan_platform_unix)

          void* lib_handle;
          open_lib_handle(&lib_handle);
          glXGetProcAddress = (decltype(glXGetProcAddress))get_lib_proc(&lib_handle, "glXGetProcAddress");
          close_lib_handle(&lib_handle);
          glXMakeCurrent = (decltype(glXMakeCurrent))glXGetProcAddress((const fan::opengl::GLubyte*)"glXMakeCurrent");
          glXGetCurrentDrawable = (decltype(glXGetCurrentDrawable))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetCurrentDrawable");
          glXSwapIntervalEXT = (decltype(glXSwapIntervalEXT))glXGetProcAddress((const fan::opengl::GLubyte*)"glXSwapIntervalEXT");
          glXDestroyContext = (decltype(glXDestroyContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXDestroyContext");
          glXChooseFBConfig = (decltype(glXChooseFBConfig))glXGetProcAddress((const fan::opengl::GLubyte*)"glXChooseFBConfig");
          glXGetVisualFromFBConfig = (decltype(glXGetVisualFromFBConfig))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetVisualFromFBConfig");
          glXQueryVersion = (decltype(glXQueryVersion))glXGetProcAddress((const fan::opengl::GLubyte*)"glXQueryVersion");
          glXGetFBConfigAttrib = (decltype(glXGetFBConfigAttrib))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetFBConfigAttrib");
          glXQueryExtensionsString = (decltype(glXQueryExtensionsString))glXGetProcAddress((const fan::opengl::GLubyte*)"glXQueryExtensionsString");
          glXGetCurrentContext = (decltype(glXGetCurrentContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXGetCurrentContext");
          glXSwapBuffers = (decltype(glXSwapBuffers))glXGetProcAddress((const fan::opengl::GLubyte*)"glXSwapBuffers");
          glXCreateNewContext = (decltype(glXCreateNewContext))glXGetProcAddress((const fan::opengl::GLubyte*)"glXCreateNewContext");
          glXCreateContextAttribsARB = (decltype(glXCreateContextAttribsARB))glXGetProcAddress((const fan::opengl::GLubyte*)"glXCreateContextAttribsARB");

        #endif
      }
      void close(const properties_t* p) {
        #if defined(fan_platform_windows)
          wglMakeCurrent(p->hdc, 0);
          wglDeleteContext(p->context);
          ReleaseDC(p->hwnd, p->hdc);
          DestroyWindow(p->hwnd);
        #elif defined(fan_platform_unix)

        #endif
      }

      #if defined(fan_platform_windows)
        fan::opengl::wgl::PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglChoosePixelFormatARB;
        fan::opengl::wgl::PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;
        fan::opengl::wgl::PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
      #elif defined(fan_platform_unix)
        fan::opengl::glx::PFNGLXGETPROCADDRESSPROC glXGetProcAddress;
        fan::opengl::glx::PFNGLXMAKECURRENTPROC glXMakeCurrent;
        fan::opengl::glx::PFNGLXGETCURRENTDRAWABLEPROC glXGetCurrentDrawable;
        fan::opengl::glx::PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT;
        fan::opengl::glx::PFNGLXDESTROYCONTEXTPROC glXDestroyContext;
        fan::opengl::glx::PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig;
        fan::opengl::glx::PFNGLXGETVISUALFROMFBCONFIGPROC glXGetVisualFromFBConfig;
        fan::opengl::glx::PFNGLXQUERYVERSIONPROC glXQueryVersion;
        fan::opengl::glx::PFNGLXGETFBCONFIGATTRIBPROC glXGetFBConfigAttrib;
        fan::opengl::glx::PFNGLXQUERYEXTENSIONSSTRINGPROC glXQueryExtensionsString;
        fan::opengl::glx::PFNGLXGETCURRENTCONTEXTPROC glXGetCurrentContext;
        fan::opengl::glx::PFNGLXSWAPBUFFERSPROC glXSwapBuffers;
        fan::opengl::glx::PFNGLXCREATENEWCONTEXTPROC glXCreateNewContext;
        fan::opengl::glx::PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB;
      #endif

    };

    inline bool opengl_initialized = false;

    struct opengl_t {

    private:

      static void* get_proc_address_(const char* name, internal_t* internal)
      {
        #if defined(fan_platform_windows)
          void *p = (void *)wglGetProcAddress(name);
        if(p == 0 ||
          (p == (void*)0x1) || (p == (void*)0x2) || (p == (void*)0x3) ||
          (p == (void*)-1) )
        {
          HMODULE module = LoadLibraryA("opengl32.dll");
          p = (void *)GetProcAddress(module, name);
        }

        #if fan_debug >= fan_debug_low
          if (p == nullptr) {
            fan::throw_error(std::string("failed to load proc:") + name + ", with error:" + std::to_string(GetLastError()));
          }
        #endif

        return p;

        #elif defined(fan_platform_unix)

        return (void*)internal->glXGetProcAddress((const GLubyte*)name);

        #endif
      }


    public:

    #if fan_debug >= fan_debug_high
      #define get_proc_address(name, internal_) \
        name = (decltype(name))get_proc_address_(#name, internal_); \
        function_map.insert(std::make_pair((void*)name, #name));
      std::unordered_map<void*, std::string> function_map;
    #else
      #define get_proc_address(name, internal_) name = (decltype(name))get_proc_address_(#name, internal_);
    #endif
      

      void open() {

        if (opengl_initialized) {
          return;
        };
        internal_t::properties_t p;
        memset(&p, 0, sizeof(p));

        internal.open(&p);

        get_proc_address(glGetString, &internal);
        get_proc_address(glViewport, &internal);
        get_proc_address(glBlendFunc, &internal);
        get_proc_address(glCreateVertexArrays, &internal);
        get_proc_address(glDeleteVertexArrays, &internal);
        get_proc_address(glBindVertexArray, &internal);
        get_proc_address(glGenBuffers, &internal);
        get_proc_address(glBindBuffer, &internal);
        get_proc_address(glBufferData, &internal);
        get_proc_address(glDeleteBuffers, &internal);
        get_proc_address(glBufferSubData, &internal);
        get_proc_address(glGetBufferSubData, &internal);
        get_proc_address(glClear, &internal);
        get_proc_address(glClearColor, &internal);
        get_proc_address(glDebugMessageCallback, &internal);
        get_proc_address(glEnable, &internal);
        get_proc_address(glDisable, &internal);
        get_proc_address(glDrawArrays, &internal);
        get_proc_address(glEnableVertexAttribArray, &internal);
        get_proc_address(glGetAttribLocation, &internal);
        get_proc_address(glGetBufferParameteriv, &internal);
        get_proc_address(glGetIntegerv, &internal);
        get_proc_address(glVertexAttribPointer, &internal);
        get_proc_address(glGenTextures, &internal);
        get_proc_address(glDeleteTextures, &internal);
        get_proc_address(glBindTexture, &internal);
        get_proc_address(glGetTexImage, &internal);
        get_proc_address(glTexImage2D, &internal);
        get_proc_address(glTexParameteri, &internal);
        get_proc_address(glActiveTexture, &internal);
        get_proc_address(glAttachShader, &internal);
        get_proc_address(glCreateShader, &internal);
        get_proc_address(glDeleteShader, &internal);
        get_proc_address(glCompileShader, &internal);
        get_proc_address(glCreateProgram, &internal);
        get_proc_address(glDeleteProgram, &internal);
        get_proc_address(glGenerateMipmap, &internal);
        get_proc_address(glGetProgramInfoLog, &internal);
        get_proc_address(glGetProgramiv, &internal);
        get_proc_address(glGetShaderInfoLog, &internal);
        get_proc_address(glGetUniformLocation, &internal);
        get_proc_address(glLinkProgram, &internal);
        get_proc_address(glShaderSource, &internal);
        get_proc_address(glUniform1d, &internal);
        get_proc_address(glUniform1f, &internal);
        get_proc_address(glUniform1i, &internal);
        get_proc_address(glUniform1iv, &internal);
        get_proc_address(glUniform1ui, &internal);
        get_proc_address(glUniform2d, &internal);
        get_proc_address(glUniform2dv, &internal);
        get_proc_address(glUniform2f, &internal);
        get_proc_address(glUniform2fv, &internal);
        get_proc_address(glUniform3d, &internal);
        get_proc_address(glUniform3f, &internal);
        get_proc_address(glUniform4d, &internal);
        get_proc_address(glUniform4f, &internal);
        get_proc_address(glUniformMatrix4fv, &internal);
        get_proc_address(glUniformMatrix4dv, &internal);
        get_proc_address(glUseProgram, &internal);
        get_proc_address(glValidateProgram, &internal);
        get_proc_address(glGetShaderiv, &internal);
        get_proc_address(glDepthFunc, &internal);
        get_proc_address(glPolygonMode, &internal);
        get_proc_address(glUniform1uiv, &internal);
        get_proc_address(glUniform1fv, &internal);
        get_proc_address(glMatrixMode, &internal);
        get_proc_address(glLoadIdentity, &internal);
        get_proc_address(glOrtho, &internal);
        get_proc_address(glHint, &internal);
        get_proc_address(glFlush, &internal);
        get_proc_address(glFinish, &internal);
        get_proc_address(glGetTexLevelParameteriv, &internal);
        get_proc_address(glGetUniformBlockIndex, &internal);
        get_proc_address(glUniformBlockBinding, &internal);
        get_proc_address(glBindBufferRange, &internal);
        
        internal.close(&p);

        opengl_initialized = true;
      }

      #undef get_proc_address

      #if fan_debug >= fan_debug_high

      fan::time::clock c;

      void execute_before(const std::string& function_name) {
        c.start();
      }

      // TODO if function empty probably some WGL/GLX function, initialized in bind window
      void execute_after(const std::string& function_name) {
        glFlush();
        glFinish();
        auto elapsed = c.elapsed();
        if (elapsed > 1000000) {
          fan::print_no_space("slow function call ns:", elapsed, ", function::" + function_name);
        }
      }
      
      //efine call_opengl

      template <typename T, typename ...T2>
      constexpr auto call(const T& t, T2&&... args) {
        execute_before(function_map[(void*)t]);
        if constexpr (std::is_same<fan::return_type_of_t<T>, void>::value) {
          t(args...);
          execute_after(function_map[(void*)t]);
        }
        else {
          auto r = t(args...);
          execute_after(function_map[(void*)t]);
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

      //protected:

      internal_t internal;

      PFNGLVIEWPORTPROC glViewport;
      PFNGLBLENDFUNCPROC glBlendFunc;
      PFNGLCREATEVERTEXARRAYSPROC glCreateVertexArrays;
      PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
      PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
      PFNGLGENBUFFERSPROC glGenBuffers;
      PFNGLBINDBUFFERPROC glBindBuffer;
      PFNGLBUFFERDATAPROC glBufferData;
      PFNGLDELETEBUFFERSPROC glDeleteBuffers;
      PFNGLBUFFERSUBDATAPROC glBufferSubData;
      PFNGLGETBUFFERSUBDATAPROC glGetBufferSubData;
      PFNGLCLEARPROC glClear;
      PFNGLCLEARCOLORPROC glClearColor;
      PFNGLDEBUGMESSAGECALLBACKPROC glDebugMessageCallback;
      PFNGLENABLEPROC glEnable;
      PFNGLDISABLEPROC glDisable;
      PFNGLDRAWARRAYSPROC glDrawArrays;
      PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
      PFNGLGETATTRIBLOCATIONPROC glGetAttribLocation;
      PFNGLGETBUFFERPARAMETERIVPROC glGetBufferParameteriv;
      PFNGLGETINTEGERVPROC glGetIntegerv;
      PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
      PFNGLGETSTRINGPROC glGetString;
      PFNGLGENTEXTURESPROC glGenTextures;
      PFNGLDELETETEXTURESPROC glDeleteTextures;
      PFNGLBINDTEXTUREPROC glBindTexture;
      PFNGLGETTEXTUREIMAGEPROC glGetTexImage;
      PFNGLTEXIMAGE2DPROC glTexImage2D;
      PFNGLTEXPARAMETERIPROC glTexParameteri;
      PFNGLACTIVETEXTUREPROC glActiveTexture;
      PFNGLATTACHSHADERPROC glAttachShader;
      PFNGLCREATESHADERPROC glCreateShader;
      PFNGLDELETESHADERPROC glDeleteShader;
      PFNGLCOMPILESHADERPROC glCompileShader;
      PFNGLCREATEPROGRAMPROC glCreateProgram;
      PFNGLDELETEPROGRAMPROC glDeleteProgram;
      PFNGLGENERATEMIPMAPPROC glGenerateMipmap;
      PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
      PFNGLGETPROGRAMIVPROC glGetProgramiv;
      PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
      PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
      PFNGLLINKPROGRAMPROC glLinkProgram;
      PFNGLSHADERSOURCEPROC glShaderSource;
      PFNGLUNIFORM1DPROC glUniform1d;
      PFNGLUNIFORM1FPROC glUniform1f;
      PFNGLUNIFORM1IPROC glUniform1i;
      PFNGLUNIFORM1IVPROC glUniform1iv;
      PFNGLUNIFORM1UIPROC glUniform1ui;
      PFNGLUNIFORM2DPROC glUniform2d;
      PFNGLUNIFORM2DVPROC glUniform2dv;
      PFNGLUNIFORM2FPROC glUniform2f;
      PFNGLUNIFORM2FVPROC glUniform2fv;
      PFNGLUNIFORM3DPROC glUniform3d;
      PFNGLUNIFORM3FPROC glUniform3f;
      PFNGLUNIFORM4DPROC glUniform4d;
      PFNGLUNIFORM4FPROC glUniform4f;
      PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;
      PFNGLUNIFORMMATRIX4DVPROC glUniformMatrix4dv;
      PFNGLUSEPROGRAMPROC glUseProgram;
      PFNGLVALIDATEPROGRAMPROC glValidateProgram;
      PFNGLGETSHADERIVPROC glGetShaderiv;
      PFNGLDEPTHFUNCPROC glDepthFunc;
      PFNGLPOLYGONMODEPROC glPolygonMode;
      PFNGLUNIFORM1UIVPROC glUniform1uiv;
      PFNGLUNIFORM4FVPROC glUniform1fv;
      PFNGLMATRIXMODEPROC glMatrixMode;
      PFNGLLOADIDENTITYPROC glLoadIdentity;
      PFNGLORTHOPROC glOrtho;
      PFNGLHINTPROC glHint;
      PFNGLFLUSHPROC glFlush;
      PFNGLFINISHPROC glFinish;
      PFNGLGETTEXTURELEVELPARAMETERIVPROC glGetTexLevelParameteriv;
      PFNGLGETUNIFORMBLOCKINDEXPROC glGetUniformBlockIndex;
      PFNGLUNIFORMBLOCKBINDINGPROC glUniformBlockBinding;
      PFNGLBINDBUFFERRANGEPROC glBindBufferRange;

    };

  }

}