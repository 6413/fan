#pragma once

#include _FAN_PATH(graphics/opengl/gl_defines.h)

#include _FAN_PATH(math/random.h)
#include _FAN_PATH(time/time.h)

#if defined(fan_platform_windows)
  #include <Windows.h>
  #pragma comment(lib, "User32.lib")
  #pragma comment(lib, "Gdi32.lib")

#elif defined(fan_platform_unix)
#endif

#include <unordered_map>

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

          // /mtd or enable maybe /LTCG to fix with sanitizer on
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
      #elif defined(fan_platform_android)
        EGLContext context;
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
            fan::throw_error(fan::string("failed to load proc:") + name + ", with error:" + fan::to_string(GetLastError()));
          }
        #endif

        return p;

        #elif defined(fan_platform_android)
        
        return (void*)eglGetProcAddress(name);

        #elif defined(fan_platform_unix)

        return (void*)internal->glXGetProcAddress((const GLubyte*)name);

        #endif
      }


    public:

    #if fan_debug >= fan_debug_high
      // todo implement debug
      #define get_proc_address(type, name, internal_) \
        type name = (type)get_proc_address_(#name, internal_)
      std::unordered_map<void*, fan::string> function_map;
    #else
      #define get_proc_address(type, name, internal_) type name = (type)get_proc_address_(#name, internal_);
    #endif
      

      void open() {

        if (opengl_initialized) {
          return;
        };
       
        
        internal.close(&p);

        opengl_initialized = true;
      }

      #if fan_debug >= fan_debug_high

      fan::time::clock c;

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

      template <typename T, typename ...T2>
      constexpr auto call(const T& t, T2&&... args) {
        if constexpr (std::is_same<fan::return_type_of_t<T>, void>::value) {
          t(args...);
        }
        else {
          return t(args...);
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

      internal_t::properties_t p{};

      inline static auto init = [](auto ptr, internal_t* internal) -> bool {
        internal->open(&ptr->p);
        return 1;
      };

      bool b_init = init(this, &internal);


      get_proc_address(PFNGLGETSTRINGPROC, glGetString, &internal);
      get_proc_address(PFNGLVIEWPORTPROC, glViewport, &internal);
      get_proc_address(PFNGLBLENDFUNCPROC, glBlendFunc, &internal);
      get_proc_address(PFNGLCREATEVERTEXARRAYSPROC, glGenVertexArrays, &internal);
      get_proc_address(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray, &internal);
      get_proc_address(PFNGLGENBUFFERSPROC, glGenBuffers, &internal);
      get_proc_address(PFNGLBINDBUFFERPROC, glBindBuffer, &internal);
      get_proc_address(PFNGLBUFFERDATAPROC, glBufferData, &internal);
      get_proc_address(PFNGLDELETEBUFFERSPROC, glDeleteBuffers, &internal);
      get_proc_address(PFNGLBUFFERSUBDATAPROC, glBufferSubData, &internal);
      get_proc_address(PFNGLCLEARPROC, glClear, &internal);
      get_proc_address(PFNGLCLEARCOLORPROC, glClearColor, &internal);
      get_proc_address(PFNGLDEBUGMESSAGECALLBACKPROC, glDebugMessageCallback, &internal);
      get_proc_address(PFNGLENABLEPROC, glEnable, &internal);
      get_proc_address(PFNGLDISABLEPROC, glDisable, &internal);
      get_proc_address(PFNGLDRAWARRAYSPROC, glDrawArrays, &internal);
      get_proc_address(PFNGLDRAWARRAYSINSTANCEDPROC, glDrawArraysInstanced, &internal);
      get_proc_address(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray, &internal);
      get_proc_address(PFNGLGETATTRIBLOCATIONPROC, glGetAttribLocation, &internal);
      get_proc_address(PFNGLGETBUFFERPARAMETERIVPROC, glGetBufferParameteriv, &internal);
      get_proc_address(PFNGLGETINTEGERVPROC, glGetIntegerv, &internal);
      get_proc_address(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer, &internal);
      get_proc_address(PFNGLGENTEXTURESPROC, glGenTextures, &internal);
      get_proc_address(PFNGLDELETETEXTURESPROC, glDeleteTextures, &internal);
      get_proc_address(PFNGLBINDTEXTUREPROC, glBindTexture, &internal);
      get_proc_address(PFNGLGETTEXIMAGEPROC, glGetTexImage, &internal);
      get_proc_address(PFNGLTEXIMAGE2DPROC, glTexImage2D, &internal);
      get_proc_address(PFNGLTEXPARAMETERIPROC, glTexParameteri, &internal);
      get_proc_address(PFNGLACTIVETEXTUREPROC, glActiveTexture, &internal);
      get_proc_address(PFNGLATTACHSHADERPROC, glAttachShader, &internal);
      get_proc_address(PFNGLCREATESHADERPROC, glCreateShader, &internal);
      get_proc_address(PFNGLDELETESHADERPROC, glDeleteShader, &internal);
      get_proc_address(PFNGLCOMPILESHADERPROC, glCompileShader, &internal);
      get_proc_address(PFNGLCREATEPROGRAMPROC, glCreateProgram, &internal);
      get_proc_address(PFNGLDELETEPROGRAMPROC, glDeleteProgram, &internal);
      get_proc_address(PFNGLGENERATEMIPMAPPROC, glGenerateMipmap, &internal);
      get_proc_address(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog, &internal);
      get_proc_address(PFNGLGETPROGRAMIVPROC, glGetProgramiv, &internal);
      get_proc_address(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog, &internal);
      get_proc_address(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation, &internal);
      get_proc_address(PFNGLLINKPROGRAMPROC, glLinkProgram, &internal);
      get_proc_address(PFNGLSHADERSOURCEPROC, glShaderSource, &internal);
      get_proc_address(PFNGLUNIFORM1DPROC, glUniform1d, &internal);
      get_proc_address(PFNGLUNIFORM1FPROC, glUniform1f, &internal);
      get_proc_address(PFNGLUNIFORM1IPROC, glUniform1i, &internal);
      get_proc_address(PFNGLUNIFORM1IVPROC, glUniform1iv, &internal);
      get_proc_address(PFNGLUNIFORM1UIPROC, glUniform1ui, &internal);
      get_proc_address(PFNGLUNIFORM2DPROC, glUniform2d, &internal);
      get_proc_address(PFNGLUNIFORM2DVPROC, glUniform2dv, &internal);
      get_proc_address(PFNGLUNIFORM2FPROC, glUniform2f, &internal);
      get_proc_address(PFNGLUNIFORM2FVPROC, glUniform2fv, &internal);
      get_proc_address(PFNGLUNIFORM3DPROC, glUniform3d, &internal);
      get_proc_address(PFNGLUNIFORM3FPROC, glUniform3f, &internal);
      get_proc_address(PFNGLUNIFORM4DPROC, glUniform4d, &internal);
      get_proc_address(PFNGLUNIFORM4FPROC, glUniform4f, &internal);
      get_proc_address(PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv, &internal);
      get_proc_address(PFNGLUNIFORMMATRIX4DVPROC, glUniformMatrix4dv, &internal);
      get_proc_address(PFNGLUSEPROGRAMPROC, glUseProgram, &internal);
      get_proc_address(PFNGLVALIDATEPROGRAMPROC, glValidateProgram, &internal);
      get_proc_address(PFNGLGETSHADERIVPROC, glGetShaderiv, &internal);
      get_proc_address(PFNGLDEPTHFUNCPROC, glDepthFunc, &internal);
      get_proc_address(PFNGLPOLYGONMODEPROC, glPolygonMode, &internal);
      get_proc_address(PFNGLUNIFORM1UIVPROC, glUniform1uiv, &internal);
      get_proc_address(PFNGLUNIFORM4FVPROC, glUniform4fv, &internal);
      get_proc_address(PFNGLMATRIXMODEPROC, glMatrixMode, &internal);
      get_proc_address(PFNGLLOADIDENTITYPROC, glLoadIdentity, &internal);
      get_proc_address(PFNGLORTHOPROC, glOrtho, &internal);
      get_proc_address(PFNGLHINTPROC, glHint, &internal);
      get_proc_address(PFNGLFLUSHPROC, glFlush, &internal);
      get_proc_address(PFNGLFINISHPROC, glFinish, &internal);
      get_proc_address(PFNGLGETTEXTURELEVELPARAMETERIVPROC, glGetTexLevelParameteriv, &internal);
      get_proc_address(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex, &internal);
      get_proc_address(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding, &internal);
      get_proc_address(PFNGLBINDBUFFERRANGEPROC, glBindBufferRange, &internal);
      get_proc_address(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers, &internal);
      get_proc_address(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers, &internal);
      get_proc_address(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer, &internal);
      get_proc_address(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D, &internal);
      get_proc_address(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers, &internal);
      get_proc_address(PFNGLDELETERENDERBUFFERSPROC, glDeleteRenderbuffers, &internal);
      get_proc_address(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer, &internal);
      get_proc_address(PFNGLRENDERBUFFERSTORAGEPROC, glRenderbufferStorage, &internal);
      get_proc_address(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer, &internal);
      get_proc_address(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus, &internal);
      get_proc_address(PFNGLDEPTHMASKPROC, glDepthMask, &internal);
      get_proc_address(PFNGLCULLFACEPROC, glCullFace, &internal);
      get_proc_address(PFNGLFRONTFACEPROC, glFrontFace, &internal);
      get_proc_address(PFNGLBLENDEQUATIONPROC, glBlendEquation, &internal);
      get_proc_address(PFNGLALPHAFUNCPROC, glAlphaFunc, &internal);
      get_proc_address(PFNGLTEXPARAMETERFPROC, glTexParameterf, &internal);
      get_proc_address(PFNGLDRAWBUFFERSPROC, glDrawBuffers, &internal);
      get_proc_address(PFNGLCLEARBUFFERFVPROC, glClearBufferfv, &internal);
      get_proc_address(PFNGLLINEWIDTHPROC, glLineWidth, &internal);
      get_proc_address(PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC, glNamedFramebufferDrawBuffers, &internal);
      get_proc_address(PFNGLDRAWBUFFERPROC, glDrawBuffer, &internal);
      get_proc_address(PFNGLGETERRORPROC, glGetError, &internal);
      get_proc_address(PFNGLPIXELSTOREIPROC, glPixelStorei, &internal);
      get_proc_address(PFNGLGETBUFFERSUBDATAPROC, glGetBufferSubData, &internal);
      get_proc_address(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays, &internal);
      get_proc_address(PFNGLUNIFORM1FVPROC, glUniform1fv, &internal);
      get_proc_address(PFNGLGETSTRINGIPROC, glGetStringi, &internal);
      get_proc_address(PFNGLBLENDFUNCSEPARATEPROC, glBlendFuncSeparate, &internal);
      get_proc_address(PFNGLSCISSORPROC, glScissor, &internal);
      get_proc_address(PFNGLDRAWELEMENTSPROC, glDrawElements, &internal);
      get_proc_address(PFNGLDETACHSHADERPROC, glDetachShader, &internal);
      get_proc_address(PFNGLBLENDEQUATIONSEPARATEPROC, glBlendEquationSeparate, &internal);
      get_proc_address(PFNGLISENABLEDPROC, glIsEnabled, &internal);
      get_proc_address(PFNGLISPROGRAMPROC, glIsProgram, &internal);




    };
    #undef get_proc_address


  }

}