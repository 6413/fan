#pragma once

#include <fan/graphics/opengl/gl_defines.h>

#include <fan/math/random.h>

#if defined(fan_platform_windows)
  #include <Windows.h>
  #pragma comment(lib, "User32.lib")
  #pragma comment(lib, "opengl32.lib")
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
          auto str = fan::random::string(10);
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
            sizeof(pfd),
            1,
            PFD_TYPE_RGBA,
            PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            32,
            8,
            PFD_MAIN_PLANE,
            24,
            8,
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

      static void* get_proc_address(const char* name, internal_t* internal)
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

      void open() {

        if (opengl_initialized) {
          return;
        }

        internal_t::properties_t p;

        internal.open(&p);

        glGetString = (PFNGLGETSTRINGPROC)get_proc_address("glGetString", &internal);
        glViewport = (decltype(glViewport))get_proc_address("glViewport", &internal);
        glBlendFunc = (decltype(glBlendFunc))get_proc_address("glBlendFunc", &internal);
        glCreateVertexArrays = (decltype(glCreateVertexArrays))get_proc_address("glCreateVertexArrays", &internal);
        glDeleteVertexArrays = (decltype(glDeleteVertexArrays))get_proc_address("glDeleteVertexArrays", &internal);
        glBindVertexArray = (decltype(glBindVertexArray))get_proc_address("glBindVertexArray", &internal);
        glGenBuffers = (decltype(glGenBuffers))get_proc_address("glGenBuffers", &internal);
        glBindBuffer = (decltype(glBindBuffer))get_proc_address("glBindBuffer", &internal);
        glBufferData = (decltype(glBufferData))get_proc_address("glBufferData", &internal);
        glDeleteBuffers = (decltype(glDeleteBuffers))get_proc_address("glDeleteBuffers", &internal);
        glBufferSubData = (decltype(glBufferSubData))get_proc_address("glBufferSubData", &internal);
        glClear = (decltype(glClear))get_proc_address("glClear", &internal);
        glClearColor = (decltype(glClearColor))get_proc_address("glClearColor", &internal);
        glDebugMessageCallback = (decltype(glDebugMessageCallback))get_proc_address("glDebugMessageCallback", &internal);
        glEnable = (decltype(glEnable))get_proc_address("glEnable", &internal);
        glDisable = (decltype(glDisable))get_proc_address("glDisable", &internal);
        glDrawArrays = (decltype(glDrawArrays))get_proc_address("glDrawArrays", &internal);
        glEnableVertexAttribArray = (decltype(glEnableVertexAttribArray))get_proc_address("glEnableVertexAttribArray", &internal);
        glGetAttribLocation = (decltype(glGetAttribLocation))get_proc_address("glGetAttribLocation", &internal);
        glGetBufferParameteriv = (decltype(glGetBufferParameteriv))get_proc_address("glGetBufferParameteriv", &internal);
        glGetIntegerv = (decltype(glGetIntegerv))get_proc_address("glGetIntegerv", &internal);
        glVertexAttribPointer = (decltype(glVertexAttribPointer))get_proc_address("glVertexAttribPointer", &internal);
        glGenTextures = (decltype(glGenTextures))get_proc_address("glGenTextures", &internal);
        glDeleteTextures = (decltype(glDeleteTextures))get_proc_address("glDeleteTextures", &internal);
        glBindTexture = (decltype(glBindTexture))get_proc_address("glBindTexture", &internal);
        glTexImage2D = (decltype(glTexImage2D))get_proc_address("glTexImage2D", &internal);
        glTexParameteri = (decltype(glTexParameteri))get_proc_address("glTexParameteri", &internal);
        glActiveTexture = (decltype(glActiveTexture))get_proc_address("glActiveTexture", &internal);
        glAttachShader = (decltype(glAttachShader))get_proc_address("glAttachShader", &internal);
        glCreateShader = (decltype(glCreateShader))get_proc_address("glCreateShader", &internal);
        glDeleteShader = (decltype(glDeleteShader))get_proc_address("glDeleteShader", &internal);
        glCompileShader = (decltype(glCompileShader))get_proc_address("glCompileShader", &internal);
        glCreateProgram = (decltype(glCreateProgram))get_proc_address("glCreateProgram", &internal);
        glDeleteProgram = (decltype(glDeleteProgram))get_proc_address("glDeleteProgram", &internal);
        glGenerateMipmap = (decltype(glGenerateMipmap))get_proc_address("glGenerateMipmap", &internal);
        glGetProgramInfoLog = (decltype(glGetProgramInfoLog))get_proc_address("glGetProgramInfoLog", &internal);
        glGetProgramiv = (decltype(glGetProgramiv))get_proc_address("glGetProgramiv", &internal);
        glGetShaderInfoLog = (decltype(glGetShaderInfoLog))get_proc_address("glGetShaderInfoLog", &internal);
        glGetUniformLocation = (decltype(glGetUniformLocation))get_proc_address("glGetUniformLocation", &internal);
        glLinkProgram = (decltype(glLinkProgram))get_proc_address("glLinkProgram", &internal);
        glShaderSource = (decltype(glShaderSource))get_proc_address("glShaderSource", &internal);
        glUniform1d = (decltype(glUniform1d))get_proc_address("glUniform1d", &internal);
        glUniform1f = (decltype(glUniform1f))get_proc_address("glUniform1f", &internal);
        glUniform1i = (decltype(glUniform1i))get_proc_address("glUniform1i", &internal);
        glUniform1iv = (decltype(glUniform1iv))get_proc_address("glUniform1iv", &internal);
        glUniform1ui = (decltype(glUniform1ui))get_proc_address("glUniform1ui", &internal);
        glUniform2d = (decltype(glUniform2d))get_proc_address("glUniform2d", &internal);
        glUniform2dv = (decltype(glUniform2dv))get_proc_address("glUniform2dv", &internal);
        glUniform2f = (decltype(glUniform2f))get_proc_address("glUniform2f", &internal);
        glUniform2fv = (decltype(glUniform2fv))get_proc_address("glUniform2fv", &internal);
        glUniform3d = (decltype(glUniform3d))get_proc_address("glUniform3d", &internal);
        glUniform3f = (decltype(glUniform3f))get_proc_address("glUniform3f", &internal);
        glUniform4d = (decltype(glUniform4d))get_proc_address("glUniform4d", &internal);
        glUniform4f = (decltype(glUniform4f))get_proc_address("glUniform4f", &internal);
        glUniformMatrix4fv = (decltype(glUniformMatrix4fv))get_proc_address("glUniformMatrix4fv", &internal);
        glUniformMatrix4dv = (decltype(glUniformMatrix4dv))get_proc_address("glUniformMatrix4dv", &internal);
        glUseProgram = (decltype(glUseProgram))get_proc_address("glUseProgram", &internal);
        glValidateProgram = (decltype(glValidateProgram))get_proc_address("glValidateProgram", &internal);
        glGetShaderiv = (decltype(glGetShaderiv))get_proc_address("glGetShaderiv", &internal);
        glDepthFunc = (decltype(glDepthFunc))get_proc_address("glDepthFunc", &internal);
        glPolygonMode = (decltype(glPolygonMode))get_proc_address("glPolygonMode", &internal);
        glUniform1uiv = (decltype(glUniform1uiv))get_proc_address("glUniform1uiv", &internal);
        glUniform1fv = (decltype(glUniform1fv))get_proc_address("glUniform1fv", &internal);
        glMatrixMode = (decltype(glMatrixMode))get_proc_address("glUniform1fv", &internal);
        glLoadIdentity = (decltype(glLoadIdentity))get_proc_address("glUniform1fv", &internal);
        glOrtho = (decltype(glOrtho))get_proc_address("glUniform1fv", &internal);

        internal.close(&p);

        opengl_initialized = true;
      }

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

    };

  }

}