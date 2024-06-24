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

    struct internal_t {

      #if defined(fan_platform_windows)
        static constexpr const char* shared_library = "opengl32.lib";
      #elif defined(fan_platform_unix)
        static constexpr const char* shared_library = "libGL.so.1";
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
       
      }
      void close(const properties_t* p) {

      }

    };

    inline bool opengl_initialized = false;

    struct opengl_t {

    private:

      static void* get_proc_address_(const char* name, internal_t* internal)
      {
        return (void*)glfwGetProcAddress(name);
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
      

      opengl_t(bool reinit = false) {

        if (opengl_initialized && reinit == false) {
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
      get_proc_address(PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC, glDrawArraysInstancedBaseInstance, &internal);
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
      get_proc_address(PFNGLUNIFORM3FVPROC, glUniform3fv, &internal);
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
      get_proc_address(PFNGLFRAMEBUFFERTEXTUREPROC , glFramebufferTexture, &internal);
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
      get_proc_address(PFNGLREADPIXELSPROC, glReadPixels, &internal);
      get_proc_address(PFNGLREADBUFFERPROC, glReadBuffer, &internal);
      get_proc_address(PFNGLDRAWPIXELSPROC, glDrawPixels, &internal);
      get_proc_address(PFNGLBLITFRAMEBUFFERPROC, glBlitFramebuffer, &internal);
      get_proc_address(PFNGLVERTEXATTRIBDIVISORPROC, glVertexAttribDivisor, &internal);
      get_proc_address(PFNGLSTENCILOPPROC, glStencilOp, &internal);
      get_proc_address(PFNGLSTENCILFUNCPROC, glStencilFunc, &internal);
      get_proc_address(PFNGLSTENCILMASKPROC, glStencilMask, &internal);
      


    };
    #undef get_proc_address


  }

}