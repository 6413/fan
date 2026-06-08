#ifdef FAN_WASM
  #include <emscripten.h>
  #include <GLES3/gl3.h>
  #include <GLES2/gl2ext.h>
#else
  #include <glad/gl.h>
  #define FAN_USE_GLAD
#endif