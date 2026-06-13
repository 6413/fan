#if !defined(NO_FUNCS)
  //inline int& fan_track_opengl_calls() {
  //  static int track = 0;
  //  return track;
  //}
  //inline std::function<void(std::string func_name, std::uint64_t elapsed)>& fan_opengl_track_print() {
  //  static std::function<void(std::string func_name, std::uint64_t elapsed)> cb = [](std::string func_name, std::uint64_t elapsed){ };
  //  return cb;
  //}
#endif
#undef NO_FUNCS
#define fan_opengl_call(func) \
\
  [&]() { \
    struct measure_func_t { \
      measure_func_t() { \
        c.start_seconds(0.01); \
      }\
      ~measure_func_t() { \
        if (/*fan_track_opengl_calls()*/0) { /*how to make fan_track_opengl_calls?*/\
          glFinish(); \
          if (c.finished()) {\
            std::string func_call = #func; \
            std::string func_name = func_call.substr(0, func_call.find('(')); \
            /*fan_opengl_track_print()(func_name, c.elapsed());*/ \
          }\
        } \
      } \
      fan::time::timer c; \
    }mf; \
    return func; \
  }()
