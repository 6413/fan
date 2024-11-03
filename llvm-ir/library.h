#pragma once

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

inline std::vector<fan::function_t<void()>> task_queue;

static void add_task(auto l) {
  task_queue.push_back(std::move(l));
}

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double x) {
  fputc((char)x, stderr);
  return 0;
}

inline std::vector<loco_t::shape_t> shapes;
inline std::unordered_map<std::string, loco_t::image_t> iamges;

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double x) {
  add_task([=] {
    fan::printcl((uint64_t)x);
  });
  return 0;
}

extern "C" DLLEXPORT double string_test(const char* str) {
  fan::print(str);
  return 0;
}

static int depth = 0;

extern "C" DLLEXPORT double rectangle1(double px, double py, double sx, double sy, double color, double angle) {
  add_task([=] {
    shapes.push_back(fan::graphics::rectangle_t{ {
        .position = fan::vec3(px, py, depth++),
        .size = fan::vec2(sx, sx),
        .color = fan::color::hex((uint32_t)color),
        .angle = angle
    } });
  });
  return 0;
}

extern "C" DLLEXPORT double rectangle0(double px, double py, double sx, double sy) {
  return rectangle1(px, py, sx, sy, fan::random::color().get_hex(), 0);
}


extern "C" DLLEXPORT double sprite0(const char* cpath, double px, double py, double sx, double sy) {
  add_task([=, path = std::string(cpath)] {
    auto found = iamges.find(path);
    if (found != iamges.end()) {
      gloco->image_unload(found->second);
    }
    loco_t::image_t image = gloco->image_load(path);;
    found->second = image;
    shapes.push_back(fan::graphics::sprite_t{ {
        .position = fan::vec3(px, py, depth++),
        .size = fan::vec2(sx, sx),
        .image = image
    } });
  });
  return 0;
}

inline bool needs_frame_skip = false;

extern "C" DLLEXPORT double clear() {
  add_task([&] {
    shapes.clear();
  });
  return 0;
}

extern "C" DLLEXPORT double sleep(double x) {
  add_task([=] {
    needs_frame_skip = true;
    std::this_thread::sleep_for(std::chrono::duration<double>(x));
  });
  return 0;
}