#include <fan/utility.h>
#include <fan/graphics/opengl/init.h>

#include <vector>
#include <string>
#include <functional>
#include <iomanip>
#include <fan/graphics/opengl/init.h>
#include <fstream>
#include <cmath>

import fan;

using namespace fan::graphics;

struct test_result_t {
  std::string name;
  bool passed;
  std::string error_message;
  double duration_ms;
};
//
struct benchmark_result_t {
  std::string name;
  uint64_t iterations;
  double total_time_ms;
  double avg_time_us;
  double ops_per_sec;
};

struct shape_tester_t {
  template<typename Func>
  void run_test(const std::string& name, Func&& test_func) {
    test_result_t result;
    result.name = name;

    fan::time::timer timer(true);

    try {
      test_func();
      result.passed = true;
    }
    catch (const std::exception& e) {
      result.passed = false;
      result.error_message = e.what();
    }

    result.duration_ms = timer.elapsed() / 1e6;

    test_results.push_back(result);
  }

  template<typename Func>
  void run_benchmark(const std::string& name, uint64_t iterations, Func&& bench_func) {
    benchmark_result_t result;
    result.name = name;
    result.iterations = iterations;

    for (int i = 0; i < 100; ++i) {
      bench_func();
    }

    fan::time::timer timer(true);

    for (uint64_t i = 0; i < iterations; ++i) {
      bench_func();
    }

    result.total_time_ms = timer.elapsed() / 1e6;
    result.avg_time_us = (result.total_time_ms * 1000.0) / iterations;
    result.ops_per_sec = (iterations / result.total_time_ms) * 1000.0;

    benchmark_results.push_back(result);
  }

  void assert_equal(const fan::vec2& a, const fan::vec2& b, const std::string& msg = "") {
    if ((a - b).length() > 0.001f) {
      throw std::runtime_error("Vec2 not equal: " + msg + " - Expected: " +
        std::to_string(b.x) + "," + std::to_string(b.y) +
        " Got: " + std::to_string(a.x) + "," + std::to_string(a.y));
    }
  }

  void assert_equal(const fan::vec3& a, const fan::vec3& b, const std::string& msg = "") {
    if ((a - b).length() > 0.001f) {
      throw std::runtime_error("Vec3 not equal: " + msg);
    }
  }

  void assert_equal(const fan::color& a, const fan::color& b, const std::string& msg = "") {
    if (std::abs(a.r - b.r) > 0.001f || std::abs(a.g - b.g) > 0.001f ||
      std::abs(a.b - b.b) > 0.001f || std::abs(a.a - b.a) > 0.001f) {
      throw std::runtime_error("Color not equal: " + msg);
    }
  }

  void assert_equal(f32_t a, f32_t b, const std::string& msg = "") {
    if (std::abs(a - b) > 0.001f) {
      throw std::runtime_error("Float not equal: " + msg + " - Expected: " +
        std::to_string(b) + " Got: " + std::to_string(a));
    }
  }

  void assert_true(bool condition, const std::string& msg = "") {
    if (!condition) {
      throw std::runtime_error(msg);
    }
  }

  void test_keypack_integrity() {
    run_test("Keypack Integrity - Rectangle", [&]() {
      rectangle_t rect{ {
        .position = fan::vec3(100, 200, 5),
        .size = fan::vec2(50, 60),
        .color = fan::colors::red
      } };

      auto keypack_size = g_shapes->shaper.GetKeysSize(rect);
      assert_true(keypack_size > 0, "Keypack size should be > 0");

      uint8_t* keypack = g_shapes->shaper.GetKeys(rect);
      assert_true(keypack != nullptr, "Keypack should not be null");
    });

    run_test("Keypack Integrity - Sprite", [&]() {
      sprite_t sprite{ {
        .position = fan::vec3(100, 200, 3),
        .size = fan::vec2(32, 32),
        .image = engine.default_texture
      } };

      auto keypack_size = g_shapes->shaper.GetKeysSize(sprite);
      assert_true(keypack_size > 0, "Keypack size should be > 0");
    });
  }

  void test_position_operations() {
    run_test("Position Set/Get - Rectangle", [&]() {
      rectangle_t rect{ {.position = fan::vec3(0, 0, 0)} };

      fan::vec3 new_pos(100, 200, 5);
      rect.set_position(new_pos);

      fan::vec3 got_pos = rect.get_position();
      assert_equal(got_pos, new_pos, "Position should match after set");
    });

    run_test("Position Set/Get - Circle", [&]() {
      circle_t circle{ {.position = fan::vec3(0, 0, 0), .radius = 50} };

      fan::vec3 new_pos(150, 250, 10);
      circle.set_position(new_pos);

      fan::vec3 got_pos = circle.get_position();
      assert_equal(got_pos, new_pos, "Position should match after set");
    });

    run_test("Position XYZ Setters", [&]() {
      rectangle_t rect{ {.position = fan::vec3(100, 200, 5)} };

      rect.set_x(500);
      assert_equal(rect.get_x(), 500.0f, "X should be updated");
      assert_equal(rect.get_y(), 200.0f, "Y should remain unchanged");

      rect.set_y(600);
      assert_equal(rect.get_y(), 600.0f, "Y should be updated");

      rect.set_z(15);
      assert_equal(rect.get_z(), 15.0f, "Z should be updated");
    });
  }

  void test_size_operations() {
    run_test("Size Set/Get - Rectangle", [&]() {
      rectangle_t rect{ {.size = fan::vec2(50, 60)} };

      fan::vec2 new_size(100, 120);
      rect.set_size(new_size);

      fan::vec2 got_size = rect.get_size();
      assert_equal(got_size, new_size, "Size should match after set");
    });

    run_test("Size Set/Get - Sprite", [&]() {
      sprite_t sprite{ {
        .size = fan::vec2(32, 32),
        .image = engine.default_texture
      } };

      fan::vec2 new_size(64, 64);
      sprite.set_size(new_size);

      fan::vec2 got_size = sprite.get_size();
      assert_equal(got_size, new_size, "Size should match after set");
    });
  }

  void test_color_operations() {
    run_test("Color Set/Get - Rectangle", [&]() {
      rectangle_t rect{ {.color = fan::colors::white} };

      fan::color new_color = fan::colors::red;
      rect.set_color(new_color);

      fan::color got_color = rect.get_color();
      assert_equal(got_color, new_color, "Color should match after set");
    });

    run_test("Color Set/Get - Circle", [&]() {
      circle_t circle{ {
        .radius = 50,
        .color = fan::colors::blue
      } };

      fan::color new_color = fan::colors::green;
      circle.set_color(new_color);

      fan::color got_color = circle.get_color();
      assert_equal(got_color, new_color, "Color should match after set");
    });
  }

  void test_angle_operations() {
    run_test("Angle Set/Get - Rectangle", [&]() {
      rectangle_t rect{ {.angle = fan::vec3(0, 0, 0)} };

      fan::vec3 new_angle(0, 0, fan::math::pi / 4);
      rect.set_angle(new_angle);

      fan::vec3 got_angle = rect.get_angle();
      assert_equal(got_angle, new_angle, "Angle should match after set");
    });

    run_test("Rotation Point Set/Get", [&]() {
      rectangle_t rect{ {
        .size = fan::vec2(100, 100),
        .rotation_point = fan::vec2(0, 0)
      } };

      fan::vec2 new_rp(50, 50);
      rect.set_rotation_point(new_rp);

      fan::vec2 got_rp = rect.get_rotation_point();
      assert_equal(got_rp, new_rp, "Rotation point should match after set");
    });
  }

  void test_camera_viewport_operations() {
    run_test("Camera Set/Get", [&]() {
      rectangle_t rect{ {.position = fan::vec3(0, 0, 0)} };

      auto new_cam = engine.orthographic_render_view.camera;
      rect.set_camera(new_cam);

      auto got_cam = rect.get_camera();
      assert_true(got_cam.NRI == new_cam.NRI, "Camera should match after set");
    });

    run_test("Viewport Set/Get", [&]() {
      rectangle_t rect{ {.position = fan::vec3(0, 0, 0)} };

      auto new_vp = engine.orthographic_render_view.viewport;
      rect.set_viewport(new_vp);

      auto got_vp = rect.get_viewport();
      assert_true(got_vp.NRI == new_vp.NRI, "Viewport should match after set");
    });
  }

  void test_image_operations() {
    run_test("Image Set/Get - Sprite", [&]() {
      auto test_img = engine.image_create(fan::colors::red);

      sprite_t sprite{ {
        .image = engine.default_texture
      } };

      sprite.set_image(test_img);

      auto got_img = sprite.get_image();
      assert_true(got_img.NRI == test_img.NRI, "Image should match after set");

      engine.image_unload(test_img);
    });
  }

  void test_shape_specific_properties() {
    run_test("Radius Set/Get - Circle", [&]() {
      circle_t circle{ {.radius = 50} };

      f32_t got_radius = circle.get_radius();
      assert_equal(got_radius, 50.0f, "Radius should match initial value");
    });

    run_test("Line Endpoints", [&]() {
      line_t line{ {
        .src = fan::vec3(0, 0, 0),
        .dst = fan::vec3(100, 100, 0)
      } };

      fan::vec3 got_src = line.get_src();
      fan::vec3 got_dst = line.get_dst();

      assert_equal(got_src, fan::vec3(0, 0, 0), "Src should match");
      assert_equal(got_dst, fan::vec3(100, 100, 0), "Dst should match");
    });
  }

  void test_keypack_changes() {
    run_test("Keypack Changes - Position Update", [&]() {
      rectangle_t rect{ {.position = fan::vec3(100, 100, 5)} };

      auto initial_keypack_size = g_shapes->shaper.GetKeysSize(rect);

      rect.set_position(fan::vec3(200, 200, 10));

      auto new_keypack_size = g_shapes->shaper.GetKeysSize(rect);

      assert_true(initial_keypack_size == new_keypack_size,
        "Keypack size should remain same after position update");

      assert_equal(rect.get_position(), fan::vec3(200, 200, 10),
        "Position should be updated");
    });

    run_test("Keypack Changes - Camera Update", [&]() {
      rectangle_t rect{ {.position = fan::vec3(0, 0, 0)} };

      auto initial_cam = rect.get_camera();
      auto new_cam = engine.camera_create();

      rect.set_camera(new_cam);

      auto got_cam = rect.get_camera();
      assert_true(got_cam.NRI != initial_cam.NRI,
        "Camera should be different after update");
    });
  }

  void test_sequential_updates() {
    run_test("Sequential Position Updates", [&]() {
      rectangle_t rect{ {.position = fan::vec3(0, 0, 0)} };

      for (int i = 1; i <= 10; ++i) {
        fan::vec3 new_pos(i * 10.0f, i * 20.0f, i * 1.0f);
        rect.set_position(new_pos);

        fan::vec3 got_pos = rect.get_position();
        assert_equal(got_pos, new_pos,
          "Position should match after update " + std::to_string(i));
      }
    });

    run_test("Mixed Property Updates", [&]() {
      rectangle_t rect{ {
        .position = fan::vec3(0, 0, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::white
      } };

      rect.set_position(fan::vec3(100, 200, 5));
      assert_equal(rect.get_position(), fan::vec3(100, 200, 5));

      rect.set_size(fan::vec2(100, 100));
      assert_equal(rect.get_size(), fan::vec2(100, 100));

      rect.set_color(fan::colors::red);
      assert_equal(rect.get_color(), fan::colors::red);

      assert_equal(rect.get_position(), fan::vec3(100, 200, 5));
      assert_equal(rect.get_size(), fan::vec2(100, 100));
      assert_equal(rect.get_color(), fan::colors::red);
    });
  }

  void assert_shape_pixels(const fan::graphics::shapes::shape_t& shape, const fan::color& expected_color) {
    fan::vec2 ws = engine.window.get_size();
    float tol = 0.09f;

    engine.process_loop();
    engine.process_loop([&] {
      try {
        engine.gl.m_framebuffer.bind(engine.context.gl);
        glReadBuffer(GL_COLOR_ATTACHMENT0);

        std::vector<uint8_t> px(ws.x * ws.y * 4);
        glReadPixels(0, 0, ws.x, ws.y, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

        engine.gl.m_framebuffer.unbind(engine.context.gl);

        fan::color bg = engine.clear_color;
        int min_x = ws.x, max_x = 0, min_y = ws.y, max_y = 0;

        for (int y = 0; y < ws.y; ++y) {
          for (int x = 0; x < ws.x; ++x) {
            int flipped_y = ws.y - 1 - y;
            int i = (flipped_y * (int)ws.x + x) * 4;

            fan::color c(px[i] / 255.f, px[i + 1] / 255.f, px[i + 2] / 255.f, px[i + 3] / 255.f);

            bool is_shape_color = false;
            if (expected_color.r > 0.5f) {
              is_shape_color = c.r > 0.5f && c.g < 0.3f && c.b < 0.3f;
            }
            else if (expected_color.g > 0.5f) {
              is_shape_color = c.g > 0.5f && c.r < 0.3f && c.b < 0.3f;
            }
            else if (expected_color.b > 0.5f) {
              is_shape_color = c.b > 0.5f && c.r < 0.3f && c.g < 0.3f;
            }

            if (is_shape_color) {
              min_x = std::min(min_x, x);
              max_x = std::max(max_x, x);
              min_y = std::min(min_y, y);
              max_y = std::max(max_y, y);
            }
          }
        }

        for (int y = 0; y < ws.y; ++y) {
          for (int x = 0; x < ws.x; ++x) {
            int flipped_y = ws.y - 1 - y;
            int i = (flipped_y * (int)ws.x + x) * 4;

            fan::color c(px[i] / 255.f, px[i + 1] / 255.f, px[i + 2] / 255.f, px[i + 3] / 255.f);

            bool inside_rendered = (x >= min_x - 1 && x <= max_x + 1 && y >= min_y - 1 && y <= max_y + 1);
            bool is_aa = (c.a > 0.01f && c.a < 0.99f);

            if (!is_aa) {
              if (expected_color.r > 0.5f && ((c.r > 0.001f && c.r < 0.1f) || (c.r > 0.05f && c.r < expected_color.r * 0.95f))) {
                is_aa = true;
              }
              else if (expected_color.g > 0.5f && ((c.g > 0.001f && c.g < 0.1f) || (c.g > 0.05f && c.g < expected_color.g * 0.95f))) {
                is_aa = true;
              }
              else if (expected_color.b > 0.5f && ((c.b > 0.001f && c.b < 0.1f) || (c.b > 0.05f && c.b < expected_color.b * 0.95f))) {
                is_aa = true;
              }
            }

            if (is_aa) {
              continue;
            }

            bool matches_bg = (std::abs(c.r - bg.r) < tol && std::abs(c.g - bg.g) < tol && std::abs(c.b - bg.b) < tol);

            if (!inside_rendered && !matches_bg) {
              assert_true(false, "Pixel outside shape was modified");
            }
          }
        }
      }
      catch (const std::exception& e) {
        fan::print_error("Exception:"_str + e.what());
        throw;
      }
    });
  }

  void test_pixel_accuracy() {
    run_test("Pixel Accuracy - Rectangle", [&]() {
      fan::vec2 ws = engine.window.get_size();

      rectangle_t rect{ {
        .position = fan::vec3(ws.x / 2, ws.y / 2, 0),
        .size = fan::vec2(100, 80),
        .color = fan::colors::red
      } };

      assert_shape_pixels(rect, fan::colors::red);
    });

    run_test("Pixel Accuracy - Circle", [&]() {
      fan::vec2 ws = engine.window.get_size();

      circle_t circle{ {
        .position = fan::vec3(ws.x / 2, ws.y / 2, 0),
        .radius = 40,
        .color = fan::colors::green
      } };

      assert_shape_pixels(circle, fan::colors::green);
    });
  }

  void test_depth_ordering() {
    run_test("Depth Ordering - Z-axis position change", [&]() {
      fan::vec2 ws = engine.window.get_size();
      fan::vec3 center_pos(ws.x / 2, ws.y / 2, 0);

      rectangle_t red_rect{ {
        .position = fan::vec3(center_pos.x - 20, center_pos.y - 20, 0),
        .size = fan::vec2(100, 100),
        .color = fan::colors::red
      } };

      rectangle_t green_rect{ {
        .position = fan::vec3(center_pos.x + 20, center_pos.y + 20, 5),
        .size = fan::vec2(100, 100),
        .color = fan::colors::green
      } };

      engine.process_loop();
      engine.process_loop([&] {
        try {
          engine.gl.m_framebuffer.bind(engine.context.gl);
          glReadBuffer(GL_COLOR_ATTACHMENT0);

          std::vector<uint8_t> px(ws.x * ws.y * 4);
          glReadPixels(0, 0, ws.x, ws.y, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

          engine.gl.m_framebuffer.unbind(engine.context.gl);

          fan::vec2 overlap_point(center_pos.x, center_pos.y);
          int flipped_y = ws.y - 1 - (int)overlap_point.y;
          int i = (flipped_y * (int)ws.x + (int)overlap_point.x) * 4;

          fan::color c(px[i] / 255.f, px[i + 1] / 255.f, px[i + 2] / 255.f, px[i + 3] / 255.f);

          bool is_green = (c.g > 0.5f && c.r < 0.3f && c.b < 0.3f);
          assert_true(is_green, "Green rect should be on top initially (z=5 > z=0)");
        }
        catch (const std::exception& e) {
          fan::print_error("Exception:"_str + e.what());
          throw;
        }
      });

      red_rect.set_position(fan::vec3(center_pos.x - 20, center_pos.y - 20, 10));

      engine.process_loop();
      engine.process_loop([&] {
        try {
          engine.gl.m_framebuffer.bind(engine.context.gl);
          glReadBuffer(GL_COLOR_ATTACHMENT0);

          std::vector<uint8_t> px(ws.x * ws.y * 4);
          glReadPixels(0, 0, ws.x, ws.y, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

          engine.gl.m_framebuffer.unbind(engine.context.gl);

          fan::vec2 overlap_point(center_pos.x, center_pos.y);
          int flipped_y = ws.y - 1 - (int)overlap_point.y;
          int i = (flipped_y * (int)ws.x + (int)overlap_point.x) * 4;

          fan::color c(px[i] / 255.f, px[i + 1] / 255.f, px[i + 2] / 255.f, px[i + 3] / 255.f);

          bool is_red = (c.r > 0.5f && c.g < 0.3f && c.b < 0.3f);
          assert_true(is_red, "Red rect should be on top after z change (z=10 > z=5)");
        }
        catch (const std::exception& e) {
          fan::print_error("Exception:"_str + e.what());
          throw;
        }
      });
    });
  }

  void test_camera_position() {
    run_test("Camera Position Persistence", [&]() {
      fan::vec2 ws = engine.window.get_size();

      auto test_cam = engine.camera_create();
      engine.camera_set_ortho(test_cam, fan::vec2(0, ws.x), fan::vec2(0, ws.y));

      engine.camera_set_position(test_cam, fan::vec3(100, 200, 0));
      fan::vec3 pos = engine.camera_get_position(test_cam);

      assert_equal(pos, fan::vec3(100, 200, 0), "Camera position mismatch");

      engine.camera_set_position(test_cam, fan::vec3(0, 0, 0));
      pos = engine.camera_get_position(test_cam);
      assert_equal(pos, fan::vec3(0, 0, 0), "Camera reset failed");

      engine.camera_erase(test_cam);
    });
  }

  void test_viewport_resize() {
    run_test("Viewport Resize", [&]() {
      fan::vec2 ws = engine.window.get_size();

      auto vp = engine.viewport_create(fan::vec2(100, 100), ws / 2);

      fan::vec2 vp_pos = engine.viewport_get_position(vp);
      fan::vec2 vp_size = engine.viewport_get_size(vp);

      assert_equal(vp_pos, fan::vec2(100, 100), "Viewport position mismatch");
      assert_equal(vp_size, ws / 2, "Viewport size mismatch");

      engine.viewport_set(vp, fan::vec2(0, 0), ws);
      vp_pos = engine.viewport_get_position(vp);
      vp_size = engine.viewport_get_size(vp);

      assert_equal(vp_pos, fan::vec2(0, 0), "Viewport position update failed");
      assert_equal(vp_size, ws, "Viewport size update failed");

      engine.viewport_erase(vp);
    });
  }

  #if defined(fan_physics)
  void test_collision_detection() {
    run_test("Rectangle AABB Intersection", [&]() {
      rectangle_t rect1{ {
        .position = fan::vec3(100, 100, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::red
      } };

      rectangle_t rect2{ {
        .position = fan::vec3(125, 125, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::blue
      } };

      assert_true(rect1.intersects(rect2), "Overlapping rectangles should intersect");

      rect2.set_position(fan::vec3(300, 300, 0));
      assert_true(!rect1.intersects(rect2), "Separated rectangles should not intersect");

      rect2.set_position(fan::vec3(100, 100, 0));
      assert_true(rect1.intersects(rect2), "Identical position rectangles should intersect");

      rect2.set_position(fan::vec3(150, 100, 0));
      assert_true(rect1.intersects(rect2), "Edge-touching rectangles should intersect");
    });

    run_test("Circle Point Inside", [&]() {
      circle_t circle{ {
        .position = fan::vec3(200, 200, 0),
        .radius = 50,
        .color = fan::colors::green
      } };

      assert_true(circle.point_inside(fan::vec2(200, 200)), "Center should be inside circle");
      assert_true(circle.point_inside(fan::vec2(225, 200)), "Point within radius should be inside");
      assert_true(!circle.point_inside(fan::vec2(300, 300)), "Point outside radius should be outside");
      assert_true(circle.point_inside(fan::vec2(200 + 49, 200)), "Point near edge should be inside");
    });

    run_test("Rectangle Point Inside", [&]() {
      rectangle_t rect{ {
        .position = fan::vec3(100, 100, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::red
      } };

      assert_true(rect.point_inside(fan::vec2(100, 100)), "Center should be inside");
      assert_true(rect.point_inside(fan::vec2(120, 120)), "Interior point should be inside");
      assert_true(!rect.point_inside(fan::vec2(200, 200)), "Exterior point should be outside");
    });
  }
#endif

  void test_image_assignment() {
    run_test("Image Creation and Validation", [&]() {
      auto img = engine.image_create(fan::colors::blue);

      assert_true(!img.iic(), "Created image should be valid");
      assert_true(engine.is_image_valid(img), "is_image_valid should return true");

      sprite_t sprite{ {
        .size = fan::vec2(32, 32),
        .image = img
      } };

      assert_true(sprite.get_image() == img, "Sprite should have correct image");

      auto img2 = engine.image_create(fan::colors::red);
      sprite.set_image(img2);

      assert_true(sprite.get_image() == img2, "Sprite image should update");
      assert_true(sprite.get_image() != img, "Sprite should no longer have old image");

      engine.image_unload(img);
      engine.image_unload(img2);
    });

    run_test("Default Texture Validity", [&]() {
      assert_true(!engine.default_texture.iic(), "Default texture should be valid");

      sprite_t sprite{ {
        .size = fan::vec2(32, 32)
      } };

      assert_true(sprite.get_image() == engine.default_texture, "Sprite should use default texture");
    });
  }

  void test_rotation_matrix() {
    run_test("Shape Basis Vectors", [&]() {
      rectangle_t rect{ {
        .position = fan::vec3(100, 100, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::red,
        .angle = fan::vec3(0, 0, 0)
      } };

      auto basis = rect.get_basis();
      assert_equal(basis.right, fan::vec3(1, 0, 0), "Right vector at 0 rotation");
      assert_equal(basis.forward, fan::vec3(0, -1, 0), "Forward vector at 0 rotation");

      rect.set_angle(fan::vec3(0, 0, fan::math::pi / 2));
      basis = rect.get_basis();

      f32_t tolerance = 0.001f;
      assert_true(std::abs(basis.right.x) < tolerance, "Right.x should be ~0 at 90deg");
      assert_true(std::abs(basis.right.y - 1.0f) < tolerance, "Right.y should be ~1 at 90deg");
    });

  #if defined(fan_physics)
    run_test("Shape AABB Calculation", [&]() {
      rectangle_t rect{ {
        .position = fan::vec3(100, 100, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::red,
        .angle = fan::vec3(0, 0, 0)
      } };

      auto aabb = rect.get_aabb();

      assert_equal(aabb.min, fan::vec2(50, 50), "AABB min incorrect");
      assert_equal(aabb.max, fan::vec2(150, 150), "AABB max incorrect");

      rect.set_angle(fan::vec3(0, 0, fan::math::pi / 4));
      aabb = rect.get_aabb();

      f32_t expected_extent = 50.0f * std::sqrt(2.0f);
      assert_true(aabb.max.x > 100 + expected_extent * 0.9f, "Rotated AABB should be larger");
      assert_true(aabb.min.x < 100 - expected_extent * 0.9f, "Rotated AABB should be larger");
    });
  #endif
  }

  void test_render_view_assignment() {
    run_test("Render View Creation", [&]() {
      auto render_view = engine.render_view_create();

      assert_true(!render_view.camera.iic(), "Camera should be valid");
      assert_true(!render_view.viewport.iic(), "Viewport should be valid");

      engine.viewport_erase(render_view.viewport);
      engine.camera_erase(render_view.camera);
    });

    run_test("Render View Assignment", [&]() {
      auto render_view = engine.render_view_create();

      rectangle_t rect{ {
        .position = fan::vec3(100, 100, 0),
        .size = fan::vec2(50, 50),
        .color = fan::colors::red
      } };

      rect.set_render_view(render_view);
      auto retrieved_view = rect.get_render_view();

      assert_true(retrieved_view.camera.NRI == render_view.camera.NRI, "Camera mismatch");
      assert_true(retrieved_view.viewport.NRI == render_view.viewport.NRI, "Viewport mismatch");

      rect.set_camera(engine.orthographic_render_view.camera);
      assert_true(rect.get_camera().NRI == engine.orthographic_render_view.camera.NRI, "Camera update failed");

      engine.viewport_erase(render_view.viewport);
      engine.camera_erase(render_view.camera);
    });
  }

  void test_coordinate_conversion() {
    run_test("Screen to NDC Conversion", [&]() {
      fan::vec2 ws = engine.window.get_size();

      fan::vec2 ndc = engine.screen_to_ndc(fan::vec2(0, 0));
      assert_equal(ndc, fan::vec2(-1, -1), "Top-left corner should be (-1, -1)");

      ndc = engine.screen_to_ndc(ws);
      assert_equal(ndc, fan::vec2(1, 1), "Bottom-right corner should be (1, 1)");

      ndc = engine.screen_to_ndc(ws / 2);
      assert_equal(ndc, fan::vec2(0, 0), "Center should be (0, 0)");
    });

    run_test("NDC to Screen Conversion", [&]() {
      fan::vec2 ws = engine.window.get_size();

      fan::vec2 screen = engine.ndc_to_screen(fan::vec2(-1, -1));
      assert_equal(screen, fan::vec2(0, 0), "NDC (-1, -1) should be (0, 0)");

      screen = engine.ndc_to_screen(fan::vec2(1, 1));
      assert_equal(screen, ws, "NDC (1, 1) should be window size");

      screen = engine.ndc_to_screen(fan::vec2(0, 0));
      assert_equal(screen, ws / 2, "NDC (0, 0) should be center");
    });

    run_test("Round-trip Coordinate Transform", [&]() {
      fan::vec2 ws = engine.window.get_size();
      fan::vec2 original(123, 456);

      fan::vec2 ndc = engine.screen_to_ndc(original);
      fan::vec2 back = engine.ndc_to_screen(ndc);

      assert_equal(back, original, "Round-trip conversion should preserve coordinates");
    });
  }

  void test_shape_independence() {
    run_test("Multiple Shapes Same Position", [&]() {
      fan::vec2 ws = engine.window.get_size();
      fan::vec3 shared_pos(ws.x / 2, ws.y / 2, 0);

      rectangle_t rect{ {
        .position = shared_pos,
        .size = fan::vec2(50, 50),
        .color = fan::colors::red
      } };

      circle_t circle{ {
        .position = shared_pos,
        .radius = 50,
        .color = fan::colors::blue
      } };

      sprite_t sprite{ {
        .position = shared_pos,
        .size = fan::vec2(50, 50),
        .color = fan::colors::green
      } };

      assert_equal(rect.get_position(), shared_pos, "Rectangle position mismatch");
      assert_equal(circle.get_position(), shared_pos, "Circle position mismatch");
      assert_equal(sprite.get_position(), shared_pos, "Sprite position mismatch");

      fan::vec3 new_pos(200, 200, 0);
      rect.set_position(new_pos);

      assert_equal(rect.get_position(), new_pos, "Rectangle should update");
      assert_equal(circle.get_position(), shared_pos, "Circle should not change");
      assert_equal(sprite.get_position(), shared_pos, "Sprite should not change");
    });
  }

  void benchmark_shape_creation() {
    run_benchmark("Rectangle Creation", 10000, [&]() {
      rectangle_t rect{ {
        .position = fan::vec3(100, 200, 5),
        .size = fan::vec2(50, 60),
        .color = fan::colors::red
      } };
    });

    run_benchmark("Sprite Creation", 10000, [&]() {
      sprite_t sprite{ {
        .position = fan::vec3(100, 200, 3),
        .size = fan::vec2(32, 32),
        .image = engine.default_texture
      } };
    });

    run_benchmark("Circle Creation", 10000, [&]() {
      circle_t circle{ {
        .position = fan::vec3(100, 200, 5),
        .radius = 50,
        .color = fan::colors::blue
      } };
    });
  }

  void benchmark_property_updates() {
    rectangle_t rect{ {.position = fan::vec3(0, 0, 0)} };

    run_benchmark("Position Update", 100000, [&]() {
      rect.set_position(fan::vec3(100, 200, 5));
    });

    run_benchmark("Size Update", 100000, [&]() {
      rect.set_size(fan::vec2(100, 100));
    });

    run_benchmark("Color Update", 100000, [&]() {
      rect.set_color(fan::colors::red);
    });

    run_benchmark("Angle Update", 100000, [&]() {
      rect.set_angle(fan::vec3(0, 0, 0.5f));
    });
  }

  void benchmark_getters() {
    rectangle_t rect{ {
      .position = fan::vec3(100, 200, 5),
      .size = fan::vec2(50, 60),
      .color = fan::colors::red
    } };

    fan::vec3 pos;
    run_benchmark("Position Get", 100000, [&]() {
      pos = rect.get_position();
    });

    fan::vec2 size;
    run_benchmark("Size Get", 100000, [&]() {
      size = rect.get_size();
    });

    fan::color color;
    run_benchmark("Color Get", 100000, [&]() {
      color = rect.get_color();
    });
  }

  void print_test_results() {
    fan::print_color(fan::colors::cyan, "\n======== TEST RESULTS ========\n");

    int passed = 0;
    int failed = 0;

    for (const auto& result : test_results) {
      std::string time = "(" + std::to_string(result.duration_ms) + "ms)";

      if (result.passed) {
        fan::print_success("[PASS] " + result.name + " " + time);
        passed++;
      }
      else {
        fan::print_error("[FAIL] " + result.name);
        fan::print_color(fan::colors::red, "       Error: " + result.error_message);
        failed++;
      }
    }

    fan::print_color(fan::colors::white, "\n--------------------------------");
    fan::print_success("Passed: " + std::to_string(passed));
    fan::print_error("Failed: " + std::to_string(failed));
    fan::print_color(fan::colors::yellow, "Total:  " + std::to_string(test_results.size()));
  }

  void print_benchmark_results() {
    fan::print_color(fan::colors::cyan, "\n======== BENCHMARK RESULTS ========\n");

    for (const auto& result : benchmark_results) {
      fan::print_color(fan::colors::yellow, result.name + ":");
      fan::print("  Iterations : " + std::to_string(result.iterations));
      fan::print("  Total Time : " + std::to_string(result.total_time_ms) + " ms");
      fan::print("  Avg Time   : " + std::to_string(result.avg_time_us) + " us");
      fan::print("  Ops/sec    : " + std::to_string(result.ops_per_sec));
      fan::print("");
    }
  }

  void run_all_tests() {
    fan::print_color(fan::colors::green, "\n=== Running Shape System Tests ===\n");

    test_keypack_integrity();
    test_position_operations();
    test_size_operations();
    test_color_operations();
    test_angle_operations();
    test_camera_viewport_operations();
    test_image_operations();
    test_shape_specific_properties();
    test_keypack_changes();
    test_sequential_updates();
    test_pixel_accuracy();
    test_depth_ordering();

    test_camera_position();
    test_viewport_resize();
  #if defined(fan_physics)
    test_collision_detection();
  #endif
    test_image_assignment();
    test_rotation_matrix();
    test_render_view_assignment();
    test_coordinate_conversion();
    test_shape_independence();

    // todo: add copy and move constructor verification to shape_t, base_shape_t, character2d_t

    print_test_results();
  }

  void run_all_benchmarks() {
    fan::print_color(fan::colors::green, "\n=== Running Shape System Benchmarks ===\n");

    benchmark_shape_creation();
    benchmark_property_updates();
    benchmark_getters();

    print_benchmark_results();
  }

  engine_t engine{ {
    .renderer = fan::window_t::renderer_t::opengl,
  } };

  std::vector<test_result_t> test_results;
  std::vector<benchmark_result_t> benchmark_results;
};

int main() {
  shape_tester_t tester;

  tester.run_all_tests();
  tester.run_all_benchmarks();
  //
  fan::print_success("\n=== ALL TESTS AND BENCHMARKS COMPLETE ===\n");

  return 0;
}////