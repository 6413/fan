#include <fan/utility.h>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>
import fan;
#include <fan/graphics/types.h>
using namespace fan::graphics;

#define BLL_set_prefix bll
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeData \
  fan::vec3 pos; \
  fan::color col;
#include <BLL/BLL.h>

template <typename T>
void do_not_optimize(const T& val) {
  volatile const T* ptr = &val;
  (void)*ptr;
}

struct perf_result_t {
  std::string category;
  std::string name;
  f64_t throughput;
  f64_t ms;
  const char* unit;
};

struct perf_suite_t {
  perf_suite_t() {
    engine.set_vsync(false);
    engine.set_culling_enabled(false);
  }

  template <typename Setup, typename Func, typename Teardown>
  void run(
    const std::string& category,
    const std::string& name,
    uint64_t batch_size,
    const char* unit,
    f64_t multiplier,
    Setup&& setup,
    Func&& bench,
    Teardown&& teardown
  ) {
    setup();
    bench(1);

    uint64_t iters = 0;
    fan::time::timer t(true);
    while (t.elapsed() / 1e6 < 1000.0) {
      bench(batch_size);
      iters += batch_size;
    }
    f64_t ms = t.elapsed() / 1e6;

    teardown();

    f64_t throughput = ((iters * multiplier) / ms) * 1000.0;
    results.push_back({category, name, throughput, ms, unit});

    char buf[160];
    snprintf(buf, sizeof(buf),
      "[%-8s] %-50s | %10.0f %-8s | %8.2f ms",
      category.c_str(),
      name.c_str(),
      throughput,
      unit,
      ms
    );
    fan::print_color(fan::colors::cyan, buf);
  }

  void run(const std::string& cat, const std::string& name, uint64_t batch_size, auto&& bench) {
    run(cat, name, batch_size, "ops/sec", 1.0, []{}, bench, []{});
  }

  void run(const std::string& cat, const std::string& name, uint64_t batch_size, auto&& setup, auto&& bench, auto&& teardown) {
    run(cat, name, batch_size, "ops/sec", 1.0, setup, bench, teardown);
  }

  void execute() {
    fan::vec2i res = engine.window.get_size();
    fan::print_color(fan::colors::yellow, "Resolution  : ", res.x, "x", res.y);
    fan::print_color(fan::colors::yellow, "Renderer    : ", engine.get_renderer_string());
    fan::print_color(fan::colors::yellow, "Platform    : ", engine.get_platform_string());
    fan::print_color(fan::colors::yellow, "Build       : ", engine.get_build_string());
    fan::print_color(fan::colors::yellow, "Physics     : ", engine.get_physics_string());

    run("MATH", "Vec3 bulk arithmetic (add/sub/mul/div)", 10000, [&](uint64_t n) {
      fan::vec3 a(1.5f, 2.5f, 3.5f), b(0.5f, 1.0f, 2.0f), r(0);
      for (uint64_t i = 0; i < n; ++i) {
        r = r + (a * b) - (a / 1.5f);
        a.x += 0.001f;
      }
      do_not_optimize(r);
    });

    run("MATH", "Vec3 dot/normalize/cross", 10000, [&](uint64_t n) {
      fan::vec3 a(1.0f, 2.0f, 3.0f), b(3.0f, 2.0f, 1.0f);
      f32_t acc = 0;
      for (uint64_t i = 0; i < n; ++i) {
        acc += a.dot(b);
        a = a.cross(b).normalize();
        b += 0.01f;
      }
      do_not_optimize(acc);
    });

    run("MATH", "Mat4 transformations (translate/rotate/scale)", 10000, [&](uint64_t n) {
      fan::mat4 m;
      for (uint64_t i = 0; i < n; ++i) {
        m = fan::mat4::identity()
          .translate(fan::vec3(i * 0.1f, 5.0f, -2.0f))
          .rotate(fan::math::pi * 0.5f, fan::vec3(0, 1, 0))
          .scale(fan::vec3(2.0f));
      }
      do_not_optimize(m);
    });

    run("MATH", "Mat4 inversion", 10000, [&](uint64_t n) {
      fan::mat4 m = fan::mat4::identity().translate(fan::vec3(1, 2, 3)).rotate(0.5f, fan::vec3(1, 0, 0));
      fan::mat4 r;
      for (uint64_t i = 0; i < n; ++i) {
        r = m.inverse();
        m[3][0] += 0.01f;
      }
      do_not_optimize(r);
    });

    bll_t test_bll;
    run("MEMORY", "BLL push/iterate/erase (1M nodes)", 1,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          for (uint32_t i = 0; i < 1000000; ++i) {
            auto nr = test_bll.NewNodeLast();
            test_bll[nr].pos = fan::vec3(i);
            test_bll[nr].col = fan::colors::white;
          }
          f32_t acc = 0;
          for (auto nr = test_bll.GetNodeFirst(); nr != test_bll.dst; nr = nr.Next(&test_bll)) {
            acc += test_bll[nr].pos.x;
          }
          do_not_optimize(acc);
          test_bll.Clear();
        }
      },
      [&]{}
    );

    std::vector<rectangle_t> rects;
    run("GRAPHICS", "Instantiate 500k rectangles", 1,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          rects.reserve(500000);
          for (int i = 0; i < 500000; ++i) {
            rects.emplace_back(rectangle_t{{
              .position = fan::vec3(0), .size = fan::vec2(10), .color = fan::colors::red
            }});
          }
        }
      },
      [&]{}
    );

    run("GRAPHICS", "Update 50k rects (position/color/angle)", 100,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          for (int i = 0; i < 50000; ++i) {
            rects[i].set_position(fan::vec3(step, i, 0));
            rects[i].set_color(fan::colors::blue);
            rects[i].set_angle(fan::vec3(0, 0, step * 0.01f));
          }
        }
      },
      [&]{}
    );

    std::vector<circle_t> circles;
    run("GRAPHICS", "Instantiate 50k circles", 1,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          circles.reserve(50000);
          for (int i = 0; i < 50000; ++i) {
            circles.emplace_back(circle_t{{
              .position = fan::vec3(0), .radius = 10.f, .color = fan::colors::green
            }});
          }
        }
      },
      [&]{}
    );

    run("GRAPHICS", "Update 50k circles (position/color)", 100,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          for (int i = 0; i < 50000; ++i) {
            circles[i].set_position(fan::vec3(step, i, 0));
            circles[i].set_color(fan::colors::red);
          }
        }
      },
      [&]{ circles.clear(); }
    );

    std::vector<sprite_t> sprites;
    run("GRAPHICS", "Instantiate 50k sprites", 1,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          sprites.reserve(50000);
          for (int i = 0; i < 50000; ++i) {
            sprites.emplace_back(sprite_t{{
              .position = fan::vec3(0), .size = fan::vec2(10)
            }});
          }
        }
      },
      [&]{}
    );

    run("GRAPHICS", "Update 50k sprites (position/size)", 100,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          for (int i = 0; i < 50000; ++i) {
            sprites[i].set_position(fan::vec3(step, i, 0));
            sprites[i].set_size(fan::vec2(step * 0.01f + 1.f));
          }
        }
      },
      [&]{ sprites.clear(); }
    );

    std::vector<line_t> lines;
    run("GRAPHICS", "Instantiate 50k lines", 1,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          lines.reserve(50000);
          for (int i = 0; i < 50000; ++i) {
            lines.emplace_back(line_t{{
              .src = fan::vec3(0), .dst = fan::vec2(100, 100), .color = fan::colors::white
            }});
          }
        }
      },
      [&]{}
    );

    run("GRAPHICS", "Update 50k lines (set_line)", 100,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          for (int i = 0; i < 50000; ++i) {
            lines[i].set_line(fan::vec2(step, i), fan::vec2(step + 100, i + 100));
          }
        }
      },
      [&]{ lines.clear(); rects.clear(); }
    );

    run("GRAPHICS", "Rectangle create+destroy (10k)", 10,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          std::vector<rectangle_t> tmp;
          tmp.reserve(10000);
          for (int i = 0; i < 10000; ++i) {
            tmp.emplace_back(rectangle_t{{
              .position = fan::vec3(0), .size = fan::vec2(10), .color = fan::colors::red
            }});
          }
        }
      },
      [&]{}
    );

    std::vector<rectangle_t> pipe_rects;
    for (int i = 0; i < 500000; ++i) {
      pipe_rects.emplace_back(rectangle_t{{
        .position = fan::vec3(0), .size = fan::vec2(10), .color = fan::colors::red
      }});
    }

    run("PIPELINE", "Process frame (500k rects, culling OFF)", 10,
      [&]{ engine.set_culling_enabled(false); },
      [&](uint64_t n) { for (uint64_t i = 0; i < n; ++i) engine.process_frame(); },
      [&]{}
    );

    run("PIPELINE", "Process frame (500k rects, culling ON)", 10,
      [&]{ engine.set_culling_enabled(true); engine.rebuild_static_culling(); },
      [&](uint64_t n) { for (uint64_t i = 0; i < n; ++i) engine.process_frame(); },
      [&]{ pipe_rects.clear(); engine.set_culling_enabled(false); }
    );

    std::vector<shape_t> mixed;
    run("PIPELINE", "Process frame (mixed: 100k rect+circle+sprite)", 10,
      [&]{
        engine.set_culling_enabled(false);
        mixed.reserve(300000);
        for (int i = 0; i < 100000; ++i) {
          mixed.emplace_back(rectangle_t{{ .position = fan::vec3(i % 1920, i % 1080, 0), .size = fan::vec2(5), .color = fan::colors::red }});
          mixed.emplace_back(circle_t{{ .position = fan::vec3(i % 1920, i % 1080, 0), .radius = 5.f, .color = fan::colors::green }});
          mixed.emplace_back(sprite_t{{ .position = fan::vec3(i % 1920, i % 1080, 0), .size = fan::vec2(5) }});
        }
      },
      [&](uint64_t n) { for (uint64_t i = 0; i < n; ++i) engine.process_frame(); },
      [&]{ mixed.clear(); }
    );

    run("GRAPHICS", "Rectangle create+destroy churn (10k)", 100,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          std::vector<rectangle_t> tmp;
          tmp.reserve(10000);
          for (int i = 0; i < 10000; ++i) {
            tmp.emplace_back(rectangle_t{{
              .position = fan::vec3(i, 0, 0), .size = fan::vec2(5), .color = fan::colors::white
            }});
          }
        }
      },
      [&]{}
    );

    run("MISC", "Camera set_position (100k calls)", 10000,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
          engine.camera_set_position(engine.orthographic_render_view.camera, fan::vec3(i * 0.1f, i * 0.1f, 0));
        }
      },
      [&]{ engine.camera_set_position(engine.orthographic_render_view.camera, fan::vec3(0)); }
    );

    run("MISC", "Screen to world (100k conversions)", 10000,
      [&]{},
      [&](uint64_t n) {
        fan::vec2 acc(0);
        for (uint64_t i = 0; i < n; ++i) {
          acc += fan::graphics::screen_to_world(fan::vec2(i % 1920, i % 1080), engine.orthographic_render_view);
        }
        do_not_optimize(acc);
      },
      [&]{}
    );

    run("IMAGES", "CPU image create (solid color)", 100,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
          auto img = engine.image_create(fan::random::color());
          engine.image_unload(img);
        }
      },
      [&]{}
    );

    {
      fan::image::info_t info;
      std::vector<uint8_t> px(1024 * 1024 * 4);
      for (auto& b : px) {
        b = 128;
      }
      info.data = px.data();
      info.size = fan::vec2ui(1024, 1024);
      fan::graphics::image_load_properties_t props;
      props.internal_format = fan::graphics::image_format_e::rgba_unorm;
      props.format          = fan::graphics::image_format_e::rgba_unorm;

      run("IMAGES", "GPU texture upload (1024x1024 RGBA)", 50,
        [&]{},
        [&](uint64_t n) {
          for (uint64_t i = 0; i < n; ++i) {
            auto img = engine.image_load(info, props);
            engine.image_unload(img);
          }
        },
        [&]{}
      );
    }

    std::vector<tilemap_t> tmaps;
    run("TILEMAP", "Create 256x256 grid", 10,
      [&] { tmaps.resize(50); },
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          tmaps[step] = tilemap_t(fan::vec2(16), fan::colors::black, fan::vec2(256 * 16), fan::vec3(0));
        }
      },
      [&] { tmaps.clear(); }
    );

    tilemap_t tmap(fan::vec2(16), fan::colors::black, fan::vec2(256 * 16), fan::vec3(0));
    run("TILEMAP", "Fill 256x256 grid colors", 10,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          tmap.fill_colors(fan::colors::white);
        }
      },
      [&]{ tmap = tilemap_t{}; }
    );

    fan::noise_t pnoise;
    run("ALGO", "Fractal noise generation (512x512)", 10,
      [&]{ pnoise.octaves = 4; pnoise.frequency = 0.01f; },
      [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
          pnoise.seed = i;
          pnoise.apply();
          auto d = pnoise.generate_data(fan::vec2(512));
          do_not_optimize(d.size());
        }
      },
      [&]{}
    );

#if defined(FAN_PHYSICS_2D)
    std::vector<fan::physics::entity_t> bodies;
    run("PHYSICS", "Step simulation (20k dynamic bodies)", 100,
      [&]{
        engine.update_physics(true);
        engine.get_physics_context().create_box(fan::vec2(0, 1000), fan::vec2(5000, 50), 0,
          fan::physics::body_type_e::static_body);
        bodies.reserve(20000);
        for (int i = 0; i < 20000; ++i) {
          bodies.push_back(engine.get_physics_context().create_box(
            fan::vec2(i % 200 * 10, (i / 200) * 10), fan::vec2(5)));
        }
      },
      [&](uint64_t n) { for (uint64_t i = 0; i < n; ++i) engine.process_frame(); },
      [&]{
        for (auto& b : bodies) {
          b.destroy();
        }
        bodies.clear();
        engine.update_physics(false);
      }
    );
#endif

    std::string io_file = "fan_bench.tmp";
    std::string io_data(1024 * 1024 * 5, 'X');
    
    run("IO", "Sync write 5MB block", 10, "MB/s", 5.0,
      [&]{ std::filesystem::remove(io_file); },
      [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
          fan::io::file::write(io_file, io_data, std::ios_base::binary | std::ios_base::app);
        }
      },
      [&]{}
    );

    run("IO", "Sync read 250MB chunked", 10, "MB/s", 250.0,
      [&]{},
      [&](uint64_t n) {
        std::string read_data;
        for (uint64_t i = 0; i < n; ++i) {
          fan::io::file::read(io_file, &read_data);
          do_not_optimize(read_data.size());
        }
      },
      [&]{ std::filesystem::remove(io_file); }
    );
  }

  engine_t engine{{
    .window_size = {1920, 1080},
    .renderer = fan::window_t::renderer_t::opengl,
  }};
  std::vector<perf_result_t> results;
};

int main() {
  perf_suite_t suite;
  suite.execute();
  return 0;
}