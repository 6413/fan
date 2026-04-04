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
  uint64_t iterations;
  uint64_t work_per_iter;
  f64_t ms;
  f64_t units_per_sec;
  f64_t score;
};

// score = total_units_processed / second
// each test declares its actual work unit count so all categories are comparable
struct perf_suite_t {
  engine_t engine{{
    .window_size = {1920, 1080},
    .renderer = fan::window_t::renderer_t::opengl,
  }};
  std::vector<perf_result_t> results;
  f64_t total_score = 0;

  perf_suite_t() {
    engine.set_vsync(false);
    engine.set_culling_enabled(false);
  }

  template <typename Setup, typename Func, typename Teardown>
  void run(
    const std::string& category,
    const std::string& name,
    uint64_t iters,
    uint64_t work_per_iter,           // actual units of work per iteration
    Setup&& setup,
    Func&& bench,
    Teardown&& teardown
  ) {
    setup();
    bench(std::min<uint64_t>(10, iters));

    fan::time::timer t(true);
    bench(iters);
    f64_t ms = std::max(t.elapsed() / 1e6, 0.001);

    teardown();

    f64_t total_work  = (f64_t)iters * (f64_t)work_per_iter;
    f64_t units_per_s = (total_work / ms) * 1000.0;
    f64_t score       = units_per_s;

    results.push_back({category, name, iters, work_per_iter, ms, units_per_s, score});
    total_score += score;

    fan::print_color(fan::colors::cyan, "[", category, "] ", name);
    fan::print("  Time:", ms, "ms | Units/sec:", (uint64_t)units_per_s, "| Score:", (uint64_t)score);
  }

  void run(const std::string& cat, const std::string& name, uint64_t iters, uint64_t wpiter, auto&& bench) {
    run(cat, name, iters, wpiter, []{}, bench, []{});
  }

  void execute() {
    fan::vec2i res = engine.window.get_size();
    fan::print_color(fan::colors::green, "=== fan Engine Comprehensive Performance Suite ===\n");
    fan::print_color(fan::colors::yellow, "Resolution  : ", res.x, "x", res.y);
    fan::print_color(fan::colors::yellow, "Renderer : ", engine.get_renderer_string());
    fan::print_color(fan::colors::yellow, "Platform : ", engine.get_platform_string());
    fan::print_color(fan::colors::yellow, "Build    : ", engine.get_build_string());
    fan::print_color(fan::colors::yellow, "Physics  : ", engine.get_physics_string());

    // ── MATH ─────────────────────────────────────── work unit = 1 op

    run("MATH", "Vec3 bulk arithmetic (add/sub/mul/div)", 5000000, 1, [&](uint64_t n) {
      fan::vec3 a(1.5f, 2.5f, 3.5f), b(0.5f, 1.0f, 2.0f), r(0);
      for (uint64_t i = 0; i < n; ++i) {
        r = r + (a * b) - (a / 1.5f);
        a.x += 0.001f;
      }
      do_not_optimize(r);
    });

    run("MATH", "Vec3 dot/normalize/cross", 2000000, 1, [&](uint64_t n) {
      fan::vec3 a(1.0f, 2.0f, 3.0f), b(3.0f, 2.0f, 1.0f);
      f32_t acc = 0;
      for (uint64_t i = 0; i < n; ++i) {
        acc += a.dot(b);
        a = a.cross(b).normalize();
        b += 0.01f;
      }
      do_not_optimize(acc);
    });

    run("MATH", "Mat4 transformations (translate/rotate/scale)", 1000000, 1, [&](uint64_t n) {
      fan::mat4 m;
      for (uint64_t i = 0; i < n; ++i) {
        m = fan::mat4::identity()
          .translate(fan::vec3(i * 0.1f, 5.0f, -2.0f))
          .rotate(fan::math::pi * 0.5f, fan::vec3(0, 1, 0))
          .scale(fan::vec3(2.0f));
      }
      do_not_optimize(m);
    });

    run("MATH", "Mat4 inversion", 500000, 1, [&](uint64_t n) {
      fan::mat4 m = fan::mat4::identity().translate(fan::vec3(1, 2, 3)).rotate(0.5f, fan::vec3(1, 0, 0));
      fan::mat4 r;
      for (uint64_t i = 0; i < n; ++i) {
        r = m.inverse();
        m[3][0] += 0.01f;
      }
      do_not_optimize(r);
    });

    // ── MEMORY ───────────────────────────────────── work unit = 1 node

    bll_t test_bll;
    run("MEMORY", "BLL push/iterate/erase (1M nodes)", 1, 1000000,
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

    // ── GRAPHICS ─────────────────────────────────── work unit = 1 shape

    // ── SHAPES ───────────────────────────────────── work unit = 1 shape

    std::vector<rectangle_t> rects;
    run("GRAPHICS", "Instantiate 500k rectangles", 1, 500000,
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

    run("GRAPHICS", "Update 50k rects (position/color/angle)", 100, 50000 * 3,
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
    run("GRAPHICS", "Instantiate 50k circles", 1, 50000,
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

    run("GRAPHICS", "Update 50k circles (position/color)", 100, 50000 * 2,
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
    run("GRAPHICS", "Instantiate 50k sprites", 1, 50000,
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

    run("GRAPHICS", "Update 50k sprites (position/size)", 100, 50000 * 2,
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
    run("GRAPHICS", "Instantiate 50k lines", 1, 50000,
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

    run("GRAPHICS", "Update 50k lines (set_line)", 100, 50000,
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

    // ── SHAPE ERASE ──────────────────────────────── work unit = 1 shape destroyed

    run("GRAPHICS", "Rectangle create+destroy (10k)", 10, 10000,
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
          // destruction happens here when tmp goes out of scope
        }
      },
      [&]{}
    );

    // ── PIPELINE ─────────────────────────────────── work unit = 1 shape rendered

    std::vector<rectangle_t> pipe_rects;
    for (int i = 0; i < 500000; ++i) {
      pipe_rects.emplace_back(rectangle_t{{
        .position = fan::vec3(0), .size = fan::vec2(10), .color = fan::colors::red
      }});
    }

    run("PIPELINE", "Process frame (500k rects, culling OFF)", 100, 500000,
      [&]{ engine.set_culling_enabled(false); },
      [&](uint64_t n) { for (uint64_t i = 0; i < n; ++i) engine.process_frame(); },
      [&]{}
    );

    run("PIPELINE", "Process frame (500k rects, culling ON)", 100, 500000,
      [&]{ engine.set_culling_enabled(true); engine.rebuild_static_culling(); },
      [&](uint64_t n) { for (uint64_t i = 0; i < n; ++i) engine.process_frame(); },
      [&]{ pipe_rects.clear(); engine.set_culling_enabled(false); }
    );

    // ── MIXED PIPELINE ───────────────────────────── mixed shape types

    std::vector<shape_t> mixed;
    run("PIPELINE", "Process frame (mixed: 100k rect+circle+sprite)", 100, 300000,
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

    // ── SHAPE POOL CHURN ─────────────────────────── work unit = 1 create+destroy cycle

    run("GRAPHICS", "Rectangle create+destroy churn (10k)", 100, 10000,
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

    // ── CAMERA ───────────────────────────────────── work unit = 1 camera op

    run("MISC", "Camera set_position (100k calls)", 100000, 1,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
          engine.camera_set_position(engine.orthographic_render_view.camera, fan::vec3(i * 0.1f, i * 0.1f, 0));
        }
      },
      [&]{ engine.camera_set_position(engine.orthographic_render_view.camera, fan::vec3(0)); }
    );

    // ── SPATIAL QUERY ────────────────────────────── work unit = 1 query

    run("MISC", "Screen to world (100k conversions)", 100000, 1,
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

    // ── IMAGES ───────────────────────────────────── work unit = 1 pixel

    run("IMAGES", "CPU image create (solid color)", 1000, 1,
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
      for (auto& b : px) b = 128;
      info.data = px.data();
      info.size = fan::vec2ui(1024, 1024);
      fan::graphics::image_load_properties_t props;
      props.internal_format = fan::graphics::image_format_e::rgba_unorm;
      props.format          = fan::graphics::image_format_e::rgba_unorm;

      run("IMAGES", "GPU texture upload (1024×1024 RGBA)", 50, 1024 * 1024,
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

    // ── TILEMAP ──────────────────────────────────── work unit = 1 tile

    std::vector<tilemap_t> tmaps;
    run("TILEMAP", "Create 256x256 grid", 50, 256 * 256,
      [&] { tmaps.resize(50); },
        [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          tmaps[step] = tilemap_t(fan::vec2(16), fan::colors::black, fan::vec2(256 * 16), fan::vec3(0));
        }
      },
      [&] { tmaps.clear(); }
    );

    tilemap_t tmap(fan::vec2(16), fan::colors::black, fan::vec2(256 * 16), fan::vec3(0));
    run("TILEMAP", "Fill 256x256 grid colors", 50, 256 * 256,
      [&]{},
      [&](uint64_t n) {
        for (uint64_t step = 0; step < n; ++step) {
          tmap.fill_colors(fan::colors::white);
        }
      },
      [&]{ tmap = tilemap_t{}; }
    );

    // ── ALGO ─────────────────────────────────────── work unit = 1 pixel generated

    fan::noise_t pnoise;
    run("ALGO", "Fractal noise generation (512×512)", 100, 512 * 512,
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

    // ── PHYSICS ──────────────────────────────────── work unit = 1 body stepped

#if defined(FAN_PHYSICS_2D)
    std::vector<fan::physics::entity_t> bodies;
    run("PHYSICS", "Step simulation (20k dynamic bodies)", 500, 20000,
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
        for (auto& b : bodies) b.destroy();
        bodies.clear();
        engine.update_physics(false);
      }
    );
#endif

    // ── IO ───────────────────────────────────────── work unit = 1 KB

    std::string io_file = "fan_bench.tmp";
    std::string io_data(1024 * 1024 * 5, 'X');
    run("IO", "Sync write 5MB block", 50, 5120,                    // 5120 KB
      [&]{ std::filesystem::remove(io_file); },
      [&](uint64_t n) {
        for (uint64_t i = 0; i < n; ++i)
          fan::io::file::write(io_file, io_data, std::ios_base::binary | std::ios_base::app);
      },
      [&]{}
    );

    run("IO", "Sync read 250MB chunked", 50, 256000,               // 256000 KB ≈ 250MB
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

    print_report();
  }

  void print_report() {
    fan::print_color(fan::colors::green, "\n=== PERFORMANCE REPORT ===");

    std::string cur_cat;
    for (const auto& r : results) {
      if (r.category != cur_cat) {
        fan::print_color(fan::colors::yellow, "\n[" + r.category + "]");
        cur_cat = r.category;
      }
      char buf[160];
      snprintf(buf, sizeof(buf),
        "  %-50s | %8.2f ms | %10.0f units/s | Score: %10.0f",
        r.name.c_str(), r.ms, r.units_per_sec, r.score);
      fan::print(buf);
    }

    fan::print_color(fan::colors::cyan, "\n=== SCORING REFERENCE ===");
    fan::print("  > 1,000,000,000 : S Tier  (exceptional throughput)");
    fan::print("  > 100,000,000   : A Tier  (excellent, typical GPU pipeline)");
    fan::print("  > 10,000,000    : B Tier  (good, safe for 144Hz budgets)");
    fan::print("  > 1,000,000     : C Tier  (acceptable for heavy subsystems)");
    fan::print("  < 1,000,000     : Bottleneck risk");

    fan::print_color(fan::colors::green, "\nTOTAL ENGINE SCORE: ", (uint64_t)total_score);
  }
};

int main() {
  perf_suite_t suite;
  suite.execute();
  return 0;
}