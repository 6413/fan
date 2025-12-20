// Library usage includes
//-------------------------------------------------------------------
#define stage_loader_path .
#include <fan/graphics/gui/stage_maker/loader.h>
//-------------------------------------------------------------------


// Library usage samples, these structs expect there is an instance of a fan::graphics::engine_t.
// They use "gloco" to access global engine pointer
struct library_usage_t {
  struct gui {
    //-------------------------------------------------------------------
    // Demonstrates GUI console print
    struct console_print {
      console_print() {
        fan::printcl("Random number on construct(0-5):" + fan::random::value(0, 5));
      }
    };
    //-------------------------------------------------------------------
    // Demonstrates custom GUI console commands
    struct console {
      console() {
        gloco()->console.commands.add("test", [&](fan::console_t* self, const fan::commands_t::arg_t& args) {
          fan::printcl("test command executed");
        });
        gloco()->console.commands.add("sum", [&](fan::console_t* self, const fan::commands_t::arg_t& args) {
          if (args.size() == 2) {
            int a = std::stoi(args[0]);
            int b = std::stoi(args[1]);
            fan::printcl("Result: " + std::to_string(a + b));
          }
        });
      }
      void update() {

      }
      void close() {
        gloco()->console.commands.remove("test");
        gloco()->console.commands.remove("sum");
      }
    };
    //-------------------------------------------------------------------
  };
  struct input {
    //-------------------------------------------------------------------
    // Demonstrates input action system with key bindings and combos
    struct action {
      action() {
        gloco()->input_action.add({ fan::key_space }, "jump");
        gloco()->input_action.add_keycombo({ fan::key_space, fan::key_a }, "combo_test");
      }
      void update() {
        if (gloco()->input_action.is_active("jump")) {
          fan::print("jump");
        }
        if (gloco()->input_action.is_active("combo_test")) {
          fan::print("combo_test");
        }
      }
      void close() {
        gloco()->input_action.remove("jump");
        gloco()->input_action.remove("combo_test");
      }
    };
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    // Demonstrates interactive visual elements
    struct vfi_interactive {
      fan::graphics::rectangle_t rectangle;
      fan::graphics::vfi_root_t vfi_root;

      vfi_interactive() {
        fan::graphics::vfi_t::properties_t vfip;
        vfip.shape_type = fan::graphics::vfi_t::shape_t::rectangle;
        vfip.shape.rectangle->position = fan::vec3(500, 500, 1);
        vfip.shape.rectangle->size = fan::vec2(100, 100);
        vfip.shape.rectangle->size.x /= 2; // hitbox takes half size
        vfip.shape.rectangle->camera = gloco()->orthographic_render_view.camera;
        vfip.shape.rectangle->viewport = gloco()->orthographic_render_view.viewport;

        vfip.mouse_button_cb = [](const fan::graphics::vfi_t::mouse_button_data_t& data) -> int {
          fan::print("Rectangle clicked.");
          return 0;
        };

        rectangle = fan::graphics::rectangle_t{ {
            .position = fan::vec3(*(fan::vec2*)&vfip.shape.rectangle->position, 0),
            .size = vfip.shape.rectangle->size,
            .color = fan::colors::red
          } };

        vfi_root.set_root(vfip);
      }

      void update() {
        fan::graphics::gui::text("Click the red rectangle");
      }
    };
    //-------------------------------------------------------------------
  };
  struct event {
    //-------------------------------------------------------------------
    // Demonstrates coroutines using fan::event::task_t by respawning rectangles every 1s
    struct task {
      fan::event::task_t spawn_rectangles() {
        while (true) {
          shapes.clear();
          shapes.push_back(fan::graphics::rectangle_t{ {
              .position = fan::vec3(fan::random::vec2(0, 400), 0),
              .size = fan::random::vec2(100, 400),
              .color = fan::random::color()
            } });
          co_await fan::co_sleep(1000); // async sleep - doesn't block main thread
        }
      }
      task() {
        task_spawn_rectangles = spawn_rectangles(); // Start the coroutine
      }
      fan::event::task_t task_spawn_rectangles;
      std::vector<fan::graphics::shape_t> shapes;
    };
    //-------------------------------------------------------------------

    //-------------------------------------------------------------------
  };
  struct io {
    //-------------------------------------------------------------------
    // Demonstrates shape serialization to JSON
    struct shape_serialization {
      shape_serialization() {
        // Test shapes
        for (int i = 0; i < 10; ++i) {
          shapes.push_back(fan::graphics::rectangle_t{ {
              .position = fan::random::vec2(0, 400),
              .size = fan::random::vec2(20, 100),
              .color = fan::random::color()
            } });
        }
      }
      void update() {
        if (fan::graphics::gui::button("Serialize to JSON")) {
          fan::json out;
          for (auto& shape : shapes) {
            out += shape;
          }
          fan::io::file::write("demo_shapes.json", out.dump(2), std::ios_base::binary);
          fan::print("JSON wrote to \"demo_shapes.json\"");
        }
      }
      std::vector<fan::graphics::rectangle_t> shapes;
    };
    //-------------------------------------------------------------------
    // Demonstrates file dialog for opening files
    struct file_dialog {
      void update() {
        if (fan::graphics::gui::button("Open File")) {
          dialog.load("png,jpg;pdf", &output_path);
        }
        if (dialog.is_finished()) {
          fan::print("Selected file: " + output_path);
          dialog.finished = false;
        }
        if (!output_path.empty()) {
          fan::graphics::gui::text("Last selected: " + output_path);
        }
      }
      std::string output_path;
      fan::graphics::file_open_dialog_t dialog;
    };
    // Demonstrates file system watching for changes
    struct file_watcher {
      file_watcher() : watcher("./") {
        watcher.start([this](const std::string& filename, int events) {
          last_event = "file: " + filename;
          if (events & fan::fs_change) {
            last_event += " (modified)";
          }
          if (events & fan::fs_rename) {
            last_event += " (renamed/deleted)";
          }
          fan::print("fs event: " + last_event);
        });
      }
      void update() {
        fan::graphics::gui::text("watching current directory for file changes");
        if (!last_event.empty()) {
          fan::graphics::gui::text("last event: " + last_event);
        }
        fan::graphics::gui::text("try creating/modifying files to see events");
      }
      fan::event::fs_watcher_t watcher;
      std::string last_event;
    };
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    // Demonstrates async file operations with coroutines
    struct async_file {
      fan::event::task_t read_file_async(const std::string& path) {
        int fd = co_await fan::io::file::async_open(path);
        int offset = 0;
        std::string buffer;

        while (true) {
          intptr_t result = co_await fan::io::file::async_read(fd, &buffer, offset);
          if (result == 0) {
            break;
          }
          file_content += buffer;
          offset += result;
          co_await fan::co_sleep(10);
        }
        co_await fan::io::file::async_close(fd);
      }

      fan::event::task_t write_file_async(const std::string& path, const std::string& data) {
        int fd = co_await fan::io::file::async_open(path, fan::fs_out);
        size_t total_written = 0;
        size_t buffer_size = 4096;

        while (total_written < data.size()) {
          size_t remaining = data.size() - total_written;
          size_t to_write = std::min(remaining, buffer_size);
          std::string buffer(data.data() + total_written, to_write);
          size_t written = co_await fan::io::file::async_write(fd, buffer.data(), buffer.size(), total_written);
          total_written += written;
        }
        co_await fan::io::file::async_close(fd);
      }

      async_file() {
        task_handle = read_file_async("CMakeLists.txt");
      }

      void update() {
        if (fan::graphics::gui::button("Read file")) {
          file_content.clear();
          task_handle = read_file_async("CMakeLists.txt");
        }
        if (fan::graphics::gui::button("Write file")) {
          task_handle = write_file_async("output.txt", file_content);
        }
        if (!file_content.empty()) {
          fan::graphics::gui::text("Content: " + file_content.substr(0, 50) + "...");
        }
      }
      fan::event::task_t task_handle;
      std::string file_content;
    };
    //-------------------------------------------------------------------
    // Demonstrates async directory iteration with image loading and GUI display
    struct async_directory {
      async_directory() {
        iterator.sort_alphabetically = true;
        iterator.callback = [&](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
          std::string path_str = entry.path().generic_string();
          if (fan::image::valid(path_str)) {
            images.emplace_back(gloco()->image_load(path_str));
            co_await fan::co_sleep(100);
          }
          co_return;
        };
        fan::io::async_directory_iterate(&iterator, "imagenet-sample-images-master");
      }
      void update() {
        fan::graphics::gui::begin("images");
        f32_t thumbnail_size = 128.0f;
        f32_t panel_width = fan::graphics::gui::get_content_region_avail().x;
        f32_t padding = 16.0f;
        int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);
        fan::graphics::gui::columns(column_count, 0, false);
        for (auto& i : images) {
          fan::graphics::gui::image(i, thumbnail_size);
          fan::graphics::gui::next_column();
        }
        fan::graphics::gui::end();
      }
      std::vector<fan::graphics::image_t> images;
      fan::io::async_directory_iterator_t iterator;
    };
    //-------------------------------------------------------------------
  };
  struct system {
    //-------------------------------------------------------------------
    // Demonstrates creating and loading custom stages
    struct custom_stage {
      // make custom stage
      lstd_defstruct(custom_t)
      #include <fan/graphics/gui/stage_maker/preset.h>
        static constexpr auto stage_name = "";
      fan::graphics::rectangle_t r;
      void open(void* sod) {
        fan::print("opened");
      }
      void close() {
        fan::print("closed");
      }
      void window_resize() {////
        fan::print("resized");
      }
      void update() {
        fan::print("update");
      }
    };
    stage_loader_t* stage_loader;
    stage_loader_t::nr_t stage_handle;
    custom_stage() {
      stage_loader = new stage_loader_t;
      stage_loader_t::stage_open_properties_t op;
      stage_handle = stage_loader->open_stage<custom_t>(op);
    }
    void close() {
      stage_loader->erase_stage(stage_handle);
      delete stage_loader;
    }
    void update() {
      fan::graphics::gui::text("Custom stage running");
    }
  };//
};
//-------------------------------------------------------------------
};//