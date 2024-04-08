#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

//#define loco_imgui

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif

#ifndef FAN_INCLUDE_PATH
#define _FAN_PATH(p0) <fan/p0>
#else
#define FAN_INCLUDE_PATH_END fan/
#define _FAN_PATH(p0) <FAN_INCLUDE_PATH/fan/p0>
#define _FAN_PATH_QUOTE(p0) STRINGIFY_DEFINE(FAN_INCLUDE_PATH) "/fan/" STRINGIFY(p0)
#endif

#if defined(loco_imgui)
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#define IMGUI_DEFINE_MATH_OPERATORS
#include _FAN_PATH(imgui/imgui.h)
#include _FAN_PATH(imgui/imgui_impl_opengl3.h)
#include _FAN_PATH(imgui/imgui_impl_glfw.h)
#include _FAN_PATH(imgui/imgui_neo_sequencer.h)
#endif

#ifndef fan_verbose_print_level
#define fan_verbose_print_level 1
#endif
#ifndef fan_debug
#define fan_debug 0
#endif
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#include <fan/graphics/loco_settings.h>
#include "loco.h"

inline global_loco_t::operator loco_t* () {
  return loco;
}

global_loco_t& global_loco_t::operator=(loco_t* l) {
  loco = l;
  return *this;
}

inline loco_t* global_loco_t::operator->() {
  return loco;
}

void loco_t::use() {
    gloco = this;
    get_context().set_current(get_window());
}

loco_t::loco_t(const properties_t& p)
#ifdef loco_window
  :
gloco_dummy(this),
window(p.window_size, fan::window_t::default_window_name, p.window_flags),
#endif
#if defined(loco_context)
context(
  #if defined(loco_window)
  &window
  #endif
)
#endif
#if defined(loco_window)
, unloaded_image(fan::webp::image_info_t{ (void*)pixel_data, 1 })
#endif
{
  m_time.start();
  #if defined(loco_window)

  #if defined(loco_opengl)
  initialize_fb_vaos(fb_vao, fb_vbo);
  #endif

  root = loco_bdbt_NewNode(&bdbt);

  // set_vsync(p.vsync);
  #if defined(loco_vfi)
  window.add_buttons_callback([this](const mouse_buttons_cb_data_t& d) {
    fan::vec2 window_size = window.get_size();
    feed_mouse_button(d.button, d.state, get_mouse_position());
    });

  window.add_keys_callback([&](const keyboard_keys_cb_data_t& d) {
    feed_keyboard(d.key, d.state);
    });

  window.add_mouse_move_callback([&](const mouse_move_cb_data_t& d) {
    feed_mouse_move(get_mouse_position());
    });

  window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
    feed_text(d.character);
    });
  #endif
  #endif
  #if fan_verbose_print_level >= 1
  #if defined(loco_opengl)
  fan::print("RENDERER BACKEND: OPENGL");
  #elif defined(loco_vulkan)
  fan::print("RENDERER BACKEND: VULKAN");
  #endif
  #endif

  #if defined(loco_letter)
  font.open(loco_font);
  #endif

  #if defined(loco_opengl)
  #if defined(loco_framebuffer)
  m_framebuffer.open(get_context());
  // can be GL_RGB16F
  m_framebuffer.bind(get_context());
  #endif
  #endif

  #if defined(loco_opengl)

  #if defined(loco_framebuffer)

  static auto load_texture = [&](fan::webp::image_info_t& image_info, auto& color_buffer, fan::opengl::GLenum attachment, bool reload = false) {
    typename loco_t::image_t::load_properties_t load_properties;
    load_properties.visual_output = fan::opengl::GL_REPEAT;
    load_properties.internal_format = fan::opengl::GL_RGBA;
    load_properties.format = fan::opengl::GL_RGBA;
    load_properties.type = fan::opengl::GL_FLOAT;
    load_properties.min_filter = fan::opengl::GL_LINEAR;
    load_properties.mag_filter = fan::opengl::GL_LINEAR;

    if (reload == true) {
      color_buffer.reload_pixels(image_info, load_properties);
    }
    else {
      color_buffer.load(image_info, load_properties);
    }
    get_context().opengl.call(get_context().opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);

    color_buffer.bind_texture();
    fan::opengl::core::framebuffer_t::bind_to_texture(get_context(), color_buffer.get_texture(), attachment);
    };

  fan::webp::image_info_t image_info;
  image_info.data = nullptr;
  image_info.size = window.get_size();

  m_framebuffer.bind(get_context());
  for (std::size_t i = 0; i < std::size(color_buffers); ++i) {
    load_texture(image_info, color_buffers[i], fan::opengl::GL_COLOR_ATTACHMENT0 + i);
  }

  window.add_resize_callback([&](const auto& d) {
    fan::webp::image_info_t image_info;
    image_info.data = nullptr;
    image_info.size = window.get_size();

    m_framebuffer.bind(get_context());
    for (std::size_t i = 0; i < std::size(color_buffers); ++i) {
      load_texture(image_info, color_buffers[i], fan::opengl::GL_COLOR_ATTACHMENT0 + i, true);
    }

    fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
    m_framebuffer.bind(get_context());
    renderbuffer_properties.size = image_info.size;
    renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
    m_rbo.set_storage(get_context(), renderbuffer_properties);

    fan::vec2 window_size = gloco->window.get_size();
    default_camera->viewport.set(fan::vec2(0, 0), d.size, d.size);
    });

  fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
  m_framebuffer.bind(get_context());
  renderbuffer_properties.size = image_info.size;
  renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
  m_rbo.open(get_context());
  m_rbo.set_storage(get_context(), renderbuffer_properties);
  renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
  m_rbo.bind_to_renderbuffer(get_context(), renderbuffer_properties);

  unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

  for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
    attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
  }

  get_context().opengl.call(get_context().opengl.glDrawBuffers, std::size(attachments), attachments);
  // finally check if framebuffer is complete
  if (!m_framebuffer.ready(get_context())) {
    fan::throw_error("framebuffer not ready");
  }

  static constexpr uint32_t mip_count = 10;
  blur[0].open(window.get_size(), mip_count);

  bloom.open();

  m_framebuffer.unbind(gloco->get_context());

  m_fbo_final_shader.open();

  m_fbo_final_shader.set_vertex(
  loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.vs))
  );
  m_fbo_final_shader.set_fragment(
  loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/effects/loco_fbo.fs))
  );
  m_fbo_final_shader.compile();
  #endif
  #endif

  #if defined(loco_vulkan) && defined(loco_window)
  fan::vulkan::pipeline_t::properties_t pipeline_p;

  render_fullscreen_shader.open(context, &m_write_queue);
  render_fullscreen_shader.set_vertex(
    context,
    "graphics/glsl/vulkan/2D/objects/loco_fbo.vert",
    #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.vert))
    );
    render_fullscreen_shader.set_fragment(
      context,
      "graphics/glsl/vulkan/2D/objects/loco_fbo.frag",
      #include _FAN_PATH(graphics/glsl/vulkan/2D/objects/loco_fbo.frag))
      );
      // NOTE order of the layouts (descriptor binds) depends about draw order of shape specific
      auto layouts = std::to_array({
        #if defined(loco_line)
        gloco->shapes.line.m_ssbo.m_descriptor.m_layout,
        #endif
        #if defined(loco_rectangle)
        gloco->shapes.rectangle.m_ssbo.m_descriptor.m_layout,
        #endif
        #if defined(loco_sprite)
        gloco->shapes.sprite.m_ssbo.m_descriptor.m_layout,
        #endif
        #if defined(loco_letter)
        gloco->shapes.letter.m_ssbo.m_descriptor.m_layout,
        #endif
        #if defined(loco_button)
        gloco->shapes.button.m_ssbo.m_descriptor.m_layout,
        #endif
        #if defined(loco_text_box)
        gloco->shapes.text_box.m_ssbo.m_descriptor.m_layout,
        #endif
        #if defined(loco_yuv420p)
        gloco->shapes.yuv420p.m_ssbo.m_descriptor.m_layout,
        #endif
        });
      // NOTE THIS
      std::reverse(layouts.begin(), layouts.end());
      pipeline_p.descriptor_layout_count = layouts.size();
      pipeline_p.descriptor_layout = layouts.data();
      pipeline_p.shader = &render_fullscreen_shader;
      pipeline_p.push_constants_size = sizeof(loco_t::push_constants_t);
      pipeline_p.subpass = 1;
      VkDescriptorImageInfo imageInfo{};

      VkPipelineColorBlendAttachmentState color_blend_attachment[1]{};
      color_blend_attachment[0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT
        ;
      color_blend_attachment[0].blendEnable = VK_TRUE;
      color_blend_attachment[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachment[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachment[0].colorBlendOp = VK_BLEND_OP_ADD;
      color_blend_attachment[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      color_blend_attachment[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      color_blend_attachment[0].alphaBlendOp = VK_BLEND_OP_ADD;
      pipeline_p.color_blend_attachment_count = std::size(color_blend_attachment);
      pipeline_p.color_blend_attachment = color_blend_attachment;
      pipeline_p.enable_depth_test = false;
      context.render_fullscreen_pl.open(context, pipeline_p);
      #endif

      #if defined(loco_opengl)
      default_texture.create_missing_texture();
      transparent_texture.create_transparent_texture();
      #endif

      fan::vec2 window_size = window.get_size();

      default_camera = add_camera(fan::graphics::direction_e::right);

      {

        default_camera_3d = new camera_impl_t;

        fan::vec2 window_size = gloco->window.get_size();
        default_camera_3d->viewport = default_camera->viewport;
        static constexpr f32_t fov = 90.f;
        gloco->open_camera(&default_camera_3d->camera, fov);
      }
      open_camera(&default_camera->camera,
        fan::vec2(0, window_size.x),
        fan::vec2(0, window_size.y)
      );


      #if defined(loco_physics)
      fan::graphics::open_bcol();
      #endif

      #if defined(loco_imgui)

      //wglMakeCurrent(g_MainWindow.hDC, g_hRC);

      ImGui::CreateContext();
      ImGuiIO& io = ImGui::GetIO(); (void)io;
      io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
      io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
      io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
      io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

      ImGuiStyle& style = ImGui::GetStyle();
      if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
      {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
      }

      bool windowed = true;
      // free this xd
      gloco->window.add_keys_callback(
        [windowed](const fan::window_t::keyboard_keys_cb_data_t& data) mutable {
          if (data.key == fan::key_enter && data.state == fan::keyboard_state::press && gloco->window.key_pressed(fan::key_left_alt)) {
            windowed = !windowed;
            gloco->window.set_size_mode(windowed ? fan::window_t::mode::windowed : fan::window_t::mode::borderless);
          }
        }
      );

      static bool init = false;
      if (init == false) {
        init = true;

        #if defined(loco_imgui)
        loco_t::imgui_themes::dark();

        //#if defined(fan_platform_windows)
        //ImGui_ImplWin32_InitForOpenGL(window.get_handle());
        ////ImGui_ImplWin32_Init(window.get_handle());


        //#elif defined(fan_platform_linux)
        //imgui_xorg_init();
        //#endif
        ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)window.glfw_window, true);
        const char* glsl_version = "#version 150";
        ImGui_ImplOpenGL3_Init(glsl_version);

        auto& style = ImGui::GetStyle();
        auto& io = ImGui::GetIO();

        static constexpr const char* font_name = "fonts/SourceCodePro-Regular.ttf";
        static constexpr f32_t font_size = 4;


        for (int i = 0; i < std::size(fonts); ++i) {
          fonts[i] = io.Fonts->AddFontFromFileTTF(font_name, (int)(font_size * (1 << i)) * 2);
          if (fonts[i] == nullptr) {
            fan::throw_error(fan::string("failed to load font") + font_name);
          }
        }
        io.Fonts->Build();
        io.FontDefault = fonts[2];
      }
      #endif
      #endif

      // register console commands

      #if defined(loco_imgui)

      console.open();

      console.commands.add("echo", [&](const fan::commands_t::arg_t& args) {
        fan::commands_t::output_t out;
        out.text = fan::append_args(args) + "\n";
        out.highlight = fan::commands_t::highlight_e::info;
        console.commands.output_cb(out);
        }).description = "prints something - usage echo [args]";

      console.commands.add("help", [&](const fan::commands_t::arg_t& args) {
        if (args.empty()) {
          fan::commands_t::output_t out;
          out.highlight = fan::commands_t::highlight_e::info;
          std::string out_str;
          out_str += "{\n";
          for (const auto& i : console.commands.func_table) {
            out_str += "\t" + i.first + ",\n";
          }
          out_str += "}\n";
          out.text = out_str;
          console.commands.output_cb(out);
          return;
        }
        else if (args.size() == 1) {
          auto found = console.commands.func_table.find(args[0]);
          if (found == console.commands.func_table.end()) {
            console.commands.print_command_not_found(args[0]);
            return;
          }
          fan::commands_t::output_t out;
          out.text = found->second.description + "\n";
          out.highlight = fan::commands_t::highlight_e::info;
          console.commands.output_cb(out);
        }
        else {
          console.commands.print_invalid_arg_count();
        }
        }).description = "get info about specific command - usage help command";

      console.commands.add("list", [&](const fan::commands_t::arg_t& args) {
        std::string out_str;
        for (const auto& i : console.commands.func_table) {
          out_str += i.first + "\n";
        }

        fan::commands_t::output_t out;
        out.text = out_str;
        out.highlight = fan::commands_t::highlight_e::info;

        console.commands.output_cb(out);
        }).description = "lists all commands - usage list";

      console.commands.add("alias", [&](const fan::commands_t::arg_t& args) {
        if (args.size() < 2 || args[1].empty()) {
          console.commands.print_invalid_arg_count();
          return;
        }
        if (console.commands.insert_to_command_chain(args)) {
          return;
        }
        console.commands.func_table[args[0]] = console.commands.func_table[args[1]];
        }).description = "can create alias commands - usage alias [cmd name] [cmd]";


      console.commands.add("show_fps", [&](const fan::commands_t::arg_t& args) {
        if (args.size() != 1) {
          console.commands.print_invalid_arg_count();
          return;
        }
        toggle_fps = std::stoi(args[0]);
        }).description = "toggles fps - usage show_fps [value]";

      console.commands.add("quit", [&](const fan::commands_t::arg_t& args) {
        exit(0);
        }).description = "quits program - usage quit";

      #endif
}


#if defined(loco_vfi)
void loco_t::push_back_input_hitbox(loco_t::shapes_t::vfi_t::shape_id_t& id, const loco_t::shapes_t::vfi_t::properties_t& p) {
  shapes.vfi.push_back(id, p);
}

inline void loco_t::feed_mouse_move(const fan::vec2& mouse_position) {
  shapes.vfi.feed_mouse_move(mouse_position);
}

inline void loco_t::feed_mouse_button(uint16_t button, fan::mouse_state button_state, const fan::vec2& mouse_position) {
  shapes.vfi.feed_mouse_button(button, button_state);
}

inline void loco_t::feed_keyboard(int key, fan::keyboard_state keyboard_state) {
  shapes.vfi.feed_keyboard(key, keyboard_state);
}

inline void loco_t::feed_text(uint32_t key) {
  shapes.vfi.feed_text(key);
}
#endif

inline void loco_t::process_frame() {
  #if defined(loco_opengl)
  #if defined(loco_framebuffer)
  m_framebuffer.bind(get_context());

  auto& opengl = get_context().opengl;

  for (int i = 0; i < std::size(color_buffers); ++i) {
    opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + i);
    color_buffers[i].bind_texture();
    opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT0 + (std::size(color_buffers) - 1 - i));
    if (i + 1 == std::size(color_buffers)) {
      opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
    }
    else {
      opengl.glClearColor(0, 0, 0, 1);
    }
    opengl.call(opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
  }


  #endif
  #endif

  auto it = m_update_callback.GetNodeFirst();
  while (it != m_update_callback.dst) {
    m_update_callback.StartSafeNext(it);
    m_update_callback[it](this);
    it = m_update_callback.EndSafeNext();
  }

  m_write_queue.process(get_context());

  #ifdef loco_window
  #if defined(loco_opengl)

  #include "draw_shapes.h"

  #if defined(loco_framebuffer)


  m_framebuffer.unbind(get_context());

  blur[0].draw(&color_buffers[0]);
  //blur[1].draw(&color_buffers[3]);

  opengl.glClearColor(0, 0, 0, 1);
  opengl.call(opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
  fan::vec2 window_size = window.get_size();
  fan::opengl::viewport_t::set_viewport(0, window_size, window_size);

  m_fbo_final_shader.use();
  m_fbo_final_shader.set_int("_t00", 0);
  m_fbo_final_shader.set_int("_t01", 1);

  opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
  color_buffers[0].bind_texture();

  get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
  blur[0].mips.front().image.bind_texture();

  //get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE2);
  //blur[1].mips.front().image.bind_texture();

  render_final_fb();

  #endif
  #if defined(loco_imgui)

  {
    auto it = m_imgui_draw_cb.GetNodeFirst();
    while (it != m_imgui_draw_cb.dst) {
      m_imgui_draw_cb.StartSafeNext(it);
      m_imgui_draw_cb[it]();
      it = m_imgui_draw_cb.EndSafeNext();
    }
  }

  opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
  color_buffers[0].bind_texture();

  //get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
  //blur.mips.front().image.bind_texture();

  //m_framebuffer.bind(get_context());
  //opengl.glBindFramebuffer(fan::opengl::GL_READ_FRAMEBUFFER, 0); // Bind default framebuffer as source
  //opengl.glBindFramebuffer(fan::opengl::GL_DRAW_FRAMEBUFFER, m_framebuffer.framebuffer); // Bind FBO as destination
  //opengl.glBlitFramebuffer(0, 0, window_size.x, window_size.y, 0, 0, window_size.x, window_size.y, fan::opengl::GL_COLOR_BUFFER_BIT, fan::opengl::GL_NEAREST);

  static constexpr uint32_t parent_window_flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiDockNodeFlags_NoDockingSplit | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoInputs;

  if (toggle_fps) {
    ImGui::SetNextWindowSize(fan::vec2(ImGui::GetIO().DisplaySize) / 3);
    ImGui::SetNextWindowPos(ImVec2(0, 0));

    ImGui::Begin("Global window", 0, parent_window_flags);
    ImGui::Text("fps:%d", (int)(1.f / delta_time));

    static constexpr int buffer_size = 100;
    static std::array<float, buffer_size> samples;
    static int insert_index = 0;

    float average_frame_time_ms = std::accumulate(samples.begin(), samples.end(), 0.0f) / buffer_size;
    float lowest_ms = *std::min_element(samples.begin(), samples.end());
    float highest_ms = *std::max_element(samples.begin(), samples.end());
    ImGui::Text("Average Frame Time: %.2f ms", average_frame_time_ms);
    ImGui::Text("Lowest Frame Time: %.2f ms", lowest_ms);
    ImGui::Text("Highest Frame Time: %.2f ms", highest_ms);
    static uint32_t frame = 0;
    frame++;
    static constexpr int refresh_speed = 25;
    if (frame % refresh_speed == 0) {
      samples[insert_index] = delta_time * 1000;
      insert_index = (insert_index + 1) % buffer_size;
    }

    ImGui::PlotLines("frame time (ms)", samples.data(), buffer_size, insert_index, nullptr, 0.0f, FLT_MAX, ImVec2(0, 80));
    ImGui::End();
  }

  if (ImGui::IsKeyPressed(ImGuiKey_F3, false)) {
    toggle_console = !toggle_console;
    // force focus xd
    console.input.InsertText("a");
    console.input.SetText("");
  }

  if (toggle_console) {
    console.render();
  }

  ImGui::Render();
  //get_context().opengl.glViewport(0, 0, window.get_size().x, window.get_size().y);
  //get_context().opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
  //get_context().opengl.glClear(fan::opengl::GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


  ImGuiIO& io = ImGui::GetIO(); (void)io;
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    get_context().set_current(get_window());
  }

  //#if defined(loco_framebuffer)

  //#endif
  #if defined(loco_framebuffer)

  m_framebuffer.unbind(get_context());

  #endif

  #endif

  get_context().render(get_window());

  #elif defined(loco_vulkan)
  get_context().begin_render(get_window());

  #include "draw_shapes.h"

  get_context().end_render(get_window());
  #endif
  #endif
}
uint32_t loco_t::get_fps() {
  return window.get_fps();
}
void loco_t::set_vsync(bool flag) {
  get_context().set_vsync(get_window(), flag);
}
fan::vec2 loco_t::transform_matrix(const fan::vec2& position) {
  fan::vec2 window_size = window.get_size();
  // not custom ortho friendly - made for -1 1
  return position / window_size * 2 - 1;
}
fan::vec2 loco_t::screen_to_ndc(const fan::vec2& screen_pos) {
  fan::vec2 window_size = window.get_size();
  return screen_pos / window_size * 2 - 1;
}
fan::vec2 loco_t::ndc_to_screen(const fan::vec2& ndc_position) {
  fan::vec2 window_size = window.get_size();
  fan::vec2 normalized_position = (ndc_position + 1) / 2;
  return normalized_position * window_size;
}

//  behaving oddly

inline fan::vec2d loco_t::get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
  fan::vec2d mouse_pos = window.get_mouse_position();
  fan::vec2d translated_pos;
  translated_pos.x = fan::math::map(mouse_pos.x, viewport.get_position().x, viewport.get_position().x + viewport.get_size().x, camera.coordinates.left, camera.coordinates.right);
  translated_pos.y = fan::math::map(mouse_pos.y, viewport.get_position().y, viewport.get_position().y + viewport.get_size().y, camera.coordinates.up, camera.coordinates.down);
  return translated_pos;
}
inline fan::vec2d loco_t::get_mouse_position() {
  return window.get_mouse_position();
  //return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
}
fan::vec2 loco_t::translate_position(const fan::vec2& p, fan::graphics::viewport_t* viewport, loco_t::camera_t* camera) {

  fan::vec2 viewport_position = viewport->get_position();
  fan::vec2 viewport_size = viewport->get_size();

  f32_t l = camera->coordinates.left;
  f32_t r = camera->coordinates.right;
  f32_t t = camera->coordinates.up;
  f32_t b = camera->coordinates.down;

  fan::vec2 tp = p - viewport_position;
  fan::vec2 d = viewport_size;
  tp /= d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  return tp;
}
fan::vec2 loco_t::translate_position(const fan::vec2& p) {
  return translate_position(p, &default_camera->viewport, &default_camera->camera);
}
bool loco_t::process_loop(const fan::function_t<void()>& lambda) {

  // enables drawing while resizing, not required for x11
  #if defined(fan_platform_windows)
  auto it = window.add_resize_callback([this, &lambda](const auto& d) {
    gloco->process_loop(lambda);
    });
  #endif

  //get_context().set_current(get_window());
  window.handle_events();
  if (glfwWindowShouldClose(window.glfw_window)) {
    window.close();
    return 1;
  }

  #if defined(fan_platform_windows)
  window.remove_resize_callback(it);
  #endif


  #if defined(loco_imgui)
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  auto& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  const ImVec4 bgColor = ImVec4(0.0, 0.0, 0.0, 0.4);
  colors[ImGuiCol_WindowBg] = bgColor;
  colors[ImGuiCol_ChildBg] = bgColor;
  colors[ImGuiCol_TitleBg] = bgColor;

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
  ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
  ImGui::PopStyleColor(2);
  #endif

  lambda();

  ev_timer.process();
  process_frame();
  return 0;
}
void loco_t::loop(const fan::function_t<void()>& lambda) {
  while (1) {
    if (process_loop(lambda)) {
      break;
    }
  }
}

#if defined(loco_vfi)

loco_t::vfi_id_t::vfi_id_t(const properties_t& p) {
  gloco->shapes.vfi.push_back(cid, *(properties_t*)&p);
}

loco_t::vfi_id_t& loco_t::vfi_id_t::operator[](const properties_t& p) {
  gloco->shapes.vfi.push_back(cid, *(properties_t*)&p);
  return *this;
}

loco_t::vfi_id_t::~vfi_id_t() {
  gloco->shapes.vfi.erase(cid);
}


#endif

void loco_t::shape_draw(loco_t::shape_type_t shape_type, const loco_t::redraw_key_t& redraw_key, loco_bdbt_NodeReference_t nr) {
  shapes.iterate([&]<auto i>(auto & shape) {
    if (shape_type != shape.shape_type) {
      return;
    }

    fan_if_has_function(&shape, draw, (redraw_key, nr));
  });
}

void loco_t::shape_erase(loco_t::cid_nt_t& id) {
  bool erased = false;
  shapes.iterate([&]<auto i>(auto & shape) {
    if (erased) {
      return;
    }
    if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) {
      return;
    }
    fan_if_has_function(&shape, erase, (id));
    erased = true;
  });
}

void loco_t::shape_set_line(loco_t::cid_nt_t& id, const fan::vec3& src, fan::vec2 dst) {
  shapes.iterate([&]<auto i, typename T>(T & shape) {
    if (shape.shape_type != (loco_t::shape_type_t)id->shape_type) {
      return;
    }
    fan_if_has_function(&shape, set_line, (id, src, dst));
  });
}

inline loco_t::camera_impl_t* loco_t::add_camera(fan::graphics::direction_e split_direction) {
  viewport_handler.push_back(new camera_impl_t(split_direction));
  int index = 0;
  fan::vec2 window_size = gloco->window.get_size();
  gloco->viewport_divider.iterate([&index, window_size, this](auto& node) {
    viewport_handler[index]->viewport.set(
      (node.position - node.size / 2) * window_size,
      ((node.size) * window_size), window_size
    );
    index++;
    });
  return viewport_handler.back();
}

#if defined(loco_sprite)
fan::string loco_t::get_sprite_vertex_shader() {
    return 
      #if defined(loco_opengl)
      loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/sprite.vs))
      #else
      "";
    ;
      #endif
      ;
  }

loco_t::shader_t loco_t::create_sprite_shader(const fan::string& fragment) {
  loco_t::shader_t shader;
  #if defined(loco_opengl)
  shader.open();
  shader.set_vertex(
    get_sprite_vertex_shader()
  );
  shader.set_fragment(fragment);
  shader.compile();
  #else
  assert(0);
  #endif
  return shader;
}

#endif

#if defined(loco_light)
loco_t::shader_t loco_t::create_light_shader(const fan::string& fragment) {
  loco_t::shader_t shader;
  shader.open();
  shader.set_vertex(
    loco_t::read_shader(_FAN_PATH_QUOTE(graphics/glsl/opengl/2D/objects/light.vs))
  );
  shader.set_fragment(fragment);
  shader.compile();
  return shader;
}
#endif

fan::vec2 loco_t::convert_mouse_to_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size) {
  return fan::vec2((2.0f * mouse_position.x) / window_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / window_size.y);
}
fan::vec2 loco_t::convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
  return convert_mouse_to_ndc(mouse_position, gloco->window.get_size());
}
fan::vec2 loco_t::convert_mouse_to_ndc() const {
  return convert_mouse_to_ndc(gloco->get_mouse_position(), gloco->window.get_size());
}
fan::ray3_t loco_t::convert_mouse_to_ray(const fan::vec2i& mouse_position, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
  fan::vec2i screen_size = gloco->window.get_size();

  fan::vec4 ray_ndc((2.0f * mouse_position.x) / screen_size.x - 1.0f, 1.0f - (2.0f * mouse_position.y) / screen_size.y, 1.0f, 1.0f);

  fan::mat4 inverted_projection = projection.inverse();

  fan::vec4 ray_clip = inverted_projection * ray_ndc;

  ray_clip.z = -1.0f;
  ray_clip.w = 0.0f;

  fan::mat4 inverted_view = view.inverse();

  fan::vec4 ray_world = inverted_view * ray_clip;

  fan::vec3 ray_dir = fan::vec3(ray_world.x, ray_world.y, ray_world.z).normalize();

  fan::vec3 ray_origin = camera_position;
  return fan::ray3_t(ray_origin, ray_dir);
}
fan::ray3_t loco_t::convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
  return convert_mouse_to_ray(gloco->get_mouse_position(), camera_position, projection, view);
}
fan::ray3_t loco_t::convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
  return convert_mouse_to_ray(gloco->get_mouse_position(), default_camera_3d->camera.position, projection, view);
}
bool loco_t::is_ray_intersecting_cube(const fan::ray3_t& ray, const fan::vec3& position, const fan::vec3& size) {
  fan::vec3 min_bounds = position - size;
  fan::vec3 max_bounds = position + size;

  fan::vec3 t_min = (min_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));
  fan::vec3 t_max = (max_bounds - ray.origin) / (ray.direction + fan::vec3(1e-6f));

  fan::vec3 t1 = t_min.min(t_max);
  fan::vec3 t2 = t_min.max(t_max);

  float t_near = fan::max(t1.x, fan::max(t1.y, t1.z));
  float t_far = fan::min(t2.x, fan::min(t2.y, t2.z));

  return t_near <= t_far && t_far >= 0.0f;
}

#if defined(loco_sprite)
void loco_t::add_fragment_shader_reload(int key, const fan::string& vs_path, const fan::string& fs_path) {
  gloco->window.add_key_callback(key, fan::keyboard_state::press, [&, vs_path, fs_path](const auto&) {
    fan::string str;
    fan::io::file::read(fs_path, &str);
    gloco->shapes.sprite.m_shader.set_vertex(loco_t::read_shader(vs_path));
    gloco->shapes.sprite.m_shader.set_fragment(str.c_str());
    gloco->shapes.sprite.m_shader.compile();
    });
}
#endif

#if defined(loco_imgui)
std::string loco_t::extract_variable_type(const std::string& string_data, const std::string& varName) {
  std::istringstream file(string_data);

  std::string type;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string word;
    while (iss >> word) {
      if (word.find(varName) != std::string::npos) {
        return type;
      }
      else {
        type = word;
      }
    }
  }

  return "";
}

void loco_t::set_imgui_viewport(loco_t::viewport_t& viewport)
{
  ImVec2 mainViewportPos = ImGui::GetMainViewport()->Pos;

  ImVec2 windowPos = ImGui::GetWindowPos();

  fan::vec2 windowPosRelativeToMainViewport;
  windowPosRelativeToMainViewport.x = windowPos.x - mainViewportPos.x;
  windowPosRelativeToMainViewport.y = windowPos.y - mainViewportPos.y;

  fan::vec2 window_size = window.get_size();
  fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
  fan::vec2 viewport_pos = fan::vec2(windowPosRelativeToMainViewport + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
  viewport.set(viewport_pos, viewport_size, window_size);
}
#endif

#if defined(loco_vulkan)
inline void fan::vulkan::context_t::begin_render(fan::window_t* window) {
  vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

  VkResult result = vkAcquireNextImageKHR(
    device,
    swapChain,
    UINT64_MAX,
    imageAvailableSemaphores[currentFrame],
    VK_NULL_HANDLE,
    &image_index
  );

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreateSwapChain(window->get_size());
    return;
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    fan::throw_error("failed to acquire swap chain image!");
  }

  vkResetFences(device, 1, &inFlightFences[currentFrame]);

  vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) {
    fan::throw_error("failed to begin recording command buffer!");
  }

  command_buffer_in_use = true;

  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = renderPass;
  renderPassInfo.framebuffer = swapChainFramebuffers[image_index];
  renderPassInfo.renderArea.offset = { 0, 0 };
  renderPassInfo.renderArea.extent.width = swap_chain_size.x;
  renderPassInfo.renderArea.extent.height = swap_chain_size.y;

  // TODO

  #if defined(loco_wboit)
  VkClearValue clearValues[4]{};
  clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f} };
  clearValues[3].depthStencil = { 1.0f, 0 };

  clearValues[0].color.float32[0] = 0.0f;
  clearValues[0].color.float32[1] = 0.0f;
  clearValues[0].color.float32[2] = 0.0f;
  clearValues[0].color.float32[3] = 0.0f;
  clearValues[1].color.float32[0] = 1.f;  // Initially, all pixels show through all the way (reveal = 100%)

  #else
  VkClearValue clearValues[
    5
  ]{};
    clearValues[0].color = { { gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[1].color = { {gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[2].color = { {gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[3].color = { {gloco->clear_color.r, gloco->clear_color.g, gloco->clear_color.b, gloco->clear_color.a} };
    clearValues[4].depthStencil = { 1.0f, 0 };
    #endif

    renderPassInfo.clearValueCount = std::size(clearValues);
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

inline fan::vulkan::viewport_list_NodeReference_t::viewport_list_NodeReference_t(fan::vulkan::viewport_t* viewport) {
  NRI = viewport->viewport_reference.NRI;
}

inline void fan::vulkan::pipeline_t::open(fan::vulkan::context_t& context, const properties_t& p) {
  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = p.shape_type;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_FALSE;//p.enable_depth_test;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = p.depth_test_compare_op;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_NO_OP;
  colorBlending.attachmentCount = p.color_blend_attachment_count;
  colorBlending.pAttachments = p.color_blend_attachment;
  colorBlending.blendConstants[0] = 1.0f;
  colorBlending.blendConstants[1] = 1.0f;
  colorBlending.blendConstants[2] = 1.0f;
  colorBlending.blendConstants[3] = 1.0f;

  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
  };
  VkPipelineDynamicStateCreateInfo dynamicState{};
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = dynamicStates.size();
  dynamicState.pDynamicStates = dynamicStates.data();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = p.descriptor_layout_count;
  pipelineLayoutInfo.pSetLayouts = p.descriptor_layout;

  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = p.push_constants_size;
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  pipelineLayoutInfo.pPushConstantRanges = &push_constant;
  pipelineLayoutInfo.pushConstantRangeCount = 1;

  if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &m_layout) != VK_SUCCESS) {
    fan::throw_error("failed to create pipeline layout!");
  }

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = ((loco_t::shader_t*)p.shader)->get_shader().shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.layout = m_layout;
  pipelineInfo.renderPass = context.renderPass;
  pipelineInfo.subpass = p.subpass;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
    fan::throw_error("failed to create graphics pipeline");
  }
}
#endif

#if defined(loco_pixel_format_renderer)
uint8_t fan::pixel_format::get_texture_amount(uint8_t format) {
  switch (format) {
    case undefined: {
      return 0;
    }
    case yuv420p: {
      return 3;
    }
    case nv12: {
      return 2;
    }
    default: {
      fan::throw_error("invalid format");
      return undefined;
    }
  }
}

std::array<fan::vec2ui, 4> fan::pixel_format::get_image_sizes(uint8_t format, const fan::vec2ui& image_size) {
  switch (format) {
    case yuv420p: {
      return std::array<fan::vec2ui, 4>{image_size, image_size / 2, image_size / 2};
    }
    case nv12: {
      return std::array<fan::vec2ui, 4>{image_size, fan::vec2ui{ image_size.x, image_size.y }};
    }
    default: {
      fan::throw_error("invalid format");
      return std::array<fan::vec2ui, 4>{};
    }
  }
}
#endif

#if defined(loco_imgui)
IMGUI_API void ImGui::Image(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col) {
  ImGui::Image((void*)img.get_texture(), size, uv0, uv1, tint_col, border_col);
}

IMGUI_API bool ImGui::ImageButton(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {
  return ImGui::ImageButton((void*)img.get_texture(), size, uv0, uv1, frame_padding, bg_col, tint_col);
}
#endif

#if defined(loco_cuda)
loco_t::cuda_textures_t::cuda_textures_t() {
  inited = false;
}

loco_t::cuda_textures_t::~cuda_textures_t() {
}

void loco_t::cuda_textures_t::close(loco_t* loco, loco_t::shape_t& cid) {
  uint8_t image_amount = fan::pixel_format::get_texture_amount(loco->shapes.pixel_format_renderer.sb_get_ri(cid).format);
  auto& ri = loco->shapes.pixel_format_renderer.sb_get_ri(cid);
  for (uint32_t i = 0; i < image_amount; ++i) {
    wresources[i].close();
    ri.images[i].unload();
  }
}

void loco_t::cuda_textures_t::resize(loco_t* loco, loco_t::cid_nt_t& id, uint8_t format, fan::vec2ui size, uint32_t filter) {
  auto& ri = loco->shapes.pixel_format_renderer.sb_get_ri(id);
  uint8_t image_amount = fan::pixel_format::get_texture_amount(format);
  if (inited == false) {
    // purge cid's images here
    // update cids images
    loco->shapes.pixel_format_renderer.reload(id, format, size, filter);
    for (uint32_t i = 0; i < image_amount; ++i) {
      wresources[i].open(loco->image_list[ri.images[i].texture_reference].texture_id);
    }
    inited = true;
  }
  else {

    if (ri.images[0].size == size) {
      return;
    }

    // update cids images
    for (uint32_t i = 0; i < fan::pixel_format::get_texture_amount(ri.format); ++i) {
      wresources[i].close();
    }

    loco->shapes.pixel_format_renderer.reload(id, format, size, filter);

    for (uint32_t i = 0; i < image_amount; ++i) {
      wresources[i].open(loco->image_list[ri.images[i].texture_reference].texture_id);
    }
  }
}

cudaArray_t& loco_t::cuda_textures_t::get_array(uint32_t index) {
  return wresources[index].cuda_array;
}

void loco_t::cuda_textures_t::graphics_resource_t::open(int texture_id) {
  fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, fan::opengl::GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
  map();
}

void loco_t::cuda_textures_t::graphics_resource_t::close() {
  unmap();
  fan::cuda::check_error(cudaGraphicsUnregisterResource(resource));
  resource = nullptr;
}

void loco_t::cuda_textures_t::graphics_resource_t::map() {
  fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
  fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(&cuda_array, resource, 0, 0));
  fan::print("+", resource);
}

void loco_t::cuda_textures_t::graphics_resource_t::unmap() {
  fan::print("-", resource);
  fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
  //fan::cuda::check_error(cudaGraphicsResourceSetMapFlags(resource, 0));
}
#endif

#if defined(loco_imgui)

loco_t::imgui_element_nr_t::imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
  if (nr.is_invalid()) {
    return;
  }
  init();
}

loco_t::imgui_element_nr_t::imgui_element_nr_t(imgui_element_nr_t&& nr) {
  NRI = nr.NRI;
  nr.invalidate_soft();
}

loco_t::imgui_element_nr_t::~imgui_element_nr_t() {
  invalidate();
}

loco_t::imgui_element_nr_t& loco_t::imgui_element_nr_t::operator=(const imgui_element_nr_t& id) {
  if (!is_invalid()) {
    invalidate();
  }
  if (id.is_invalid()) {
    return *this;
  }

  if (this != &id) {
    init();
  }
  return *this;
}

loco_t::imgui_element_nr_t& loco_t::imgui_element_nr_t::operator=(imgui_element_nr_t&& id) {
  if (!is_invalid()) {
    invalidate();
  }
  if (id.is_invalid()) {
    return *this;
  }

  if (this != &id) {
    if (!is_invalid()) {
      invalidate();
    }
    NRI = id.NRI;

    id.invalidate_soft();
  }
  return *this;
}

void loco_t::imgui_element_nr_t::init() {
  *(base_t*)this = gloco->m_imgui_draw_cb.NewNodeLast();
}

bool loco_t::imgui_element_nr_t::is_invalid() const {
  return loco_t::imgui_draw_cb_inric(*this);
}

void loco_t::imgui_element_nr_t::invalidate_soft() {
  *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
}

void loco_t::imgui_element_nr_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  gloco->m_imgui_draw_cb.unlrec(*this);
  *(base_t*)this = gloco->m_imgui_draw_cb.gnric();
}

void loco_t::imgui_element_nr_t::set(const auto& lambda) {
  gloco->m_imgui_draw_cb[*this] = lambda;
}

#endif

loco_t::shape_t::shape_t(const shape_t& id) :
  inherit_t(id)
{
  if (id.is_invalid()) {
    return;
  }
  gloco->shape_get_properties(*(shape_t*)&id, [&]<typename T>(const T & properties) {
    gloco->push_shape(*this, properties);
  });
}

loco_t::shape_t::shape_t(shape_t&& id) : inherit_t(std::move(id)) {}

loco_t::shape_t& loco_t::shape_t::operator=(const shape_t& id) {
  if (!is_invalid()) {
    erase();
  }
  if (id.is_invalid()) {
    return *this;
  }
  if (this != &id) {
    gloco->shape_get_properties(*(shape_t*)&id, [&]<typename T>(const T & properties) {
      init();
      gloco->push_shape(*this, properties);
    });
  }
  return *this;
}

loco_t::shape_t& loco_t::shape_t::operator=(shape_t&& id) {
  if (!is_invalid()) {
    erase();
  }
  if (id.is_invalid()) {
    return *this;
  }
  if (this != &id) {
    if (!is_invalid()) {
      erase();
    }
    *(inherit_t*)this = std::move(id);
    id.invalidate();
  }
  return *this;
}

loco_t::shape_t::~shape_t() {
  erase();
}

void loco_t::shape_t::erase() {
  if (is_invalid()) {
    return;
  }
  gloco->shape_erase(*this);
  inherit_t::invalidate();
}

loco_t::cid_nr_t::cid_nr_t() { *(cid_list_NodeReference_t*)this = cid_list_gnric(); }

loco_t::cid_nr_t::cid_nr_t(const cid_nt_t& c) : cid_nt_t(c) {

}

loco_t::cid_nr_t::cid_nr_t(const cid_nr_t& nr) : cid_nr_t() {
  if (nr.is_invalid()) {
    return;
  }
  init();
  gloco->cid_list[*this].cid.shape_type = gloco->cid_list[nr].cid.shape_type;
}

loco_t::cid_nr_t::cid_nr_t(cid_nr_t&& nr) {
  NRI = nr.NRI;
  nr.invalidate_soft();
}

loco_t::cid_nr_t& loco_t::cid_nr_t::operator=(const cid_nr_t& id) {
  if (!is_invalid()) {
    invalidate();
  }
  if (id.is_invalid()) {
    return *this;
  }

  if (this != &id) {
    init();
    gloco->cid_list[*this].cid.shape_type = gloco->cid_list[id].cid.shape_type;
  }
  return *this;
}

loco_t::cid_nr_t& loco_t::cid_nr_t::operator=(cid_nr_t&& id) {
  if (!is_invalid()) {
    invalidate();
  }
  if (id.is_invalid()) {
    return *this;
  }

  if (this != &id) {
    if (!is_invalid()) {
      invalidate();
    }
    NRI = id.NRI;

    id.invalidate_soft();
  }
  return *this;
}

loco_t::cid_t* loco_t::cid_nt_t::operator->() const {
  return &gloco->cid_list[*(cid_list_NodeReference_t*)(this)].cid;
}

void loco_t::cid_nt_t::init() {
  *(base_t*)this = gloco->cid_list.NewNodeLast();
}

bool loco_t::cid_nt_t::is_invalid() const {
  return cid_list_inric(*this);
}

void loco_t::cid_nt_t::invalidate_soft() {
  *(base_t*)this = gloco->cid_list.gnric();
}

void loco_t::cid_nt_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  gloco->cid_list.unlrec(*this);
  *(base_t*)this = gloco->cid_list.gnric();
}

uint32_t* loco_t::cid_nt_t::gdp4() {
  return (uint32_t*)&(*this)->bm_id;
}

loco_t::camera_t::resize_callback_id_t::resize_callback_id_t() : loco_t::viewport_resize_callback_NodeReference_t() {}

loco_t::camera_t::resize_callback_id_t::resize_callback_id_t(const inherit_t& i) : inherit_t(i) {}

loco_t::camera_t::resize_callback_id_t::resize_callback_id_t(resize_callback_id_t&& i) : inherit_t(i) {
  i.sic();
}

loco_t::camera_t::resize_callback_id_t& loco_t::camera_t::resize_callback_id_t::operator=(resize_callback_id_t&& i) {
  if (this != &i) {
    *(inherit_t*)this = *(inherit_t*)&i;
    i.sic();
  }
  return *this;
}

loco_t::camera_t::resize_callback_id_t::operator loco_t::viewport_resize_callback_NodeReference_t() {
  return *(loco_t::viewport_resize_callback_NodeReference_t*)this;
}

loco_t::camera_t::resize_callback_id_t::~resize_callback_id_t() {
  if (iic()) {
    return;
  }
  gloco->m_viewport_resize_callback.unlrec(*this);
  sic();
}
