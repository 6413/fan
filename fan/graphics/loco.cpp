#include <pch.h>

#define loco_framebuffer
#define loco_post_process

//#define depth_debug
//
global_loco_t::operator loco_t* () {
  return loco;
}

global_loco_t& global_loco_t::operator=(loco_t* l) {
  loco = l;
  return *this;
}

//thread_local global_loco_t gloco;

void loco_t::use() {
  gloco = this;
}

void loco_t::camera_move(fan::opengl::context_t::camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction) {
  camera.velocity /= friction * dt + 1;
  static constexpr auto minimum_velocity = 0.001;
  static constexpr f32_t camera_rotate_speed = 100;
  if (camera.velocity.x < minimum_velocity && camera.velocity.x > -minimum_velocity) {
    camera.velocity.x = 0;
  }
  if (camera.velocity.y < minimum_velocity && camera.velocity.y > -minimum_velocity) {
    camera.velocity.y = 0;
  }
  if (camera.velocity.z < minimum_velocity && camera.velocity.z > -minimum_velocity) {
    camera.velocity.z = 0;
  }

  f64_t msd = (movement_speed * dt);
  if (gloco->window.key_pressed(fan::input::key_w)) {
    camera.velocity += camera.m_front * msd;
  }
  if (gloco->window.key_pressed(fan::input::key_s)) {
    camera.velocity -= camera.m_front * msd;
  }
  if (gloco->window.key_pressed(fan::input::key_a)) {
    camera.velocity -= camera.m_right * msd;
  }
  if (gloco->window.key_pressed(fan::input::key_d)) {
    camera.velocity += camera.m_right * msd;
  }

  if (gloco->window.key_pressed(fan::input::key_space)) {
    camera.velocity.y += movement_speed * gloco->delta_time;
  }
  if (gloco->window.key_pressed(fan::input::key_left_shift)) {
    camera.velocity.y -= movement_speed * gloco->delta_time;
  }

  f64_t rotate = camera.sensitivity * camera_rotate_speed * gloco->delta_time;
  if (gloco->window.key_pressed(fan::input::key_left)) {
    camera.set_yaw(camera.get_yaw() - rotate);
  }
  if (gloco->window.key_pressed(fan::input::key_right)) {
    camera.set_yaw(camera.get_yaw() + rotate);
  }
  if (gloco->window.key_pressed(fan::input::key_up)) {
    camera.set_pitch(camera.get_pitch() + rotate);
  }
  if (gloco->window.key_pressed(fan::input::key_down)) {
    camera.set_pitch(camera.get_pitch() - rotate);
  }

  camera.position += camera.velocity * gloco->delta_time;
  camera.update_view();

  camera.m_view = camera.get_view_matrix();
}

void loco_t::render_final_fb() {
  auto& context = get_context();
  context.opengl.glBindVertexArray(fb_vao);
  context.opengl.glDrawArrays(fan::opengl::GL_TRIANGLE_STRIP, 0, 4);
  context.opengl.glBindVertexArray(0);
}


void loco_t::initialize_fb_vaos(uint32_t& vao, uint32_t& vbo) {
  static constexpr f32_t quad_vertices[] = {
     -1.0f, 1.0f, 0, 0.0f, 1.0f,
     -1.0f, -1.0f, 0, 0.0f, 0.0f,
     1.0f, 1.0f, 0, 1.0f, 1.0f,
     1.0f, -1.0f, 0, 1.0f, 0.0f,
  };
  auto& context = get_context();
  context.opengl.glGenVertexArrays(1, &vao);
  context.opengl.glGenBuffers(1, &vbo);
  context.opengl.glBindVertexArray(vao);
  context.opengl.glBindBuffer(fan::opengl::GL_ARRAY_BUFFER, vbo);
  context.opengl.glBufferData(fan::opengl::GL_ARRAY_BUFFER, sizeof(quad_vertices), &quad_vertices, fan::opengl::GL_STATIC_DRAW);
  context.opengl.glEnableVertexAttribArray(0);
  context.opengl.glVertexAttribPointer(0, 3, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)0);
  context.opengl.glEnableVertexAttribArray(1);
  context.opengl.glVertexAttribPointer(1, 2, fan::opengl::GL_FLOAT, fan::opengl::GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
}

void generate_commands(loco_t* loco) {
#if defined(loco_imgui)
  loco->console.open();

  loco->console.commands.add("echo", [](const fan::commands_t::arg_t& args) {
    fan::commands_t::output_t out;
    out.text = fan::append_args(args) + "\n";
    out.highlight = fan::commands_t::highlight_e::info;
    gloco->console.commands.output_cb(out);
  }).description = "prints something - usage echo [args]";

  loco->console.commands.add("help", [](const fan::commands_t::arg_t& args) {
    if (args.empty()) {
      fan::commands_t::output_t out;
      out.highlight = fan::commands_t::highlight_e::info;
      std::string out_str;
      out_str += "{\n";
      for (const auto& i : gloco->console.commands.func_table) {
        out_str += "\t" + i.first + ",\n";
      }
      out_str += "}\n";
      out.text = out_str;
      gloco->console.commands.output_cb(out);
      return;
    }
    else if (args.size() == 1) {
      auto found = gloco->console.commands.func_table.find(args[0]);
      if (found == gloco->console.commands.func_table.end()) {
        gloco->console.commands.print_command_not_found(args[0]);
        return;
      }
      fan::commands_t::output_t out;
      out.text = found->second.description + "\n";
      out.highlight = fan::commands_t::highlight_e::info;
      gloco->console.commands.output_cb(out);
    }
    else {
      gloco->console.commands.print_invalid_arg_count();
    }
  }).description = "get info about specific command - usage help command";

  loco->console.commands.add("list", [](const fan::commands_t::arg_t& args) {
    std::string out_str;
    for (const auto& i : gloco->console.commands.func_table) {
      out_str += i.first + "\n";
    }

    fan::commands_t::output_t out;
    out.text = out_str;
    out.highlight = fan::commands_t::highlight_e::info;

    gloco->console.commands.output_cb(out);
  }).description = "lists all commands - usage list";

  loco->console.commands.add("alias", [](const fan::commands_t::arg_t& args) {
    if (args.size() < 2 || args[1].empty()) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    if (gloco->console.commands.insert_to_command_chain(args)) {
      return;
    }
    gloco->console.commands.func_table[args[0]] = gloco->console.commands.func_table[args[1]];
  }).description = "can create alias commands - usage alias [cmd name] [cmd]";


  loco->console.commands.add("show_fps", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->toggle_fps = std::stoi(args[0]);
  }).description = "toggles fps - usage show_fps [value]";

  loco->console.commands.add("quit", [](const fan::commands_t::arg_t& args) {
    exit(0);
  }).description = "quits program - usage quit";

  loco->console.commands.add("clear", [](const fan::commands_t::arg_t& args) {
    gloco->console.output_buffer.clear();
    gloco->console.editor.SetText("");
  }).description = "clears output buffer - usage clear";

  loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocess shader";

  loco->console.commands.add("set_gamma", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocess shader";

  loco->console.commands.add("set_exposure", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "exposure", std::stof(args[0]));
  }).description = "sets exposure for postprocess shader";

  loco->console.commands.add("bloom_strength", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->shader_set_value(gloco->m_fbo_final_shader, "bloom_strength", std::stof(args[0]));
  }).description = "sets bloom strength for postprocess shader";

  loco->console.commands.add("set_vsync", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->set_vsync(std::stoi(args[0]));
    }).description = "sets vsync";

  /*loco->console.commands.add("console_transparency", [](const fan::commands_t::arg_t& args) {
    if (args.size() != 1) {
      gloco->console.commands.print_invalid_arg_count();
      return;
    }
    gloco->console.transparency = std::stoull(args[0]);
    for (int i = 0; i < 21; ++i) {
      (gloco->console.editor.GetPalette().data() + i = gloco->console.transparency;
    }
    }).description = "";*/

#endif
}

void init_imgui(loco_t* loco) {
#if defined(loco_imgui)
  ImGui::CreateContext();
  ImPlot::CreateContext();
  auto& input_map = ImPlot::GetInputMap();
  input_map.Pan = ImGuiMouseButton_Middle;

  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ///    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
  {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }

  static bool init = false;
  if (init == false) {
    init = true;

    //loco_t::imgui_themes::dark();

    ImGui_ImplGlfw_InitForOpenGL(loco->window.glfw_window, true);
    const char* glsl_version = "#version 150";
    ImGui_ImplOpenGL3_Init(glsl_version);

    auto& style = ImGui::GetStyle();
    auto& io = ImGui::GetIO();

    static constexpr const char* font_name = "fonts/SourceCodePro-Regular.ttf";
    static constexpr f32_t font_size = 4;


    for (std::size_t i = 0; i < std::size(loco->fonts); ++i) {
      loco->fonts[i] = io.Fonts->AddFontFromFileTTF(font_name, (int)(font_size * (1 << i)) * 2);
      if (loco->fonts[i] == nullptr) {
        fan::throw_error(fan::string("failed to load font:") + font_name);
      }
    }
    io.Fonts->Build();
    io.FontDefault = loco->fonts[2];
  }
#endif
}

void loco_t::init_framebuffer() {

  auto& context = get_context();

#if defined(loco_opengl)
#if defined(loco_framebuffer)
  m_framebuffer.open(context);
  // can be GL_RGB16F
  m_framebuffer.bind(context);
#endif
#endif

#if defined(loco_opengl)

#if defined(loco_framebuffer)

  static auto load_texture = [&](fan::image::image_info_t& image_info, loco_t::image_t& color_buffer, fan::opengl::GLenum attachment, bool reload = false) {
    typename fan::opengl::context_t::image_load_properties_t load_properties;
    load_properties.visual_output = fan::opengl::GL_REPEAT;
    load_properties.internal_format = fan::opengl::GL_RGB;
    load_properties.format = fan::opengl::GL_RGB;
    load_properties.type = fan::opengl::GL_FLOAT;
    load_properties.min_filter = fan::opengl::GL_LINEAR;
    load_properties.mag_filter = fan::opengl::GL_LINEAR;
    if (reload == true) {
      context.image_reload_pixels(color_buffer, image_info, load_properties);
    }
    else {
      color_buffer = context.image_load(image_info, load_properties);
    }
    context.opengl.call(context.opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
    context.image_bind(color_buffer);
    fan::opengl::core::framebuffer_t::bind_to_texture(context, context.image_get(color_buffer), attachment);
  };

  fan::image::image_info_t image_info;
  image_info.data = nullptr;
  image_info.size = window.get_size();

  m_framebuffer.bind(context);
  for (uint32_t i = 0; i < (uint32_t)std::size(color_buffers); ++i) {
    load_texture(image_info, color_buffers[i], fan::opengl::GL_COLOR_ATTACHMENT0 + i);
  }

  window.add_resize_callback([&](const auto& d) {
    fan::image::image_info_t image_info;
    image_info.data = nullptr;
    image_info.size = window.get_size();

    m_framebuffer.bind(context);
    for (uint32_t i = 0; i < (uint32_t)std::size(color_buffers); ++i) {
      load_texture(image_info, color_buffers[i], fan::opengl::GL_COLOR_ATTACHMENT0 + i, true);
    }

    fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
    m_framebuffer.bind(context);
    renderbuffer_properties.size = image_info.size;
    renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
    m_rbo.set_storage(context, renderbuffer_properties);

    fan::vec2 window_size = gloco->window.get_size();
    context.viewport_set(orthographic_camera.viewport, fan::vec2(0, 0), d.size, d.size);

    context.viewport_set(perspective_camera.viewport, fan::vec2(0, 0), d.size, d.size);
  });

  fan::opengl::core::renderbuffer_t::properties_t renderbuffer_properties;
  m_framebuffer.bind(context);
  renderbuffer_properties.size = image_info.size;
  renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_COMPONENT;
  m_rbo.open(context);
  m_rbo.set_storage(context, renderbuffer_properties);
  renderbuffer_properties.internalformat = fan::opengl::GL_DEPTH_ATTACHMENT;
  m_rbo.bind_to_renderbuffer(context, renderbuffer_properties);

  unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

  for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
    attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
  }

  context.opengl.call(context.opengl.glDrawBuffers, std::size(attachments), attachments);
  // finally check if framebuffer is complete
  if (!m_framebuffer.ready(context)) {
    fan::throw_error("framebuffer not ready");
  }


#if defined(loco_post_process)
  static constexpr uint32_t mip_count = 8;
  blur[0].open(window.get_size(), mip_count);

  bloom.open();
#endif

  m_framebuffer.unbind(context);

  
  m_fbo_final_shader = context.shader_create();

  context.shader_set_vertex(
    m_fbo_final_shader,
    context.read_shader("shaders/opengl/2D/effects/loco_fbo.vs")
  );
  context.shader_set_fragment(
    m_fbo_final_shader,
    context.read_shader("shaders/opengl/2D/effects/loco_fbo.fs")
  );
  context.shader_compile(m_fbo_final_shader);

#endif
#endif
}

loco_t::loco_t() : loco_t(properties_t()) {

}

loco_t::loco_t(const properties_t& p) :
  window(p.window_size, fan::window_t::default_window_name, p.window_flags)
{
  context_t::open(&window);
  gloco = this;
  //fan::print("less pain", this, (void*)&lighting, (void*)((uint8_t*)&lighting - (uint8_t*)this), sizeof(*this), lighting.ambient);
  glfwMakeContextCurrent(window);

#if fan_debug >= fan_debug_high
  get_context().set_error_callback();
#endif

  default_texture = get_context().create_missing_texture();

#if defined(loco_opengl)
  initialize_fb_vaos(fb_vao, fb_vbo);
#endif


#if defined(loco_vfi)
  window.add_buttons_callback([this](const fan::window_t::mouse_buttons_cb_data_t& d) {
    fan::vec2 window_size = window.get_size();
    vfi.feed_mouse_button(d.button, d.state);
    });

  window.add_keys_callback([&](const fan::window_t::keyboard_keys_cb_data_t& d) {
    vfi.feed_keyboard(d.key, d.state);
  });

  window.add_mouse_move_callback([&](const fan::window_t::mouse_move_cb_data_t& d) {
    vfi.feed_mouse_move(d.position);
  });

  window.add_text_callback([&](const fan::window_t::text_cb_data_t& d) {
    vfi.feed_text(d.character);
    });
#endif

  shaper.Open();

  {

    // filler
    shaper.AddKey(Key_e::light, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::light_end, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::depth, sizeof(loco_t::depth_t), shaper_t::KeyBitOrderLow);
    shaper.AddKey(Key_e::blending, sizeof(loco_t::blending_t), shaper_t::KeyBitOrderLow);
    shaper.AddKey(Key_e::image, sizeof(loco_t::image_t), shaper_t::KeyBitOrderLow);
    shaper.AddKey(Key_e::viewport, sizeof(loco_t::viewport_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::camera, sizeof(loco_t::camera_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::ShapeType, sizeof(shaper_t::ShapeTypeIndex_t), shaper_t::KeyBitOrderAny);
    shaper.AddKey(Key_e::filler, sizeof(uint8_t), shaper_t::KeyBitOrderAny);

    //gloco->shaper.AddKey(Key_e::image4, sizeof(loco_t::image_t) * 4, shaper_t::KeyBitOrderLow);
  }

  //{
  //  shaper_t::KeyTypeIndex_t ktia[] = {
  //    Key_e::depth,
  //    Key_e::blending,
  //    Key_e::image,
  //    Key_e::image,
  //    Key_e::multitexture,
  //    Key_e::viewport,
  //    Key_e::camera,
  //    Key_e::ShapeType
  //  };
  //  gloco->shaper.AddKeyPack(kp::multitexture, sizeof(ktia) / sizeof(ktia[0]), ktia);
  //}

  // order of open needs to be same with shapes enum
  
  shape_functions.resize(shape_functions.size() + 1); // button
  shape_open<loco_t::sprite_t>(
    &sprite,
    "shaders/opengl/2D/objects/sprite.vs",
    "shaders/opengl/2D/objects/sprite.fs"
  );
  shape_functions.resize(shape_functions.size() + 1); // text
  shape_functions.resize(shape_functions.size() + 1); // hitbox
  shape_open<loco_t::line_t>(
    &line,
    "shaders/opengl/2D/objects/line.vs",
    "shaders/opengl/2D/objects/line.fs"
  );
  shape_functions.resize(shape_functions.size() + 1); // mark
  shape_open<loco_t::rectangle_t>(
    &rectangle,
    "shaders/opengl/2D/objects/rectangle.vs",
    "shaders/opengl/2D/objects/rectangle.fs"
  );
  shape_open<loco_t::light_t>(
    &light,
    "shaders/opengl/2D/objects/light.vs",
    "shaders/opengl/2D/objects/light.fs"
  );
  shape_open<loco_t::unlit_sprite_t>(
    &unlit_sprite,
    "shaders/opengl/2D/objects/sprite.vs",
    "shaders/opengl/2D/objects/unlit_sprite.fs"
  );
  shape_open<loco_t::letter_t>(
    &letter,
    "shaders/opengl/2D/objects/letter.vs",
    "shaders/opengl/2D/objects/letter.fs"
  );
  shape_open<loco_t::circle_t>(
    &circle,
    "shaders/opengl/2D/objects/circle.vs",
    "shaders/opengl/2D/objects/circle.fs"
  );
  shape_open<loco_t::grid_t>(
    &grid,
    "shaders/opengl/2D/objects/grid.vs",
    "shaders/opengl/2D/objects/grid.fs"
  );
  vfi.open();
  shape_open<loco_t::particles_t>(
    &particles,
    "shaders/opengl/2D/effects/particles.vs",
    "shaders/opengl/2D/effects/particles.fs"
  );
  shape_open<loco_t::universal_image_renderer_t>(
    &universal_image_renderer,
    "shaders/opengl/2D/objects/pixel_format_renderer.vs",
    "shaders/opengl/2D/objects/yuv420p.fs"
  );

  shape_open<loco_t::gradient_t>(
    &gradient,
    "shaders/opengl/2D/effects/gradient.vs",
    "shaders/opengl/2D/effects/gradient.fs"
  );

  shape_functions.resize(shape_functions.size() + 1); // light_end

  shape_open<loco_t::shader_shape_t>(
    &shader_shape,
    "shaders/opengl/2D/objects/sprite.vs",
    "shaders/opengl/2D/objects/sprite.fs"
  );

  shape_open<loco_t::rectangle3d_t>(
    &rectangle3d,
    "shaders/opengl/3D/objects/rectangle.vs",
    "shaders/opengl/3D/objects/rectangle.fs"
  );

#if defined(loco_letter)
#if !defined(loco_font)
#define loco_font "fonts/bitter"
#endif
  font.open(loco_font);
#endif

  {
    fan::vec2 window_size = window.get_size();
    {
      orthographic_camera.camera = open_camera(
        fan::vec2(0, window_size.x),
        fan::vec2(0, window_size.y)
      );
      orthographic_camera.viewport = open_viewport(
        fan::vec2(0, 0),
        window_size
      );
    }
    {
      perspective_camera.camera = open_camera_perspective();
      perspective_camera.viewport = open_viewport(
        fan::vec2(0, 0),
        window_size
      );
    }

    //wglMakeCurrent(g_MainWindow.hDC, g_hRC);

#if defined(loco_physics)
    fan::graphics::open_bcol();
#endif


#if defined(loco_imgui)
    init_imgui(this);
    generate_commands(this);
#endif

    init_framebuffer();

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

    loco_t::shader_t shader = shader_create();

    shader_set_vertex(shader,
      read_shader("shaders/opengl/2D/objects/circle.vs")
    );
      
    shader_set_fragment(shader,
      read_shader("shaders/opengl/2D/objects/circle.fs")
    );

    shader_compile(shader);

    gloco->shaper.AddShapeType(
      loco_t::shape_type_t::light_end,
      {
        .MaxElementPerBlock = (shaper_t::MaxElementPerBlock_t)MaxElementPerBlock,
        .RenderDataSize = 0,
        .DataSize = 0,
        .locations = {},
        .shader = shader
      }
    );
    shape_add(
      loco_t::shape_type_t::light_end,
      0,
      0,
      Key_e::light_end, (uint8_t)0,
      Key_e::ShapeType, (loco_t::shaper_t::ShapeTypeIndex_t)loco_t::shape_type_t::light_end
    );
  }
}

void loco_t::process_frame() {

  auto& context = gloco->get_context();

  get_context().opengl.glViewport(0, 0, window.get_size().x, window.get_size().y);

#if defined(loco_framebuffer)
  m_framebuffer.bind(get_context());


  context.opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
  for (std::size_t i = 0; i < std::size(color_buffers); ++i) {
    context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + i);
    context.image_bind(color_buffers[i]);
    context.opengl.glDrawBuffer(fan::opengl::GL_COLOR_ATTACHMENT0 + (uint32_t)(std::size(color_buffers) - 1 - i));
    if (i + (std::size_t)1 == std::size(color_buffers)) {
      context.opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
    }
    context.opengl.call(context.opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
  }
#else
  context.opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
context.opengl.call(context.opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
#endif

  auto it = m_update_callback.GetNodeFirst();
  while (it != m_update_callback.dst) {
    m_update_callback.StartSafeNext(it);
    m_update_callback[it](this);
    it = m_update_callback.EndSafeNext();
  }

  for (const auto& i : single_queue) {
    i();
  }

  single_queue.clear();

  shaper.ProcessBlockEditQueue();

  context.viewport_set(0, window.get_size(), window.get_size());
  static int frames = 0;
  frames++;

    shaper_t::KeyTraverse_t KeyTraverse;
    KeyTraverse.Init(shaper);

    uint32_t texture_count = 0;
    viewport_t viewport;
    viewport.sic();
    camera_t camera;
    camera.sic();

    bool light_buffer_enabled = false;

    { // update 3d view every frame
      auto& camera_perspective = camera_get(perspective_camera.camera);
      camera_perspective.update_view();

      camera_perspective.m_view = camera_perspective.get_view_matrix();
    }

    while (KeyTraverse.Loop(shaper)) {
      
      shaper_t::KeyTypeIndex_t kti = KeyTraverse.kti(shaper);


      switch (kti) {
      case Key_e::blending: {
        uint8_t Key = *(uint8_t*)KeyTraverse.kd();
        if (Key) {
          context.set_depth_test(false);
          context.opengl.call(get_context().opengl.glEnable, fan::opengl::GL_BLEND);
          context.opengl.call(get_context().opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
         // shaper.SetKeyOrder(Key_e::depth, shaper_t::KeyBitOrderLow);
        }
        else {
          context.opengl.call(get_context().opengl.glDisable, fan::opengl::GL_BLEND);
          context.set_depth_test(true);

          //shaper.SetKeyOrder(Key_e::depth, shaper_t::KeyBitOrderHigh);
        }
        break;
      }
      case Key_e::depth: {
#if defined(depth_debug)
        depth_t Key = *(depth_t*)KeyTraverse.kd();
        depth_Key = true;
        fan::print(Key);
#endif
        break;
      }
      case Key_e::image: {
        loco_t::image_t texture = *(loco_t::image_t*)KeyTraverse.kd();
        if (texture.iic() == false) {
          // TODO FIX + 0
          context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 0);
          context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(texture));
          //++texture_count;
        }
        break;
      }
      case Key_e::viewport: {
        viewport = *(loco_t::viewport_t*)KeyTraverse.kd();
        break;
      }
      case Key_e::camera: {
        camera = *(loco_t::camera_t*)KeyTraverse.kd();
        break;
      }
      case Key_e::ShapeType: {
        // if i remove this why it breaks/corrupts?
        if (*(loco_t::shaper_t::ShapeTypeIndex_t*)KeyTraverse.kd() == loco_t::shape_type_t::light_end) {
          continue;
        }
        break;
      }
      case Key_e::light: {
        if (light_buffer_enabled == false) {
#if defined(loco_framebuffer)
          gloco->get_context().set_depth_test(false);
          gloco->get_context().opengl.call(gloco->get_context().opengl.glEnable, fan::opengl::GL_BLEND);
          gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);
          unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

          for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
            attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
          }

          context.opengl.call(context.opengl.glDrawBuffers, std::size(attachments), attachments);
          light_buffer_enabled = true;
#endif
        }
        break;
      }
      case Key_e::light_end: {
        if (light_buffer_enabled) {
#if defined(loco_framebuffer)
          gloco->get_context().set_depth_test(true);
          unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

          for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
            attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
          }

          context.opengl.call(context.opengl.glDrawBuffers, 1, attachments);
          light_buffer_enabled = false;
#endif
          continue;
        }
        break;
      }
      }

      if (KeyTraverse.isbm) {
        
        shaper_t::BlockTraverse_t BlockTraverse;
        shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(shaper, KeyTraverse.bmid());

        if (shape_type == shape_type_t::light_end) {
          break;
        }

   /*     if (shape_type == shape_type_t::vfi) {
          break;
        }*/

        do {
          auto shader = shaper.GetShader(shape_type);
#if fan_debug >= fan_debug_medium
          if (shape_type == loco_t::shape_type_t::vfi || shape_type == loco_t::shape_type_t::light_end) {
            break;
          }
          else if ((shape_type == 0 || shader.iic())) {
            fan::throw_error("invalid stuff");
          }
#endif
          context.shader_use(shader);

          if (camera.iic() == false) {
            context.shader_set_camera(shader, &camera);
          }
          else {
            context.shader_set_camera(shader, &orthographic_camera.camera);
          }
          if (viewport.iic() == false) {
            auto& v = viewport_get(viewport);
            context.viewport_set(v.viewport_position, v.viewport_size, window.get_size());
          }
          context.shader_set_value(shader, "_t00", 0);
          context.shader_set_value(shader, "_t01", 1);
#if defined(depth_debug)
          if (depth_Key) {
            auto& ri = *(fan::vec3*)BlockTraverse.GetRenderData(shaper);
            fan::print("d", ri.z);
          }
#endif
#if fan_debug >= fan_debug_high
          switch (shape_type) {
          default: {
            if (camera.iic()) {
              fan::throw_error("failed to get camera");
            }
            if (viewport.iic()) {
              fan::throw_error("failed to get viewport");
            }
            break;
          }
          }
#endif

          if (shape_type == loco_t::shape_type_t::universal_image_renderer) {
            auto shader = shaper.GetShader(shape_type);
            
            auto& ri = *(universal_image_renderer_t::ri_t*)BlockTraverse.GetData(shaper);

            if (ri.images_rest[0].iic() == false) {
              context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 1);
              context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(ri.images_rest[0]));
              context.shader_set_value(shader, "_t01", 1);
            }
            if (ri.images_rest[1].iic() == false) {
              context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 2);
              context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(ri.images_rest[1]));
              context.shader_set_value(shader, "_t02", 2);
            }

            if (ri.images_rest[2].iic() == false) {
              context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 3);
              context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(ri.images_rest[2]));
              context.shader_set_value(shader, "_t03", 3);
            }
            //fan::throw_error("shaper design is changed");
          }
          else if (shape_type == loco_t::shape_type_t::sprite ||
            shape_type == loco_t::shape_type_t::unlit_sprite || 
            shape_type == loco_t::shape_type_t::shader_shape) {
            //fan::print("shaper design is changed");
            auto& ri = *(sprite_t::ri_t*)BlockTraverse.GetData(shaper);
            auto shader = shaper.GetShader(shape_type);
            for (std::size_t i = 2; i < std::size(ri.images) + 2; ++i) {
              if (ri.images[i - 2].iic() == false) {
                context.shader_set_value(shader, "_t0" + std::to_string(i), i);
                context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + i);
                context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(ri.images[i - 2]));
              }
            }
          }

          if (shape_type != loco_t::shape_type_t::light) {

            if (shape_type == loco_t::shape_type_t::sprite || shape_type == loco_t::shape_type_t::unlit_sprite) {
              context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
              context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(color_buffers[1]));
            }

            auto& c = camera_get(camera);

            context.shader_set_value(
              shader,
              "matrix_size",
              fan::vec2(c.coordinates.right - c.coordinates.left, c.coordinates.down - c.coordinates.up).abs()
            );
            context.shader_set_value(
              shader,
              "viewport",
              fan::vec4(
                viewport_get_position(viewport),
                viewport_get_size(viewport)
              )
            );
            context.shader_set_value(
              shader,
              "window_size",
              fan::vec2(window.get_size())
            );
            context.shader_set_value(
              shader,
              "camera_position",
              c.position
            );
            context.shader_set_value(
              shader,
              "m_time",
              f32_t((fan::time::clock::now() - start_time) / 1e+9)
            );
            //fan::print(fan::time::clock::now() / 1e+9);
            context.shader_set_value(shader, loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
          }

          auto m_vao = shaper.GetVAO(shape_type);
          auto m_vbo = shaper.GetVAO(shape_type);

          m_vao.bind(context);
          m_vbo.bind(context);

          if (context.major < 4 || (context.major == 4 && context.minor < 2)) {
            uintptr_t offset = BlockTraverse.GetRenderDataOffset(shaper);
            std::vector<shape_gl_init_t>& locations = shaper.GetLocations(shape_type);
            for (const auto& location : locations) {
              context.opengl.glVertexAttribPointer(location.index, location.size, location.type, fan::opengl::GL_FALSE, location.stride, (void*)offset);
              switch (location.type) {
              case fan::opengl::GL_FLOAT: {
                offset += location.size * sizeof(fan::opengl::GLfloat);
                break;
              }
              case fan::opengl::GL_UNSIGNED_INT: {
                offset += location.size * sizeof(fan::opengl::GLuint);
                break;
              }
              default: {
                fan::throw_error_impl();
              }
              }
            }
          }

          switch (shape_type) {
          case shape_type_t::rectangle3d: {
            if (context.major >= 4 && context.minor >= 2) {
              context.opengl.glDrawArraysInstancedBaseInstance(
                fan::opengl::GL_TRIANGLES,
                0,
                36,
                BlockTraverse.GetAmount(shaper),
                BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
              );
            }
            else {
              // this is broken somehow with rectangle3d
              context.opengl.glDrawArraysInstanced(
                fan::opengl::GL_TRIANGLES,
                0,
                36,
                BlockTraverse.GetAmount(shaper)
              );
            }
            break;
          }
          case shape_type_t::line: {
            if (context.major >= 4 && context.minor >= 2) {
              context.opengl.glDrawArraysInstancedBaseInstance(
                fan::opengl::GL_LINES,
                0,
                2,
                BlockTraverse.GetAmount(shaper),
                BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
              );
            }
            else {
              context.opengl.glDrawArraysInstanced(
                fan::opengl::GL_LINES,
                0,
                2,
                BlockTraverse.GetAmount(shaper)
              );
            }


            break;
          }
          case shape_type_t::particles: {
            //fan::print("shaper design is changed");
            particles_t::ri_t* pri = (particles_t::ri_t*)BlockTraverse.GetData(shaper);
            loco_t::shader_t shader = shaper.GetShader(shape_type_t::particles);

            for (int i = 0; i < BlockTraverse.GetAmount(shaper); ++i) {
              auto& ri = pri[i];
              context.shader_set_value(shader, "time", (f32_t)((fan::time::clock::now() - ri.begin_time) / 1e+9));
              context.shader_set_value(shader, "vertex_count", 6);
              context.shader_set_value(shader, "count", ri.count);
              context.shader_set_value(shader, "alive_time", (f32_t)(ri.alive_time / 1e+9));
              context.shader_set_value(shader, "respawn_time", (f32_t)(ri.respawn_time / 1e+9));
              context.shader_set_value(shader, "position", *(fan::vec2*)&ri.position);
              context.shader_set_value(shader, "size", ri.size);
              context.shader_set_value(shader, "position_velocity", ri.position_velocity);
              context.shader_set_value(shader, "angle_velocity", ri.angle_velocity);
              context.shader_set_value(shader, "begin_angle", ri.begin_angle);
              context.shader_set_value(shader, "end_angle", ri.end_angle);
              context.shader_set_value(shader, "angle", ri.angle);
              context.shader_set_value(shader, "color", ri.color);
              context.shader_set_value(shader, "gap_size", ri.gap_size);
              context.shader_set_value(shader, "max_spread_size", ri.max_spread_size);
              context.shader_set_value(shader, "size_velocity", ri.size_velocity);

              context.shader_set_value(shader, "shape", ri.shape);

              // TODO how to get begin?
              context.opengl.glDrawArrays(
                fan::opengl::GL_TRIANGLES,
                0,
                ri.count
              );
            }

            break;
          }
          case shape_type_t::letter: {// intended fallthrough
            context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
            context.shader_set_value(
              shader,
              "_t00",
              0
            );
            gloco->image_bind(gloco->font.image);

          }// fallthrough
          default: {
            if (context.major >= 4) {
              context.opengl.glDrawArraysInstancedBaseInstance(
                fan::opengl::GL_TRIANGLES,
                0,
                6,
                BlockTraverse.GetAmount(shaper),
                BlockTraverse.GetRenderDataOffset(shaper) / shaper.GetRenderDataSize(shape_type)
              );
            }
            else {
              context.opengl.glDrawArraysInstanced(
                fan::opengl::GL_TRIANGLES,
                0,
                6,
                BlockTraverse.GetAmount(shaper)
              );
            }

            break;
          }
          }

          //offset = 0;
          //// might be unnecessary
          //for (const auto& location : block.locations) {
          //  context.opengl.glVertexAttribPointer(location.index, location.size, location.type, fan::opengl::GL_FALSE, location.stride, (void*)offset);
          //  switch (location.type) {
          //  case fan::opengl::GL_FLOAT: {
          //    offset += location.size * sizeof(f32_t);
          //    break;
          //  }
          //  case fan::opengl::GL_UNSIGNED_INT: {
          //    offset += location.size * sizeof(uint32_t);
          //    break;
          //  }
          //  default: {
          //    fan::throw_error_impl();
          //  }
          //  }
          //}
          } while (BlockTraverse.Loop(shaper));
      }

    }


  //uint8_t draw_range = 0;
  //for (auto& shape : shape_info_list) {
  //  for (uint8_t i = 0; i < shape.functions.orderio.draw_range; ++i) {
  //    shape.functions.draw(i);
  //  }
  //}


#if defined(loco_framebuffer)

  m_framebuffer.unbind(get_context());

  //fan::print(m_framebuffer.ready(get_context()));

#if defined(loco_post_process)
  blur[0].draw(&color_buffers[0]);
#endif
  //blur[1].draw(&color_buffers[3]);

  context.opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
  context.opengl.call(context.opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
  fan::vec2 window_size = window.get_size();
  context.viewport_set(0, window_size, window_size);

  context.shader_set_value(m_fbo_final_shader, "_t00", 0);
  context.shader_set_value(m_fbo_final_shader, "_t01", 1);

  context.shader_set_value(m_fbo_final_shader, "window_size", window_size);

  context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
  context.image_bind(color_buffers[0]);

#if defined(loco_post_process)
  get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
  context.image_bind(blur[0].mips.front().image);
#endif

  //get_context().opengl.glActiveTexture(fan::opengl::GL_TEXTURE2);
  //blur[1].mips.front().image.bind_texture();

  render_final_fb();

#endif

  for (const auto& i : m_post_draw) {
    i();
  }

#if defined(loco_imgui)

  {
    auto it = m_imgui_draw_cb.GetNodeFirst();
    while (it != m_imgui_draw_cb.dst) {
      m_imgui_draw_cb.StartSafeNext(it);
      m_imgui_draw_cb[it]();
      it = m_imgui_draw_cb.EndSafeNext();
    }
  }

  static constexpr uint32_t parent_window_flags = 0;

  if (toggle_fps) {
    static int initial = 0;
    if (initial == 0) {
      initial = 1;
      ImGui::SetNextWindowSize(fan::vec2(ImGui::GetIO().DisplaySize) / 2.5);
      ImGui::SetNextWindowPos(ImVec2(0, 0));
    }

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

    ImGui::PlotLines("", samples.data(), buffer_size, insert_index, nullptr, 0.0f, FLT_MAX, ImVec2(0, 80));
    ImGui::Text("Current Frame Time: %.2f ms", delta_time);
    ImGui::End();
  }

  if (ImGui::IsKeyPressed(ImGuiKey_F3, false)) {
    render_console = !render_console;
    
    // force focus xd
    console.input.InsertText("a");
    console.input.SetText("");
    console.init_focus = true;
    //TextEditor::Coordinates c;
    //c.mColumn = 0;
    //c.mLine = 0;
    //console.input.SetSelection(c, c);
    //console.input.SetText("a");
    
    //console.input.
    //console.input.SelectAll();
    //console.input.SetCursorPosition(TextEditor::Coordinates(0, 0));
  }
  if (render_console) {
    console.render();
  }
  

#if defined(loco_framebuffer)

  //m_framebuffer.unbind(get_context());

#endif

  ImGui::Render();
  //get_context().opengl.glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
  //get_context().opengl.glClear(fan::opengl::GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

#endif

  glfwSwapBuffers(window);
}

bool loco_t::process_loop(const fan::function_t<void()>& lambda) {
#if defined(loco_imgui)
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  auto& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  const ImVec4 bgColor = ImVec4(0.0f, 0.0f, 0.0f, 0.4f);
  colors[ImGuiCol_WindowBg] = bgColor;
  colors[ImGuiCol_ChildBg] = bgColor;
  colors[ImGuiCol_TitleBg] = bgColor;

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_DockingEmptyBg, ImVec4(0, 0, 0, 0));
  ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
  ImGui::PopStyleColor(2);
#endif

  lambda();
  process_frame();
  window.handle_events();
  if (glfwWindowShouldClose(window)) {
    window.close();
    return 1;
  }//
  return 0;
}

void loco_t::loop(const fan::function_t<void()>& lambda) {
  while (1) {
    if (process_loop(lambda)) {
      break;
    }
  }
}

loco_t::camera_t loco_t::open_camera(const fan::vec2& x, const fan::vec2& y) {
  auto& context = get_context();
  loco_t::camera_t camera = context.camera_create();
  context.camera_set_ortho(camera, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return camera;
}

loco_t::camera_t loco_t::open_camera_perspective(f32_t fov) {
  auto& context = get_context();
  loco_t::camera_t camera = context.camera_create();
  context.camera_set_perspective(camera, fov, window.get_size());
  return camera;
}

loco_t::viewport_t loco_t::open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  auto& context = get_context();
  loco_t::viewport_t viewport = context.viewport_create();
  context.viewport_set(viewport, viewport_position, viewport_size, window.get_size());
  return viewport;
}

void loco_t::set_viewport(loco_t::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
    get_context().viewport_set(viewport, viewport_position, viewport_size, window.get_size());
}

//

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

uint32_t loco_t::get_fps() {
  return window.get_fps();
}

void loco_t::set_vsync(bool flag) {
  get_context().set_vsync(window, flag);
}


#if defined(loco_imgui)
void loco_t::set_imgui_viewport(loco_t::viewport_t viewport) {
  ImVec2 mainViewportPos = ImGui::GetMainViewport()->Pos;

  ImVec2 windowPos = ImGui::GetWindowPos();

  fan::vec2 windowPosRelativeToMainViewport;
  windowPosRelativeToMainViewport.x = windowPos.x - mainViewportPos.x;
  windowPosRelativeToMainViewport.y = windowPos.y - mainViewportPos.y;

  fan::vec2 window_size = window.get_size();
  fan::vec2 viewport_size = ImGui::GetContentRegionAvail();
  fan::vec2 viewport_pos = fan::vec2(windowPosRelativeToMainViewport + fan::vec2(0, ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2));
  viewport_set(
    viewport,
    viewport_pos,
    viewport_size,
    window_size
  );
}
#endif

fan::opengl::context_t& loco_t::get_context() {
  return *dynamic_cast<fan::opengl::context_t*>(this);
}



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

#endif


void loco_t::input_action_t::add(const int* keys, std::size_t count, std::string_view action_name) {
  action_data_t action_data;
  action_data.count = (uint8_t)count;
  std::memcpy(action_data.keys, keys, sizeof(int) * count);
  input_actions[action_name] = action_data;
}

void loco_t::input_action_t::add(int key, std::string_view action_name) {
  add(&key, 1, action_name);
}

void loco_t::input_action_t::add(std::initializer_list<int> keys, std::string_view action_name) {
  add(keys.begin(), keys.size(), action_name);
}

void loco_t::input_action_t::add_keycombo(std::initializer_list<int> keys, std::string_view action_name) {
  action_data_t action_data;
  action_data.combo_count = (uint8_t)keys.size();
  std::memcpy(action_data.key_combos, keys.begin(), sizeof(int) * action_data.combo_count);
  input_actions[action_name] = action_data;
}

bool loco_t::input_action_t::is_active(std::string_view action_name, int pstate) {
  auto found = input_actions.find(action_name);
  if (found != input_actions.end()) {
    action_data_t& action_data = found->second;

    if (action_data.combo_count) {
      int state = none;
      for (int i = 0; i < action_data.combo_count; ++i) {
        int s = gloco->window.key_state(action_data.key_combos[i]);
        if (s == none) {
          return none == loco_t::input_action_t::press;
        }
        if (state == input_action_t::press && s == input_action_t::repeat) {
          state = 1;
        }
        else {
          state = s;
        }
      }
      return state == pstate;
    }
    else if (action_data.count){
      int state = none;
      for (int i = 0; i < action_data.count; ++i) {
        int s = gloco->window.key_state(action_data.keys[i]);
        if (s != none) {
          state = s;
        }
      }
      return state == pstate;
    }
  }
  return none == pstate;
}

static fan::vec2 transform_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

  auto& context = gloco->get_context();
  auto& v = context.viewport_get(viewport);
  auto& c = context.camera_get(camera);

  fan::vec2 viewport_position = v.viewport_position;
  fan::vec2 viewport_size = v.viewport_size;

  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.up;
  f32_t b = c.coordinates.down;

  fan::vec2 tp = p - viewport_position;
  fan::vec2 d = viewport_size;
  tp /= d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  tp += c.position;
  return tp;
}

fan::vec2 loco_t::get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
  return transform_position(get_mouse_position(), viewport, camera);
}

fan::vec2 loco_t::get_mouse_position() {
  return window.get_mouse_position();
  //return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
}

fan::vec2 fan::graphics::get_mouse_position(const fan::graphics::camera_t& camera) {
  return transform_position(gloco->get_mouse_position(), camera.viewport, camera.camera);
}

fan::vec2 loco_t::translate_position(const fan::vec2& p, loco_t::viewport_t viewport, loco_t::camera_t camera) {

  auto& v = gloco->viewport_get(viewport);
  fan::vec2 viewport_position = v.viewport_position;
  fan::vec2 viewport_size = v.viewport_size;

  auto& c = gloco->camera_get(camera);

  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.up;
  f32_t b = c.coordinates.down;

  fan::vec2 tp = p - viewport_position;
  fan::vec2 d = viewport_size;
  tp /= d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  return tp;
}

fan::vec2 loco_t::translate_position(const fan::vec2& p) {
  return translate_position(p, orthographic_camera.viewport, orthographic_camera.camera);
}

void loco_t::shape_t::set_position(const fan::vec3& position) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_position3(this, position);
}

fan::vec3 loco_t::shape_t::get_position() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_position(this);
}

void loco_t::shape_t::set_size(const fan::vec2& size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_size(this, size);
}

void loco_t::shape_t::set_size(const fan::vec3& size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_size3(this, size);
}

fan::vec2 loco_t::shape_t::get_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_size(this);
}

void loco_t::shape_t::set_rotation_point(const fan::vec2& rotation_point) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_rotation_point(this, rotation_point);
}

fan::vec2 loco_t::shape_t::get_rotation_point() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_rotation_point(this);
}

void loco_t::shape_t::set_color(const fan::color& color) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_color(this, color);
}

fan::color loco_t::shape_t::get_color() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_color(this);
}

void loco_t::shape_t::set_angle(const fan::vec3& angle) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_angle(this, angle);
}

fan::vec3 loco_t::shape_t::get_angle() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_angle(this);
}

fan::vec2 loco_t::shape_t::get_tc_position() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tc_position(this);
}

void loco_t::shape_t::set_tc_position(const fan::vec2& tc_position) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_tc_position(this, tc_position);
}

fan::vec2 loco_t::shape_t::get_tc_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tc_size(this);
}

void loco_t::shape_t::set_tc_size(const fan::vec2& tc_size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_tc_size(this, tc_size);
}

bool loco_t::shape_t::load_tp(loco_t::texturepack_t::ti_t* ti) {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].load_tp(this, ti);
}

loco_t::texturepack_t::ti_t loco_t::shape_t::get_tp() {
  loco_t::texturepack_t::ti_t ti;
  ti.image = &gloco->default_texture;
  auto& img = gloco->image_get_data(*ti.image);
  ti.position = get_tc_position() * img.size;
  ti.size = get_tc_size() * img.size;
  return ti;
  //return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_tp(this);
}

bool loco_t::shape_t::set_tp(loco_t::texturepack_t::ti_t* ti) {
  return load_tp(ti);
}

loco_t::camera_t loco_t::shape_t::get_camera() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_camera(this);
}

void loco_t::shape_t::set_camera(loco_t::camera_t camera) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_camera(this, camera);
}

loco_t::viewport_t loco_t::shape_t::get_viewport() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_viewport(this);
}

void loco_t::shape_t::set_viewport(loco_t::viewport_t viewport) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_viewport(this, viewport);
}

fan::vec2 loco_t::shape_t::get_grid_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_grid_size(this);
}

void loco_t::shape_t::set_grid_size(const fan::vec2& grid_size) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_grid_size(this, grid_size);
}

loco_t::image_t loco_t::shape_t::get_image() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_image(this);
}

void loco_t::shape_t::set_image(loco_t::image_t image) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_image(this, image);
}

f32_t loco_t::shape_t::get_parallax_factor() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_parallax_factor(this);
}

void loco_t::shape_t::set_parallax_factor(f32_t parallax_factor) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_parallax_factor(this, parallax_factor);
}

fan::vec3 loco_t::shape_t::get_rotation_vector() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_rotation_vector(this);
}

uint32_t loco_t::shape_t::get_flags() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_flags(this);
}

void loco_t::shape_t::set_flags(uint32_t flag) {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_flags(this, flag);
}

f32_t loco_t::shape_t::get_radius() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_radius(this);
}

fan::vec3 loco_t::shape_t::get_src() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_src(this);
}

fan::vec3 loco_t::shape_t::get_dst() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_dst(this);
}

f32_t loco_t::shape_t::get_outline_size() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_outline_size(this);
}

fan::color loco_t::shape_t::get_outline_color() {
  return gloco->shape_functions[gloco->shaper.GetSTI(*this)].get_outline_color(this);
}

void loco_t::shape_t::reload(uint8_t format, void** image_data, const fan::vec2& image_size, uint32_t filter) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].reload(this, format, image_data, image_size, filter);
}

void loco_t::shape_t::reload(uint8_t format, const fan::vec2& image_size, uint32_t filter) {
  void* data[4]{};
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].reload(this, format, data, image_size, filter);
}

void loco_t::shape_t::set_line(const fan::vec2& src, const fan::vec2& dst) {
  gloco->shape_functions[gloco->shaper.GetSTI(*this)].set_line(this, src, dst);
}

/// shapes +
/// shapes +
/// shapes +
/// shapes +

loco_t::shape_t loco_t::light_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.parallax_factor = properties.parallax_factor;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.rotation_vector = properties.rotation_vector;
  vi.flags = properties.flags;
  vi.angle = properties.angle;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::light, (uint8_t)0,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::line_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.src = properties.src;
  vi.dst = properties.dst;
  vi.color = properties.color;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.src.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::rectangle_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.rotation_point = properties.rotation_point;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::circle_t::push_back(const circle_t::properties_t& properties) {
  circle_t::vi_t vi;
  vi.position = properties.position;
  vi.radius = properties.radius;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.rotation_vector = properties.rotation_vector;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  circle_t::ri_t ri;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::sprite_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  vi.parallax_factor = properties.parallax_factor;
  vi.seed = properties.seed;
  ri_t ri;
  ri.images = properties.images;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::text_t::push_back(const properties_t& properties) {
  return gloco->shaper.add(shape_type_t::text, nullptr, 0, nullptr, nullptr);
}

loco_t::shape_t loco_t::unlit_sprite_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  vi.parallax_factor = properties.parallax_factor;
  vi.seed = properties.seed;
  ri_t ri;
  ri.images = properties.images;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::letter_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.outline_size = properties.outline_size;
  vi.size = properties.size;
  vi.tc_position = properties.tc_position;
  vi.color = properties.color;
  vi.outline_color = properties.outline_color;
  vi.tc_size = properties.tc_size;
  vi.angle = properties.angle;
  ri_t ri;
  ri.font_size = properties.font_size;
  ri.letter_id = properties.letter_id;

  fan::font::character_info_t si = gloco->font.info.get_letter_info(properties.letter_id, properties.font_size);
  auto& image = gloco->image_get_data(gloco->font.image);
  vi.tc_position = si.glyph.position / image.size;
  vi.tc_size.x = si.glyph.size.x / image.size.x;
  vi.tc_size.y = si.glyph.size.y / image.size.y;

  vi.size = si.metrics.size / 2;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::grid_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.grid_size = properties.grid_size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  ri_t ri;
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::particles_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  ri_t ri;
  ri.position = properties.position;
  ri.size = properties.size;
  ri.color = properties.color;

  ri.begin_time = fan::time::clock::now();
  ri.alive_time = properties.alive_time;
  ri.respawn_time = properties.respawn_time;
  ri.count = properties.count;
  ri.position_velocity = properties.position_velocity;
  ri.angle_velocity = properties.angle_velocity;
  ri.begin_angle = properties.begin_angle;
  ri.end_angle = properties.end_angle;
  ri.angle = properties.angle;
  ri.gap_size = properties.gap_size;
  ri.max_spread_size = properties.max_spread_size;
  ri.size_velocity = properties.size_velocity;
  ri.shape = properties.shape;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::universal_image_renderer_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  ri_t ri;
  // + 1
  std::memcpy(ri.images_rest, &properties.images[1], sizeof(ri.images_rest));
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.images[0],
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::gradient_t::push_back(const properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  std::memcpy(vi.color, properties.color, sizeof(vi.color));
  vi.angle = properties.angle;
  vi.rotation_point = properties.rotation_point;
  ri_t ri;

  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

loco_t::shape_t loco_t::shader_shape_t::push_back(const properties_t& properties) {
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.flags = properties.flags;
  vi.tc_position = properties.tc_position;
  vi.tc_size = properties.tc_size;
  vi.parallax_factor = properties.parallax_factor;
  vi.seed = properties.seed;
  ri_t ri;
  ri.images = properties.images;
  loco_t::shape_t ret = shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::image, properties.image,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
  gloco->shaper.GetShader(shape_type) = properties.shader;
  return ret;
}


loco_t::shape_t loco_t::rectangle3d_t::push_back(const properties_t& properties) {
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.color = properties.color;
  vi.angle = properties.angle;
  ri_t ri;

  // might not need depth
  return shape_add(shape_type, vi, ri,
    Key_e::depth, (uint16_t)properties.position.z,
    Key_e::blending, (uint8_t)properties.blending,
    Key_e::viewport, properties.viewport,
    Key_e::camera, properties.camera,
    Key_e::ShapeType, shape_type
  );
}

//-------------------------------------shapes-------------------------------------

void fan::graphics::gl_font_impl::font_t::open(const fan::string& image_path) {
  fan::opengl::context_t::image_load_properties_t lp;
#if defined(loco_opengl)
  lp.min_filter = fan::opengl::GL_LINEAR;
  lp.mag_filter = fan::opengl::GL_LINEAR;
#elif defined(loco_vulkan)
  // fill here
#endif
  image = gloco->image_load(image_path + ".webp", lp);
  fan::font::parse_font(info, image_path + "_metrics.txt");
}

void fan::graphics::gl_font_impl::font_t::close() {
  gloco->image_erase(image);
}

inline fan::vec2 fan::graphics::gl_font_impl::font_t::get_text_size(const fan::string& text, f32_t font_size) {
  fan::vec2 text_size = 0;

  text_size.y = info.get_line_height(font_size);


  for (std::size_t i = 0; i < text.utf8_size(); i++) {
    auto letter = info.get_letter_info(text.get_utf8(i), font_size);

    //auto p = letter_info.metrics.offset.x + letter_info.metrics.size.x / 2 + letter_info.metrics.offset.x;
    text_size.x += letter.metrics.advance;
    //text_size.x += letter.metrics.size.x + letter.metrics.offset.x;
    //if (i + 1 != text.size()) {
    //  text_size.x += letter.metrics.offset.x;
    //}
  }

  return text_size;
}

#if defined(loco_imgui)
IMGUI_API void ImGui::Image(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col) {
  ImGui::Image((ImTextureID)gloco->image_get(img), size, uv0, uv1, tint_col, border_col);
}

IMGUI_API bool ImGui::ImageButton(loco_t::image_t img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {
  return ImGui::ImageButton((ImTextureID)gloco->image_get(img), size, uv0, uv1, frame_padding, bg_col, tint_col);
}

bool ImGui::ToggleButton(const char* str_id, bool* v) {
  ImVec2 p = ImGui::GetCursorScreenPos();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  float height = ImGui::GetFrameHeight();
  float width = height * 1.55f;
  float radius = height * 0.50f;

  bool changed = ImGui::InvisibleButton(str_id, ImVec2(width, height));
  if (changed)
    *v = !*v;
  ImU32 col_bg;
  if (ImGui::IsItemHovered())
    col_bg = *v ? IM_COL32(145 + 20, 211, 68 + 20, 255) : IM_COL32(218 - 20, 218 - 20, 218 - 20, 255);
  else
    col_bg = *v ? IM_COL32(145, 211, 68, 255) : IM_COL32(218, 218, 218, 255);

  draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
  draw_list->AddCircleFilled(ImVec2(*v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));

  return changed;
}


bool ImGui::ToggleImageButton(loco_t::image_t image, const ImVec2& size, bool* toggle)
{
  bool clicked = false;

  ImVec4 tintColor = ImVec4(1, 1, 1, 1);
  if (*toggle) {
    tintColor = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
  }
  if (ImGui::IsItemHovered()) {
    tintColor = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
  }

  if (ImGui::ImageButton(image, size, ImVec2(0, 0), ImVec2(1, 1), 0, ImVec4(0, 0, 0, 0), tintColor)) {
    *toggle = !(*toggle);
    clicked = true;
  }

  return clicked;
}

ImVec2 ImGui::GetPositionBottomCorner(const char* text, uint32_t reverse_yoffset) {
  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  ImVec2 text_size = ImGui::CalcTextSize(text);

  ImVec2 text_pos;
  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

  text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

  return text_pos;
}

void ImGui::DrawTextBottomRight(const char* text, uint32_t reverse_yoffset)
{
    // Retrieve the current window draw list
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Retrieve the current window position and size
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 window_size = ImGui::GetWindowSize();

    // Calculate the size of the text
    ImVec2 text_size = ImGui::CalcTextSize(text);

    // Calculate the position to draw the text (bottom-right corner)
    ImVec2 text_pos;
    text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
    text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;

    text_pos.y -= reverse_yoffset * ImGui::GetTextLineHeightWithSpacing();

    // Draw the text at the calculated position
    draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), text);
}


void fan::graphics::imgui_content_browser_t::render() {
  ImGuiStyle& style = ImGui::GetStyle();
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 16.0f));
  ImGuiWindowClass window_class;
  window_class.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
  ImGui::SetNextWindowClass(&window_class);
  if (ImGui::Begin("Content Browser", 0, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoTitleBar)) {
    if (ImGui::BeginMenuBar()) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

      if (ImGui::ImageButton(icon_arrow_left, fan::vec2(32))) {
        if (std::filesystem::equivalent(current_directory, asset_path) == false) {
          current_directory = current_directory.parent_path();
        }
        update_directory_cache();
      }
      ImGui::SameLine();
      ImGui::ImageButton(icon_arrow_right, fan::vec2(32));
      ImGui::SameLine();
      ImGui::PopStyleColor(3);

      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.f, 0.f, 0.f, 0.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.3f));

      auto image_list = std::to_array({ icon_files_list, icon_files_big_thumbnail });

      fan::vec2 bc = ImGui::GetPositionBottomCorner();

      bc.x -= ImGui::GetWindowPos().x;
      ImGui::SetCursorPosX(bc.x / 2);

      ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - (fan::vec2(64).x + style.ItemSpacing.x) * image_list.size());

      ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
      f32_t y_pos = ImGui::GetCursorPosY() + ImGui::GetStyle().WindowPadding.y;
      ImGui::SetCursorPosY(y_pos);


      if (ImGui::InputText("##content_browser_search", search_buffer.data(), search_buffer.size())) {

      }
      ImGui::PopStyleVar(2);

      ImGui::ToggleImageButton(image_list, fan::vec2(64), (int*)&current_view_mode);

      ImGui::PopStyleColor(3);


      ///ImGui::InputText("Search", search_buffer.data(), search_buffer.size());

      ImGui::EndMenuBar();
    }

    ImGui::PopStyleVar(1);
    // Render content based on view mode
    switch (current_view_mode) {
    case view_mode_large_thumbnails:
      render_large_thumbnails_view();
      break;
    case view_mode_list:
      render_list_view();
      break;
    default:
      break;
    }

    ImGui::End();
  }
}

#endif

#if defined(loco_json)
bool fan::graphics::shape_to_json(loco_t::shape_t& shape, fan::json* json) {
  fan::json& out = *json;
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    out["shape"] = "light";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["rotation_vector"] = shape.get_rotation_vector();
    out["flags"] = shape.get_flags();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::line: {
    out["shape"] = "line";
    out["color"] = shape.get_color();
    out["src"] = shape.get_src();
    out["dst"] = shape.get_dst();
    break;
  }
  case loco_t::shape_type_t::rectangle: {
    out["shape"] = "rectangle";
    out["position"] = shape.get_position();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::sprite: {
    out["shape"] = "sprite";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    out["flags"] = shape.get_flags();
    out["tc_position"] = shape.get_tc_position();
    out["tc_size"] = shape.get_tc_size();
    break;
  }
  case loco_t::shape_type_t::unlit_sprite: {
    out["shape"] = "unlit_sprite";
    out["position"] = shape.get_position();
    out["parallax_factor"] = shape.get_parallax_factor();
    out["size"] = shape.get_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    out["flags"] = shape.get_flags();
    out["tc_position"] = shape.get_tc_position();
    out["tc_size"] = shape.get_tc_size();
    break;
  }
  case loco_t::shape_type_t::letter: {
    out["shape"] = "letter";
    out["position"] = shape.get_position();
    out["outline_size"] = shape.get_outline_size();
    out["size"] = shape.get_size();
    out["tc_position"] = shape.get_tc_position();
    out["color"] = shape.get_color();
    out["outline_color"] = shape.get_outline_color();
    out["tc_size"] = shape.get_tc_size();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::text: {
    out["shape"] = "text";
    break;
  }
  case loco_t::shape_type_t::circle: {
    out["shape"] = "circle";
    out["position"] = shape.get_position();
    out["radius"] = shape.get_radius();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["rotation_vector"] = shape.get_rotation_vector();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::grid: {
    out["shape"] = "grid";
    out["position"] = shape.get_position();
    out["size"] = shape.get_size();
    out["grid_size"] = shape.get_grid_size();
    out["rotation_point"] = shape.get_rotation_point();
    out["color"] = shape.get_color();
    out["angle"] = shape.get_angle();
    break;
  }
  case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)gloco->shaper.GetData(shape);
    out["shape"] = "particles";
    out["position"] = ri.position;
    out["size"] = ri.size;
    out["color"] = ri.color;
    out["begin_time"] = ri.begin_time;
    out["alive_time"] = ri.alive_time;
    out["respawn_time"] = ri.respawn_time;
    out["count"] = ri.count;
    out["position_velocity"] = ri.position_velocity;
    out["angle_velocity"] = ri.angle_velocity;
    out["begin_angle"] = ri.begin_angle;
    out["end_angle"] = ri.end_angle;
    out["angle"] = ri.angle;
    out["gap_size"] = ri.gap_size;
    out["max_spread_size"] = ri.max_spread_size;
    out["size_velocity"] = ri.size_velocity;
    out["particle_shape"] = ri.shape;
    out["blending"] = ri.blending;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}

bool fan::graphics::json_to_shape(const fan::json& in, loco_t::shape_t* shape) {
  std::string shape_type = in["shape"];
  switch (fan::get_hash(shape_type.c_str())) {
  case fan::get_hash("rectangle"): {
    loco_t::rectangle_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
    case fan::get_hash("light"): {
    loco_t::light_t::properties_t p;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.rotation_vector = in["rotation_vector"];
    p.flags = in["flags"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("line"): {
    loco_t::line_t::properties_t p;
    p.color = in["color"];
    p.src = in["src"];
    p.dst = in["dst"];
    *shape = p;
    break;
  }
  case fan::get_hash("sprite"): {
    loco_t::sprite_t::properties_t p;
    p.blending = true;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    p.flags = in["flags"];
    p.tc_position = in["tc_position"];
    p.tc_size = in["tc_size"];
    *shape = p;
    break;
  }
  case fan::get_hash("unlit_sprite"): {
    loco_t::unlit_sprite_t::properties_t p;
    p.blending = true;
    p.position = in["position"];
    p.parallax_factor = in["parallax_factor"];
    p.size = in["size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    p.flags = in["flags"];
    p.tc_position = in["tc_position"];
    p.tc_size = in["tc_size"];
    *shape = p;
    break;
  }
  case fan::get_hash("circle"): {
    loco_t::circle_t::properties_t p;
    p.position = in["position"];
    p.radius = in["radius"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.rotation_vector = in["rotation_vector"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("grid"): {
    loco_t::grid_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.grid_size = in["grid_size"];
    p.rotation_point = in["rotation_point"];
    p.color = in["color"];
    p.angle = in["angle"];
    *shape = p;
    break;
  }
  case fan::get_hash("particles"): {
    loco_t::particles_t::properties_t p;
    p.position = in["position"];
    p.size = in["size"];
    p.color = in["color"];
    p.begin_time = in["begin_time"];
    p.alive_time = in["alive_time"];
    p.respawn_time = in["respawn_time"];
    p.count = in["count"];
    p.position_velocity = in["position_velocity"];
    p.angle_velocity = in["angle_velocity"];
    p.begin_angle = in["begin_angle"];
    p.end_angle = in["end_angle"];
    p.angle = in["angle"];
    p.gap_size = in["gap_size"];
    p.max_spread_size = in["max_spread_size"];
    p.size_velocity = in["size_velocity"];
    p.shape = in["particle_shape"];
    p.blending = in["blending"];
    *shape = p;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}

bool fan::graphics::shape_serialize(loco_t::shape_t& shape, fan::json* out) {
  return shape_to_json(shape, out);
}

#endif

/*
shape
data{
}
*/


bool fan::graphics::shape_to_bin(loco_t::shape_t& shape, std::string* str) {
  std::string& out = *str;
  switch (shape.get_shape_type()) {
  case loco_t::shape_type_t::light: {
    // shape
    fan::write_to_string(out, std::string("light"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_rotation_vector());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_angle());
    break;
  }
  case loco_t::shape_type_t::line: {
    fan::write_to_string(out, std::string("line"));
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_src());
    fan::write_to_string(out, shape.get_dst());
    break;
    case loco_t::shape_type_t::rectangle: {
    fan::write_to_string(out, std::string("rectangle"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::sprite: {
    fan::write_to_string(out, std::string("sprite"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::unlit_sprite: {
    fan::write_to_string(out, std::string("unlit_sprite"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_parallax_factor());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    fan::write_to_string(out, shape.get_flags());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_tc_size());
    break;
    }
    case loco_t::shape_type_t::letter: {
    fan::write_to_string(out, std::string("letter"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_outline_size());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_tc_position());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_outline_color());
    fan::write_to_string(out, shape.get_tc_size());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::circle: {
    fan::write_to_string(out, std::string("circle"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_radius());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_rotation_vector());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::grid: {
    fan::write_to_string(out, std::string("grid"));
    fan::write_to_string(out, shape.get_position());
    fan::write_to_string(out, shape.get_size());
    fan::write_to_string(out, shape.get_grid_size());
    fan::write_to_string(out, shape.get_rotation_point());
    fan::write_to_string(out, shape.get_color());
    fan::write_to_string(out, shape.get_angle());
    break;
    }
    case loco_t::shape_type_t::particles: {
    auto& ri = *(loco_t::particles_t::ri_t*)gloco->shaper.GetData(shape);
    fan::write_to_string(out, std::string("particles"));
    fan::write_to_string(out, ri.position);
    fan::write_to_string(out, ri.size);
    fan::write_to_string(out, ri.color);
    fan::write_to_string(out, ri.begin_time);
    fan::write_to_string(out, ri.alive_time);
    fan::write_to_string(out, ri.respawn_time);
    fan::write_to_string(out, ri.count);
    fan::write_to_string(out, ri.position_velocity);
    fan::write_to_string(out, ri.angle_velocity);
    fan::write_to_string(out, ri.begin_angle);
    fan::write_to_string(out, ri.end_angle);
    fan::write_to_string(out, ri.angle);
    fan::write_to_string(out, ri.gap_size);
    fan::write_to_string(out, ri.max_spread_size);
    fan::write_to_string(out, ri.size_velocity);
    fan::write_to_string(out, ri.shape);
    fan::write_to_string(out, ri.blending);
    break;
    }
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}

bool fan::graphics::bin_to_shape(const std::string& in, loco_t::shape_t* shape, uint64_t& offset) {
  std::string shape_type = fan::read_data<std::string>(in, offset);
  switch (fan::get_hash(shape_type.c_str())) {
  case fan::get_hash("rectangle"): {
    loco_t::rectangle_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    return false;
  }
  case fan::get_hash("light"): {
    loco_t::light_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::read_data<decltype(p.rotation_vector)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("line"): {
    loco_t::line_t::properties_t p;
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.src = fan::read_data<decltype(p.src)>(in, offset);
    p.dst = fan::read_data<decltype(p.dst)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("sprite"): {
    loco_t::sprite_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case fan::get_hash("unlit_sprite"): {
    loco_t::unlit_sprite_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.parallax_factor = fan::read_data<decltype(p.parallax_factor)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.flags = fan::read_data<decltype(p.flags)>(in, offset);
    p.tc_position = fan::read_data<decltype(p.tc_position)>(in, offset);
    p.tc_size = fan::read_data<decltype(p.tc_size)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::circle: {
    loco_t::circle_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.radius = fan::read_data<decltype(p.radius)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.rotation_vector = fan::read_data<decltype(p.rotation_vector)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::grid: {
    loco_t::grid_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.grid_size = fan::read_data<decltype(p.grid_size)>(in, offset);
    p.rotation_point = fan::read_data<decltype(p.rotation_point)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    *shape = p;
    break;
  }
  case loco_t::shape_type_t::particles: {
    loco_t::particles_t::properties_t p;
    p.position = fan::read_data<decltype(p.position)>(in, offset);
    p.size = fan::read_data<decltype(p.size)>(in, offset);
    p.color = fan::read_data<decltype(p.color)>(in, offset);
    p.begin_time = fan::read_data<decltype(p.begin_time)>(in, offset);
    p.alive_time = fan::read_data<decltype(p.alive_time)>(in, offset);
    p.respawn_time = fan::read_data<decltype(p.respawn_time)>(in, offset);
    p.count = fan::read_data<decltype(p.count)>(in, offset);
    p.position_velocity = fan::read_data<decltype(p.position_velocity)>(in, offset);
    p.angle_velocity = fan::read_data<decltype(p.angle_velocity)>(in, offset);
    p.begin_angle = fan::read_data<decltype(p.begin_angle)>(in, offset);
    p.end_angle = fan::read_data<decltype(p.end_angle)>(in, offset);
    p.angle = fan::read_data<decltype(p.angle)>(in, offset);
    p.gap_size = fan::read_data<decltype(p.gap_size)>(in, offset);
    p.max_spread_size = fan::read_data<decltype(p.max_spread_size)>(in, offset);
    p.size_velocity = fan::read_data<decltype(p.size_velocity)>(in, offset);
    p.shape = fan::read_data<decltype(p.shape)>(in, offset);
    p.blending = fan::read_data<decltype(p.blending)>(in, offset);
    *shape = p;
    break;
  }
  default: {
    fan::throw_error("unimplemented");
  }
  }
  return false;
}

bool fan::graphics::shape_serialize(loco_t::shape_t& shape, std::string* out) {
  return shape_to_bin(shape, out);
}

bool fan::graphics::texture_packe0::push_texture(fan::opengl::context_t::image_nr_t image, const texture_properties_t& texture_properties) {

  if (texture_properties.image_name.empty()) {
    fan::print_warning("texture properties name empty");
    return 1;
  }

  for (uint32_t gti = 0; gti < texture_list.size(); gti++) {
    if (texture_list[gti].image_name == texture_properties.image_name) {
      texture_list.erase(texture_list.begin() + gti);
      break;
    }
  }

  auto& context = gloco->get_context();
  auto& img = context.image_get_data(image);

  auto data = context.image_get_pixel_data(image, fan::opengl::GL_RGBA, texture_properties.uv_pos, texture_properties.uv_size);
  fan::vec2ui image_size(
    (uint32_t)(img.size.x * texture_properties.uv_size.x),
    (uint32_t)(img.size.y * texture_properties.uv_size.y)
  );


  if ((int)image_size.x % 2 != 0 || (int)image_size.y % 2 != 0) {
    fan::print_warning("failed to load, image size is not divideable by 2");
    fan::print(texture_properties.image_name, image_size);
    return 1;
  }

  texture_t t;
  t.size = image_size;
  t.decoded_data.resize(t.size.multiply() * 4);
  std::memcpy(t.decoded_data.data(), data.get(), t.size.multiply() * 4);
  t.image_name = texture_properties.image_name;
  t.visual_output = texture_properties.visual_output;
  t.min_filter = texture_properties.min_filter;
  t.mag_filter = texture_properties.mag_filter;
  t.group_id = texture_properties.group_id;

  texture_list.push_back(t);
  return 0;
}

void fan::graphics::texture_packe0::load_compiled(const char* filename) {
  std::ifstream file(filename);
  fan::json j;
  file >> j;

  loaded_pack.resize(j["pack_amount"]);

  std::vector<loco_t::image_t> images;

  for (std::size_t i = 0; i < j["pack_amount"]; i++) {
    loaded_pack[i].texture_list.resize(j["packs"][i]["count"]);

    for (std::size_t k = 0; k < j["packs"][i]["count"]; k++) {
      pack_t::texture_t* t = &loaded_pack[i].texture_list[k];
      std::string image_name = j["packs"][i]["textures"][k]["image_name"];
      t->position = j["packs"][i]["textures"][k]["position"];
      t->size = j["packs"][i]["textures"][k]["size"];
      t->image_name = image_name;
    }

    std::vector<uint8_t> pixel_data = j["packs"][i]["pixel_data"].get<std::vector<uint8_t>>();
    fan::image::image_info_t image_info;
    image_info.data = WebPDecodeRGBA(
      pixel_data.data(),
      pixel_data.size(),
      &image_info.size.x,
      &image_info.size.y
    );
    loaded_pack[i].pixel_data = std::vector<uint8_t>((uint8_t*)image_info.data, (uint8_t*)image_info.data + image_info.size.x * image_info.size.y * 4);


    loaded_pack[i].visual_output = j["packs"][i]["visual_output"];
    loaded_pack[i].min_filter = j["packs"][i]["min_filter"];
    loaded_pack[i].mag_filter = j["packs"][i]["mag_filter"];
    images.push_back(gloco->image_load(image_info));
    WebPFree(image_info.data);
    for (std::size_t k = 0; k < loaded_pack[i].texture_list.size(); ++k) {
      auto& tl = loaded_pack[i].texture_list[k];
      fan::graphics::texture_packe0::texture_properties_t tp;
      tp.group_id = 0;
      tp.uv_pos = fan::vec2(tl.position) / fan::vec2(image_info.size);
      tp.uv_size = fan::vec2(tl.size) / fan::vec2(image_info.size);
      tp.visual_output = loaded_pack[i].visual_output;
      tp.min_filter = loaded_pack[i].min_filter;
      tp.mag_filter = loaded_pack[i].mag_filter;
      tp.image_name = tl.image_name;
      push_texture(images.back(), tp);
    }
  }
}//

void fan::camera::move(f32_t movement_speed, f32_t friction) {
  this->velocity /= friction * gloco->delta_time + 1;
  static constexpr auto minimum_velocity = 0.001;
  if (this->velocity.x < minimum_velocity && this->velocity.x > -minimum_velocity) {
    this->velocity.x = 0;
  }
  if (this->velocity.y < minimum_velocity && this->velocity.y > -minimum_velocity) {
    this->velocity.y = 0;
  }
  if (this->velocity.z < minimum_velocity && this->velocity.z > -minimum_velocity) {
    this->velocity.z = 0;
  }
  if (gloco->window.key_pressed(fan::input::key_w)) {
    this->velocity += this->m_front * (movement_speed * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_s)) {
    this->velocity -= this->m_front * (movement_speed * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_a)) {
    this->velocity -= this->m_right * (movement_speed * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_d)) {
    this->velocity += this->m_right * (movement_speed * gloco->delta_time);
  }

  if (gloco->window.key_pressed(fan::input::key_space)) {
    this->velocity.y += movement_speed * gloco->delta_time;
  }
  if (gloco->window.key_pressed(fan::input::key_left_shift)) {
    this->velocity.y -= movement_speed * gloco->delta_time;
  }

  if (gloco->window.key_pressed(fan::input::key_left)) {
    this->set_yaw(this->get_yaw() - sensitivity * 100 * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_right)) {
    this->set_yaw(this->get_yaw() + sensitivity * 100 * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_up)) {
    this->set_pitch(this->get_pitch() + sensitivity * 100 * gloco->delta_time);
  }
  if (gloco->window.key_pressed(fan::input::key_down)) {
    this->set_pitch(this->get_pitch() - sensitivity * 100 * gloco->delta_time);
  }

  this->position += this->velocity * gloco->delta_time;
  this->update_view();
}

loco_t::shader_t loco_t::create_sprite_shader(const fan::string& fragment) {
  loco_t::shader_t shader = shader_create();
  shader_set_vertex(
    shader,
    loco_t::read_shader("shaders/opengl/2D/objects/sprite.vs")
  );
  shader_set_fragment(shader, fragment);
  shader_compile(shader);
  return shader;
}