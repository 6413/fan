#include "loco.h"
//#define loco_post_process


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
#endif
}

void init_imgui(loco_t* loco) {
#if defined(loco_imgui)
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
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
        fan::throw_error(fan::string("failed to load font") + font_name);
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

  static auto load_texture = [&](fan::webp::image_info_t& image_info, loco_t::image_t& color_buffer, fan::opengl::GLenum attachment, bool reload = false) {
    typename fan::opengl::context_t::image_load_properties_t load_properties;
    load_properties.visual_output = fan::opengl::GL_REPEAT;
    load_properties.internal_format = fan::opengl::GL_RGBA;
    load_properties.format = fan::opengl::GL_RGBA;
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

  fan::webp::image_info_t image_info;
  image_info.data = nullptr;
  image_info.size = window.get_size();

  m_framebuffer.bind(context);
  for (uint32_t i = 0; i < (uint32_t)std::size(color_buffers); ++i) {
    load_texture(image_info, color_buffers[i], fan::opengl::GL_COLOR_ATTACHMENT0 + i);
  }

  window.add_resize_callback([&](const auto& d) {
    fan::webp::image_info_t image_info;
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

  get_context().set_error_callback();

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
    gloco->shaper.AddKey(Key_e::blending, sizeof(loco_t::blending_t), shaper_t::KeyBitOrderLow);
    gloco->shaper.AddKey(Key_e::depth, sizeof(loco_t::depth_t), shaper_t::KeyBitOrderLow);
    gloco->shaper.AddKey(Key_e::image, sizeof(loco_t::image_t), shaper_t::KeyBitOrderLow);
    gloco->shaper.AddKey(Key_e::viewport, sizeof(loco_t::viewport_t), shaper_t::KeyBitOrderAny);
    gloco->shaper.AddKey(Key_e::camera, sizeof(loco_t::camera_t), shaper_t::KeyBitOrderAny);
    gloco->shaper.AddKey(Key_e::ShapeType, sizeof(shaper_t::ShapeTypeIndex_t), shaper_t::KeyBitOrderAny);
    gloco->shaper.AddKey(Key_e::filler, sizeof(uint8_t), shaper_t::KeyBitOrderAny);
  }

  {
    shaper_t::KeyTypeIndex_t ktia[] = {
      Key_e::blending,
      Key_e::depth,
      Key_e::viewport,
      Key_e::camera,
      Key_e::ShapeType
    };
    gloco->shaper.AddKeyPack(kp::common, sizeof(ktia) / sizeof(ktia[0]), ktia);
  }
  {
    shaper_t::KeyTypeIndex_t ktia[] = {
      Key_e::filler
    };
    gloco->shaper.AddKeyPack(kp::vfi, sizeof(ktia) / sizeof(ktia[0]), ktia);
  }
  {
    shaper_t::KeyTypeIndex_t ktia[] = {
      Key_e::blending,
      Key_e::depth,
      Key_e::image,
      Key_e::viewport,
      Key_e::camera,
      Key_e::ShapeType
    };
    gloco->shaper.AddKeyPack(kp::texture, sizeof(ktia) / sizeof(ktia[0]), ktia);
  }
  {
    shaper_t::KeyTypeIndex_t ktia[] = {
      Key_e::viewport,
      Key_e::camera,
      Key_e::ShapeType
    };
    gloco->shaper.AddKeyPack(kp::light, sizeof(ktia) / sizeof(ktia[0]), ktia);
  }

  // order of open needs to be same with shapes enum
  
  gloco->shape_functions.resize(gloco->shape_functions.size() + 1); // button
  shape_open<loco_t::sprite_t>(
    "shaders/opengl/2D/objects/sprite.vs",
    "shaders/opengl/2D/objects/sprite.fs"
  );
  gloco->shape_functions.resize(gloco->shape_functions.size() + 1); // text
  gloco->shape_functions.resize(gloco->shape_functions.size() + 1); // hitbox
  shape_open<loco_t::line_t>(
    "shaders/opengl/2D/objects/line.vs",
    "shaders/opengl/2D/objects/line.fs"
  );
  shape_open<loco_t::rectangle_t>(
    "shaders/opengl/2D/objects/rectangle.vs",
    "shaders/opengl/2D/objects/rectangle.fs"
  );
  shape_open<loco_t::light_t>(
    "shaders/opengl/2D/objects/light.vs",
    "shaders/opengl/2D/objects/light.fs"
  );
  shape_open<loco_t::unlit_sprite_t>(
    "shaders/opengl/2D/objects/sprite.vs",
    "shaders/opengl/2D/objects/unlit_sprite.fs"
  );
  shape_open<loco_t::letter_t>(
    "shaders/opengl/2D/objects/letter.vs",
    "shaders/opengl/2D/objects/letter.fs"
  );
  shape_open<loco_t::circle_t>(
    "shaders/opengl/2D/objects/circle.vs",
    "shaders/opengl/2D/objects/circle.fs"
  );
  shape_open<loco_t::grid_t>(
    "shaders/opengl/2D/objects/grid.vs",
    "shaders/opengl/2D/objects/grid.fs"
  );
  vfi.open();
  shape_open<loco_t::particles_t>(
    "shaders/opengl/2D/effects/particles.vs",
    "shaders/opengl/2D/effects/particles.fs"
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
      perspective_camera.camera = open_camera(
        fan::vec2(0, window_size.x),
        fan::vec2(0, window_size.y)
      );
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

  }
}

void process_render_data_queue(shaper_t& shaper, fan::opengl::context_t& context) {
  // iterate gpu write queues
  auto nr = shaper.BlockQueue.GetNodeFirst();
  while (nr != shaper.BlockQueue.dst) {
    auto& bq = shaper.BlockQueue[nr];
    shaper_t::bm_BaseData_t& bm = *(shaper_t::bm_BaseData_t*)shaper.KeyPacks[shaper.ShapeTypes[bq.sti].KeyPackIndex].bm[bq.bmid];
    uint8_t* renderdata = shaper._GetRenderData(bm.sti, bq.blid, 0);
    auto& perblockdata = shaper.GetPerBlockData(bm.sti, bq.blid);
    uint64_t src = perblockdata.MinEdit;
    uint64_t dst = perblockdata.MaxEdit;
    auto& block = shaper.GetShapeType(bm.sti);

    renderdata += src;

    block.m_vao.bind(context);
    fan::opengl::core::edit_glbuffer(
      gloco->get_context(),
      block.m_vbo.m_buffer,
      renderdata,
      (uintptr_t)bq.blid.NRI * block.RenderDataSize * block.MaxElementPerBlock() + src,
      dst - src,
      fan::opengl::GL_ARRAY_BUFFER
    );
    shaper.GetPerBlockData(bm.sti, bq.blid).clear();
    nr = nr.Next(&shaper.BlockQueue);
  }

  shaper.BlockQueue.Clear();
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

  process_render_data_queue(shaper, context);

  shaper_t::BlockID_t BlockID;

  context.viewport_set(0, window.get_size(), window.get_size());
  static int frames = 0;
  frames++;

  shaper_t::KeyPackTraverse_t KeyPackTraverse;
  KeyPackTraverse.Init(shaper);
  while (KeyPackTraverse.Loop(shaper)) {

    shaper_t::KeyTraverse_t KeyTraverse;
    KeyTraverse.Init(shaper, KeyPackTraverse.kpi);

    shaper_t::KeyTypeIndex_t kti;
    uint32_t texture_count = 0;
    viewport_t viewport;
    viewport.sic();
    camera_t camera;
    camera.sic();
    while (KeyTraverse.Loop(shaper, kti)) {
      
      switch (KeyPackTraverse.kpi) {
      case kp::light: {
#if defined(loco_framebuffer)
        gloco->get_context().set_depth_test(false);
        gloco->get_context().opengl.call(gloco->get_context().opengl.glEnable, fan::opengl::GL_BLEND);
        gloco->get_context().opengl.call(gloco->get_context().opengl.glBlendFunc, fan::opengl::GL_ONE, fan::opengl::GL_ONE);
        unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

        for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
          attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
        }

        context.opengl.call(context.opengl.glDrawBuffers, std::size(attachments), attachments);
#endif
        break;
      }
      }
      if (KeyPackTraverse.kpi == kp::vfi) {
        continue;
      }


      switch (kti) {
      case (shaper_t::KeyTypeIndex_t)-1: {
        shaper_t::BlockTraverse_t BlockTraverse;
        shaper_t::ShapeTypeIndex_t shape_type = BlockTraverse.Init(shaper, KeyPackTraverse.kpi, KeyTraverse.bmid(shaper));
        do {
          auto& block = shaper.ShapeTypes[shape_type];
#if fan_debug >= fan_debug_medium
          if (shape_type == 0 || block.shader.iic()) {
            fan::print("invalid stuff");
            break;
            //fan::throw_error("invalid stuff");
          }
#endif
          context.shader_use(block.shader);
          if (camera.iic() == false) {
            context.shader_set_camera(block.shader, &camera);
          }
          else {
            context.shader_set_camera(block.shader, &orthographic_camera.camera);
          }
          if (viewport.iic() == false) {
            auto& v = viewport_get(viewport);
            context.viewport_set(v.viewport_position, v.viewport_size, window.get_size());
          }
          context.shader_set_value(block.shader, "_t00", 0);
          context.shader_set_value(block.shader, "_t01", 1);

#if fan_debug >= fan_debug_high
          switch (shape_type) {
          case shape_type_t::light: {
            break;
          }
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

          if (shape_type != loco_t::shape_type_t::light) {
            auto& c = camera_get(camera);

            context.shader_set_value(
              block.shader,
              "matrix_size",
              fan::vec2(c.coordinates.right - c.coordinates.left, c.coordinates.down - c.coordinates.up).abs()
            );
            context.shader_set_value(
              block.shader,
              "viewport",
              fan::vec4(
                viewport_get_position(viewport),
                viewport_get_size(viewport)
              )
            );
            context.shader_set_value(
              block.shader,
              "window_size",
              fan::vec2(window.get_size())
            );
            context.shader_set_value(
              block.shader,
              "camera_position",
              camera_get_position(camera)
            );

            context.shader_set_value(block.shader, loco_t::lighting_t::ambient_name, gloco->lighting.ambient);
          }

          block.m_vao.bind(context);

          auto nri = BlockTraverse.GetBlockID().NRI;

          block.m_vbo.bind(context);

          uintptr_t offset = (uintptr_t)nri * block.MaxElementPerBlock() * block.RenderDataSize;
          for (const auto& location : block.locations) {
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

          switch (shape_type) {
          case shape_type_t::line: {
            context.opengl.glDrawArraysInstanced(
              fan::opengl::GL_LINES,
              0,
              6,
              BlockTraverse.GetAmount(shaper)
            );
            break;
          }
          case shape_type_t::particles: {
            auto& ri = *(particles_t::ri_t*)BlockTraverse.GetData(shaper);

            context.shader_set_value(block.shader, "time", (f64_t)(fan::time::clock::now() - ri.begin_time) / 1e+9);
            context.shader_set_value(block.shader, "vertex_count", 6);
            context.shader_set_value(block.shader, "count", ri.count);
            context.shader_set_value(block.shader, "alive_time", (f32_t)ri.alive_time / 1e+9);
            context.shader_set_value(block.shader, "respawn_time", (f32_t)ri.respawn_time / 1e+9);
            context.shader_set_value(block.shader, "position", *(fan::vec2*)&ri.position);
            context.shader_set_value(block.shader, "size", ri.size);
            context.shader_set_value(block.shader, "position_velocity", ri.position_velocity);
            context.shader_set_value(block.shader, "angle_velocity", ri.angle_velocity);
            context.shader_set_value(block.shader, "begin_angle", ri.begin_angle);
            context.shader_set_value(block.shader, "end_angle", ri.end_angle);
            context.shader_set_value(block.shader, "angle", ri.angle);
            context.shader_set_value(block.shader, "color", ri.color);
            context.shader_set_value(block.shader, "gap_size", ri.gap_size);
            context.shader_set_value(block.shader, "max_spread_size", ri.max_spread_size);
            context.shader_set_value(block.shader, "size_velocity", ri.size_velocity);

            context.shader_set_value(block.shader, "shape", ri.shape);

            // TODO how to get begin?
            context.opengl.glDrawArrays(
              fan::opengl::GL_TRIANGLES,
              0,
              ri.count
            );
            break;
          }
          case shape_type_t::letter: {// fallthrough
            context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0);
            context.shader_set_value(
              block.shader,
              "_t00",
              0
            );
            gloco->image_bind(gloco->font.image);
            
          }// fallthrough
          default: {
            // if gl_major ..
            /*context.opengl.glDrawArraysInstancedBaseInstance(
              fan::opengl::GL_TRIANGLES,
              0,
              6,
              BlockTraverse.GetAmount(shaper),
              nri * block.MaxElementPerBlock()
            );*/
            context.opengl.glDrawArraysInstanced(
              fan::opengl::GL_TRIANGLES,
              0,
              6,
              BlockTraverse.GetAmount(shaper)
            );
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
        break;
      }
      case Key_e::blending: {
        uint8_t Key = *(uint8_t*)KeyTraverse.KeyData;
        if (Key) {
          context.set_depth_test(false);
          context.opengl.call(get_context().opengl.glEnable, fan::opengl::GL_BLEND);
          context.opengl.call(get_context().opengl.glBlendFunc, fan::opengl::GL_SRC_ALPHA, fan::opengl::GL_ONE_MINUS_SRC_ALPHA);
        }
        else {
          context.opengl.call(get_context().opengl.glDisable, fan::opengl::GL_BLEND);
          context.set_depth_test(true);
        }
        break;
      }
      case Key_e::depth: {
        break;
      }
      case Key_e::image: {
        loco_t::image_t texture = *(loco_t::image_t*)KeyTraverse.KeyData;
        if (texture.iic() == false) {
          // TODO FIX + 0
          context.opengl.glActiveTexture(fan::opengl::GL_TEXTURE0 + 0);
          context.opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, context.image_get(texture));
          //++texture_count;
        }
        break;
      }
      case Key_e::viewport: {
        viewport = *(loco_t::viewport_t*)KeyTraverse.KeyData;
        break;
      }
      case Key_e::camera: {
        camera = *(loco_t::camera_t*)KeyTraverse.KeyData;
        break;
      }
      case Key_e::ShapeType: {
        break;
      }
      }
    }

    switch (KeyPackTraverse.kpi) {
    case kp::light: {
#if defined(loco_framebuffer)
      gloco->get_context().set_depth_test(true);
      unsigned int attachments[sizeof(color_buffers) / sizeof(color_buffers[0])];

      for (uint8_t i = 0; i < std::size(color_buffers); ++i) {
        attachments[i] = fan::opengl::GL_COLOR_ATTACHMENT0 + i;
      }

      context.opengl.call(context.opengl.glDrawBuffers, 1, attachments);
#endif
      break;
    }
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
    toggle_console = !toggle_console;
    
    // force focus xd
    console.input.InsertText("a");
    console.input.SetText("");
    //TextEditor::Coordinates c;
    //c.mColumn = 0;
    //c.mLine = 0;
    //console.input.SetSelection(c, c);
    //console.input.SetText("a");
    
    //console.input.
    //console.input.SelectAll();
    //console.input.SetCursorPosition(TextEditor::Coordinates(0, 0));
    console.set_input_focus();
  }
  if (toggle_console) {
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

int loco_t::input_action_t::is_active(std::string_view action_name, int press) {
  auto found = input_actions.find(action_name);
  if (found != input_actions.end()) {
    action_data_t& action_data = found->second;

    int state = none;
    for (int i = 0; i < action_data.count; ++i) {
      int s = gloco->window.key_state(action_data.keys[i]);
      if (s != none) {
        state = s;
      }
    }
    return state;
  }
  return none;
}

fan::vec2 loco_t::get_mouse_position(const loco_t::camera_t& camera, const loco_t::viewport_t& viewport) {
  fan::vec2 mouse_pos = window.get_mouse_position();
  fan::vec2 translated_pos;
  auto& context = get_context();
  auto& v = context.viewport_get(viewport);
  auto& c = context.camera_get(camera);
  translated_pos.x = fan::math::map(mouse_pos.x, v.viewport_position.x, v.viewport_position.x + v.viewport_size.x, c.coordinates.left, c.coordinates.right);
  translated_pos.y = fan::math::map(mouse_pos.y, v.viewport_position.y, v.viewport_position.y + v.viewport_size.y, c.coordinates.up, c.coordinates.down);
  return translated_pos;
}

fan::vec2 loco_t::get_mouse_position() {
  return window.get_mouse_position();
  //return get_mouse_position(gloco->default_camera->camera, gloco->default_camera->viewport); behaving oddly
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
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_position3(this, position);
}

fan::vec3 loco_t::shape_t::get_position() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_position(this);
}

void loco_t::shape_t::set_size(const fan::vec2& size) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_size(this, size);
}

fan::vec2 loco_t::shape_t::get_size() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_size(this);
}

void loco_t::shape_t::set_rotation_point(const fan::vec2& rotation_point) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_rotation_point(this, rotation_point);
}

fan::vec2 loco_t::shape_t::get_rotation_point() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_rotation_point(this);
}

void loco_t::shape_t::set_color(const fan::color& color) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_color(this, color);
}

fan::color loco_t::shape_t::get_color() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_color(this);
}

void loco_t::shape_t::set_angle(const fan::vec3& angle) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_angle(this, angle);
}

fan::vec3 loco_t::shape_t::get_angle() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_angle(this);
}

fan::vec2 loco_t::shape_t::get_tc_position() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_tc_position(this);
}

void loco_t::shape_t::set_tc_position(const fan::vec2& tc_position) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_tc_position(this, tc_position);
}

fan::vec2 loco_t::shape_t::get_tc_size() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_tc_size(this);
}

void loco_t::shape_t::set_tc_size(const fan::vec2& tc_size) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_tc_size(this, tc_size);
}

bool loco_t::shape_t::load_tp(loco_t::texturepack_t::ti_t* ti) {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].load_tp(this, ti);
}

loco_t::texturepack_t::ti_t loco_t::shape_t::get_tp() {
  loco_t::texturepack_t::ti_t ti;
  ti.image = &gloco->default_texture;
  auto& img = gloco->image_get_data(*ti.image);
  ti.position = get_tc_position() * img.size;
  ti.size = get_tc_size() * img.size;
  return ti;
  //return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_tp(this);
}

bool loco_t::shape_t::set_tp(loco_t::texturepack_t::ti_t* ti) {
  return load_tp(ti);
}

loco_t::camera_t loco_t::shape_t::get_camera() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_camera(this);
}

void loco_t::shape_t::set_camera(loco_t::camera_t camera) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_camera(this, camera);
}

loco_t::viewport_t loco_t::shape_t::get_viewport() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_viewport(this);
}

void loco_t::shape_t::set_viewport(loco_t::viewport_t viewport) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_viewport(this, viewport);
}

fan::vec2 loco_t::shape_t::get_grid_size() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_grid_size(this);
}

void loco_t::shape_t::set_grid_size(const fan::vec2& grid_size) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_grid_size(this, grid_size);
}

void loco_t::shape_t::set_image(loco_t::image_t image) {
  gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].set_image(this, image);
}

f32_t loco_t::shape_t::get_parallax_factor() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_parallax_factor(this);
}

fan::vec3 loco_t::shape_t::get_rotation_vector() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_rotation_vector(this);
}

uint32_t loco_t::shape_t::get_flags() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_flags(this);
}

f32_t loco_t::shape_t::get_radius() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_radius(this);
}

fan::vec3 loco_t::shape_t::get_src() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_src(this);
}

fan::vec3 loco_t::shape_t::get_dst() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_dst(this);
}

f32_t loco_t::shape_t::get_outline_size() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_outline_size(this);
}

fan::color loco_t::shape_t::get_outline_color() {
  return gloco->shape_functions[gloco->shaper.ShapeList[*this].sti].get_outline_color(this);
}

/// shapes +
/// shapes +
/// shapes +
/// shapes +

loco_t::shape_t loco_t::light_t::push_back(const properties_t& properties) {
  kps_t::light_t KeyPack;
  KeyPack.ShapeType = shape_type_t::light;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  //KeyPack.ShapeType = shape_type;
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

  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::line_t::push_back(const properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type_t::line;
  KeyPack.depth = properties.src.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.src = properties.src;
  vi.dst = properties.dst;
  vi.color = properties.color;
  ri_t ri;

  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::rectangle_t::push_back(const properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type_t::rectangle;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.color = properties.color;
  vi.angle = properties.angle;
  vi.rotation_point = properties.rotation_point;
  ri_t ri;

  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

shaper_t::ShapeID_t loco_t::circle_t::push_back(const circle_t::properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type_t::circle;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  circle_t::vi_t vi;
  vi.position = properties.position;
  vi.radius = properties.radius;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.rotation_vector = properties.rotation_vector;
  vi.angle = properties.angle;
  circle_t::ri_t ri;
  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::sprite_t::push_back(const properties_t& properties) {
  kps_t::texture_t KeyPack;
  KeyPack.ShapeType = properties.shape_type;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.image = properties.image;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
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
  ri_t ri;
  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::text_t::push_back(const properties_t& properties) {
  return gloco->shaper.add(shape_type_t::text, nullptr, nullptr, nullptr);
}

loco_t::shape_t loco_t::unlit_sprite_t::push_back(const properties_t& properties) {
  kps_t::texture_t KeyPack;
  KeyPack.ShapeType = properties.shape_type;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.image = properties.image;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
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
  ri_t ri;
  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::letter_t::push_back(const properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type_t::letter;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
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

  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::grid_t::push_back(const properties_t& properties) {
  kps_t::common_t KeyPack;
  KeyPack.ShapeType = shape_type_t::grid;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
  //KeyPack.ShapeType = shape_type;
  vi_t vi;
  vi.position = properties.position;
  vi.size = properties.size;
  vi.grid_size = properties.grid_size;
  vi.rotation_point = properties.rotation_point;
  vi.color = properties.color;
  vi.angle = properties.angle;
  ri_t ri;
  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

loco_t::shape_t loco_t::particles_t::push_back(const properties_t& properties) {
  kps_t::texture_t KeyPack;
  KeyPack.ShapeType = shape_type_t::particles;
  KeyPack.depth = properties.position.z;
  KeyPack.blending = properties.blending;
  KeyPack.image = properties.image;
  KeyPack.camera = properties.camera;
  KeyPack.viewport = properties.viewport;
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

  return gloco->shaper.add(KeyPack.ShapeType, &KeyPack, &vi, &ri);
}

void shaper_t::SetPerBlockData(bm_t::nr_t bm_id, ShapeTypeAmount_t ShapeType, BlockID_t BlockID) {
  BlockUnique_t& block_data = GetPerBlockData(ShapeType, BlockID);
  block_data.clear();
  bm_BaseData_t& bm_base = *(bm_BaseData_t*)KeyPacks[ShapeTypes[ShapeType].KeyPackIndex].bm[bm_id];
  // opengl todo 
  auto nr = bm_base.FirstBlockNR;
  auto& block = ShapeTypes[ShapeType];


  block.m_vbo.bind(gloco->get_context());
  if (block.render_data_size < block.BlockList.NodeList.Possible * block.RenderDataSize * block.MaxElementPerBlock()) {
    block.render_data_size = block.BlockList.NodeList.Possible * block.RenderDataSize * block.MaxElementPerBlock();
    // TODO dont malloc everytime
    fan::opengl::core::write_glbuffer(
      gloco->get_context(),
      block.m_vbo.m_buffer,
      0,
      block.render_data_size, // ?
      fan::opengl::GL_DYNAMIC_DRAW,
      fan::opengl::GL_ARRAY_BUFFER
    );
    ElementFullyEdited(bm_id, bm_base.LastBlockNR, 0, ShapeType);
    gl_buffer_is_reseted(ShapeType);
  }
}

void shaper_t::gl_buffer_is_reseted(shaper_t::ShapeTypeAmount_t ShapeType) {
  auto& block = ShapeTypes[ShapeType];
  decltype(block.BlockList)::BlockList_nrtra_t traverse;
  traverse.Open(&block.BlockList);
  block.m_vao.bind(gloco->get_context());
  while (traverse.Loop(&block.BlockList)) {
    fan::opengl::core::edit_glbuffer(
      gloco->get_context(),
      block.m_vbo.m_buffer,
      _GetRenderData(ShapeType, traverse.nr, 0),
      (uintptr_t)traverse.nr.NRI * block.RenderDataSize * block.MaxElementPerBlock(),
      block.RenderDataSize * block.MaxElementPerBlock(),
      fan::opengl::GL_ARRAY_BUFFER
    );

  }
  traverse.Close(&block.BlockList);
}

shaper_t::ShapeID_t shaper_t::add(
  ShapeTypeIndex_t sti,
  const void* KeyDataArray,
  const void* RenderData,
  const void* Data
) {
  bm_NodeReference_t bmnr;
  static int salsa = 0;
  bm_BaseData_t* bmbase;

  auto& st = ShapeTypes[sti];
  auto& kp = KeyPacks[st.KeyPackIndex];

  auto _KeyDataArray = (KeyData_t*)KeyDataArray;

  KeyTree_NodeReference_t nr = kp.KeyTree_root;
  for (KeyIndexInPack_t kiip = 0; kiip < kp.KeyAmount; kiip++) {
    auto kt = &KeyTypes[kp.KeyIndexes[kiip]];
    Key_t::KeySize_t bdbt_ki;
    Key_t::q(&KeyTree, kt->sibit(), _KeyDataArray, &bdbt_ki, &nr);
    if (bdbt_ki != kt->sibit()) {
      /* query failed to find rest so lets make new block manager */

      bmnr = kp.bm.NewNode();
      bmbase = (bm_BaseData_t*)kp.bm[bmnr];
      bmbase->sti = sti;
      bmbase->FirstBlockNR = st.BlockList.NewNode();
      bmbase->LastBlockNR = bmbase->FirstBlockNR;
      SetPerBlockData(bmnr, bmbase->sti, bmbase->FirstBlockNR);
      bmbase->LastBlockElementCount = 0;
      __MemoryCopy(KeyDataArray, &bmbase[1], kp.KeySizesSum);

      do {
        KeyTree_NodeReference_t out;
        if (kiip + 1 != kp.KeyAmount) {
          out = KeyTree_NewNode(&KeyTree);
        }
        else {
          out = *(KeyTree_NodeReference_t*)&bmnr;
        }

        Key_t::a(&KeyTree, kt->sibit(), _KeyDataArray, bdbt_ki, nr, out);
        bdbt_ki = 0;

        nr = out;

        _KeyDataArray += kt->Size;

        if (++kiip == kp.KeyAmount) {
          break;
        }

        kt = &KeyTypes[kp.KeyIndexes[kiip]];
      } while (1);

      goto gt_NoNewBlockManager;
    }

    _KeyDataArray += kt->Size;
  }
  //printf("%u\n", bmnr.NRI);
  //printf("%u\n", nr);
  //printf("%u\n", (*(bm_NodeReference_t*)&nr).NRI);
  bmnr = *(bm_NodeReference_t*)&nr;
  //printf("%u\n", bmnr.NRI);
  bmbase = (bm_BaseData_t*)kp.bm[bmnr];

  if (bmbase->LastBlockElementCount == st.MaxElementPerBlock_m1) {
    bmbase->LastBlockElementCount = 0;
    auto blnr = st.BlockList.NewNode();
    st.BlockList.linkNextOfOrphan(bmbase->LastBlockNR, blnr);
    bmbase->LastBlockNR = blnr;
    SetPerBlockData(bmnr, bmbase->sti, blnr);
  }
  else {
    bmbase->LastBlockElementCount++;
  }

gt_NoNewBlockManager:

  auto shapeid = ShapeList.NewNode();
  ShapeList[shapeid].sti = sti;
  ShapeList[shapeid].bmid = bmnr;
  ShapeList[shapeid].blid = bmbase->LastBlockNR;
  ShapeList[shapeid].ElementIndex = bmbase->LastBlockElementCount;

  __MemoryCopy(
    RenderData,
    _GetRenderData(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount),
    st.RenderDataSize
  );
  auto var = _GetRenderData(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount);
  __MemoryCopy(
    Data,
    _GetData(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount),
    st.DataSize
  );
  _GetShapeID(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount) = shapeid;
  ElementFullyEdited(bmnr, bmbase->LastBlockNR, bmbase->LastBlockElementCount, sti);

  if (shapeid.NRI == 5) {
    printf("");
  }

  return shapeid;
}

void shaper_t::AddShapeType(
  ShapeTypeIndex_t sti,
  KeyPackIndex_t kpi,
  const BlockProperties_t bp
) {
  if (sti >= ShapeTypeAmount) {
    ShapeTypes.resize(((uintptr_t)sti + 1));
    for (; ShapeTypeAmount < (ShapeTypeAmount_t)sti + 1; ShapeTypeAmount++) { /* filler open */
      ShapeTypes[ShapeTypeAmount].BlockList.Open(1);
    }
  }
  auto& st = ShapeTypes[sti];
  st.BlockList.Close();
  st.BlockList.Open(
    (
      (uintptr_t)bp.RenderDataSize + bp.DataSize + sizeof(ShapeList_t::nr_t)
      ) * (bp.MaxElementPerBlock) + sizeof(BlockUnique_t)
  );

  st.KeyPackIndex = kpi;

  st.MaxElementPerBlock_m1 = bp.MaxElementPerBlock - 1;
  st.RenderDataSize = bp.RenderDataSize;
  st.DataSize = bp.DataSize;

  auto& context = gloco->get_context();

  st.m_vao.open(context);
  st.m_vbo.open(context, fan::opengl::GL_ARRAY_BUFFER);
  st.m_vao.bind(context);
  st.m_vbo.bind(context);
  st.shader = bp.shader;
  st.locations = bp.locations;
  st.render_data_size = 0;
  uint64_t ptr_offset = 0;
  for (const auto& location : st.locations) {
    context.opengl.glEnableVertexAttribArray(location.index);
    context.opengl.glVertexAttribPointer(location.index, location.size, location.type, fan::opengl::GL_FALSE, location.stride, (void*)ptr_offset);
    context.opengl.glVertexAttribDivisor(location.index, 1);
    switch (location.type) {
    case fan::opengl::GL_FLOAT: {
      ptr_offset += location.size * sizeof(f32_t);
      break;
    }
    case fan::opengl::GL_UNSIGNED_INT: {
      ptr_offset += location.size * sizeof(fan::opengl::GLuint);
      break;
    }
    default: {
      fan::throw_error_impl();
    }
    }
  }
}

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
IMGUI_API void ImGui::Image(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col) {
  ImGui::Image((ImTextureID)gloco->image_get(img), size, uv0, uv1, tint_col, border_col);
}

IMGUI_API bool ImGui::ImageButton(loco_t::image_t& img, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col) {
  return ImGui::ImageButton((ImTextureID)gloco->image_get(img), size, uv0, uv1, frame_padding, bg_col, tint_col);
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
    out["shape"] = ri.shape;
    out["blending"] = ri.blending;
    break;
  }
  default: {
    fan::throw_error("unimplemented shape");
  }
  }
  return false;
}

bool fan::graphics::json_to_shape(const nlohmann::json& in, loco_t::shape_t* shape) {
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
    p.shape = in["shape"];
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