#include <fan/pch.h>
#include <fan/graphics/webp.h>

// TODO make controller demo with player animations
// move player with gamepad joystick with different actions

void joystick_callback(int jid, int event)
{
  if (event == GLFW_CONNECTED)
  {
    fan::print("connected");
    // The joystick was connected
  }
  else if (event == GLFW_DISCONNECTED)
  {
    fan::print("disconnected");
    // The joystick was disconnected
  }
}

bool LoadTextureFromFile(const char* filename, fan::opengl::GLuint* out_texture, fan::vec2* image_size) {
  using namespace fan::opengl;
  fan::webp::image_info_t ii;
  if (fan::webp::load(filename, &ii)) {
    return true;
  }

  auto& opengl = gloco->get_context().opengl;
  // Create a OpenGL texture identifier
  fan::opengl::GLuint image_texture;
  opengl.glGenTextures(1, &image_texture);
  opengl.glBindTexture(GL_TEXTURE_2D, image_texture);

  // Setup filtering parameters for display
  opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
  opengl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

  opengl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ii.size.x, ii.size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, ii.data);

  *out_texture = image_texture;
  *image_size = ii.size;

  return false;
}

int main() {
  loco_t loco;
  glfwSetJoystickCallback(joystick_callback);

  loco.input_action.add({ fan::key_space, fan::key_w, fan::gamepad_a }, "jump");

  loco.get_context().set_vsync(loco.window, false);
  std::string str = fan::random::string(32);

  fan::opengl::GLuint texture;
  fan::vec2 image_size;
  LoadTextureFromFile("images/controller.webp", &texture, &image_size);

  fan::opengl::GLuint texture2;
  {
    fan::vec2 image_size;
    LoadTextureFromFile("images/tatti.webp", &texture2, &image_size);
  }

  fan::opengl::GLuint texture3;
  {
    fan::vec2 image_size;
    LoadTextureFromFile("images/tatti.webp", &texture3, &image_size);
  }

  fan::opengl::GLuint texture4;
  {
    fan::vec2 image_size;
    LoadTextureFromFile("images/l2r2.webp", &texture4, &image_size);
  }

  fan::opengl::GLuint texture5;
  {
    fan::vec2 image_size;
    LoadTextureFromFile("images/button.webp", &texture5, &image_size);
  }

  fan::opengl::GLuint texture6;
  {
    fan::vec2 image_size;
    LoadTextureFromFile("images/bumper.webp", &texture6, &image_size);
  }

  ImGuiStyle& style = ImGui::GetStyle();

  style.Colors[ImGuiCol_WindowBg] = fan::colors::white;

  loco.loop([&] {

    if (loco.input_action.is_active("jump") == loco.input_action_t::press) {
      fan::print("jump");
    }

    ImGui::Begin("test");

    {
      ImGui::Image((void*)texture, fan::vec2(512, 512));
    }
    {
      fan::vec2 v = loco.window.get_gamepad_axis(fan::gamepad_left_thumb);
      fan::color c = fan::color(1) - v.length();
      c.a = 1;
      ImGui::SetCursorPos(fan::vec2(146, 168) + v * 10);
      ImGui::Image((void*)texture2, fan::vec2(64, 64), fan::vec2(0), fan::vec2(1), c);
    }
    {
      fan::vec2 v = loco.window.get_gamepad_axis(fan::gamepad_right_thumb);
      ImGui::SetCursorPos(fan::vec2(312, 245) + v * 10);
      fan::color c = fan::color(1) - v.length();
      c.a = 1;
      ImGui::Image((void*)texture3, fan::vec2(64, 64), fan::vec2(0), fan::vec2(1), c);
    }
    {
      fan::vec2 v = loco.window.get_gamepad_axis(fan::gamepad_l2);
      ImGui::SetCursorPos(fan::vec2(149, 27));
      fan::color c = fan::color(1) - ((v.x + 1) / 2);
      c.a = 1;
      ImGui::Image((void*)texture4, fan::vec2(64, 64), fan::vec2(0), fan::vec2(1), c);
    }
    {
      fan::vec2 v = loco.window.get_gamepad_axis(fan::gamepad_r2);
      ImGui::SetCursorPos(fan::vec2(335, 27));
      fan::color c = fan::color(1) - ((v.x + 1) / 2);
      c.a = 1;
      ImGui::Image((void*)texture4, fan::vec2(64, 64), fan::vec2(0), fan::vec2(1), c);
    }
    //
    {
      int state = loco.window.key_state(fan::gamepad_up);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(fan::vec2(197, 229));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_down);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(fan::vec2(197, 229 + 48));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_right);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(fan::vec2(197 + 48, 229 + 48 / 2));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_left);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(fan::vec2(197 - 48, 229 + 48 / 2));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    static fan::vec2 offset = fan::vec2(135, -90);

    {
      int state = loco.window.key_state(fan::gamepad_y);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(offset + fan::vec2(197, 229));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_a);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(offset + fan::vec2(197, 229 + 48));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_b);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(offset + fan::vec2(197 + 48, 229 + 48 / 2));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_x);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(offset + fan::vec2(197 - 48, 229 + 48 / 2));
      ImGui::Image((void*)texture5, fan::vec2(48, 48), fan::vec2(0), fan::vec2(1), c);
    }

    {
      int state = loco.window.key_state(fan::gamepad_left_bumper);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(fan::vec2(148, 77));
      ImGui::Image((void*)texture6, fan::vec2(64, 64), fan::vec2(0), fan::vec2(1), c);
    }
    {
      int state = loco.window.key_state(fan::gamepad_right_bumper);
      fan::color c = 1;
      if (state == (int)fan::keyboard_state::press || state == (int)fan::keyboard_state::repeat) {
        c = 0;
        c.a = 1;
      }
      ImGui::SetCursorPos(fan::vec2(335, 77));
      ImGui::Image((void*)texture6, fan::vec2(64, 64), fan::vec2(0), fan::vec2(1), c);
    }
    ImGui::End();
  });
}