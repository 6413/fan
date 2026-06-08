#include <vulkan/vulkan.h>
import fan;
using namespace fan::graphics;
import fan.graphics.vulkan.ray_tracing.hardware_renderer;

int main() {
  engine_t engine{{
    .window_size ={2560, 1440},
    .renderer = fan::window_t::renderer_t::vulkan,
  }};
  engine.set_vsync(0);
  engine.set_target_fps(0);
  engine.clear_color = 0.3;
  engine.set_culling_enabled(false);

  auto& vk = fan::graphics::get_vk_context();
  fan::vec2ui window_size = fan::vec2(2560, 1440);

  fan::graphics::vulkan::ray_tracing::context_t rt;
  bool rt_opened = false;
  rt.open(vk, window_size);
  rt_opened = true;

  fan::graphics::image_nr_t rt_image = rt.accum_image;


  fan::graphics::sprite_t s{{
      .size = fan::vec2(window_size) / 2.f
    }};
  s.set_image(rt_image);

  bool pending = false;
  fan::vec2ui pending_size = window_size;
  bool rt_ready = true;
  bool rt_image_valid = true;

  auto resize_cb = engine.window.add_resize_callback([&](const auto& d) {
    pending_size = d.size;
    pending = true;
  });

  auto camera_handle = engine.perspective_render_view.camera;
  auto& camera = engine.camera_get(camera_handle);
  int cursor_mode = 0;

  auto cb = engine.window.add_mouse_motion_callback([&](const auto& d) {
    if (cursor_mode == 1) {
      camera.rotate_camera(d.motion);
    }
  });

  vk.begin_cmd_cb.push_back([&](VkCommandBuffer cmd) {
    if (!rt_ready) {
      return;
    }
    rt.record_trace_rays(cmd);
  });

  bool update_camera = true;
  engine.loop([&] {
    rt.on_camera_updated(update_camera);

    if (engine.is_key_pressed(fan::key_r)) {
      vkDeviceWaitIdle(vk.device);

      // destroy old pipeline + SBT
      vkDestroyPipeline(vk.device, rt.pipeline, nullptr);
      vkDestroyBuffer(vk.device, rt.shader_binding_table, nullptr);
      vkFreeMemory(vk.device, rt.sbt_memory, nullptr);

      // recreate
      rt.create_pipeline();
      rt.create_sbt();

      fan::print("Ray tracing shaders reloaded.");
    }

    rt.update_exposure(engine.delta_time);

    if (pending) {
      pending = false;
      rt_ready = false;

      vkDeviceWaitIdle(vk.device);


      if (rt_opened) {
        engine.image_erase(rt_image);

        rt.close();
        rt_opened = false;
      }

      rt.open(vk, pending_size);
      rt_opened = true;


      rt_image = rt.accum_image;
      rt_image_valid = true;

      s.set_image(rt_image);
      //s.set_size(fan::vec2(pending_size) / 2.f);

      rt_ready = true;
    }

    s.set_size(engine.window.get_size() / 2.f);

    if (!pending && rt_ready) {
      //rt.update_tlas(engine.start_time.seconds());
    }

    cursor_mode = engine.is_mouse_down(fan::mouse_right);
    engine.window.set_cursor(!cursor_mode);
    engine.camera_move(camera, engine.delta_time, 10.f);
    engine.camera_set_perspective(camera_handle, 90.f, engine.window.get_size());
    if (rt_ready) {
      rt.update_camera_from_engine();
    }


    {
      fan::graphics::gui::begin("light");

      fan::graphics::gui::checkbox("update camera", &update_camera);

      static fan::vec3 light_pos = fan::vec3(5.0f, 10.0f, 5.0f);
      static fan::vec3 light_color = fan::vec3(1.0f, 1.0f, 1.0f);
      static float light_intensity = 3.0f;

      if (fan::graphics::gui::drag("Light Position", &light_pos)) {
        // updated
      }
      fan::graphics::gui::drag("Light Color", &light_color);
      fan::graphics::gui::drag("Light Intensity", &light_intensity);

      fan::graphics::vulkan::ray_tracing::light_ubo_t ubo{};
      ubo.position = light_pos;
      ubo.color = light_color;
      ubo.intensity = light_intensity;

      void* data;
      vkMapMemory(engine.context.vk.device, rt.light_memory, 0, sizeof(ubo), 0, &data);
      memcpy(data, &ubo, sizeof(ubo));
      vkUnmapMemory(engine.context.vk.device, rt.light_memory);
      fan::graphics::gui::end();
    }

  });

  vkDeviceWaitIdle(vk.device);

  if (rt_image_valid) {
    engine.image_erase(rt_image);
    rt_image_valid = false;
  }

  if (rt_opened) {
    rt.close();
    rt_opened = false;
  }
}
