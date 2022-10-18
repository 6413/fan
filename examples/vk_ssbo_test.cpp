// Creates window, opengl context

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_vulkan

#define loco_window
#define loco_context
#include _FAN_PATH(graphics/loco.h)

struct pile_t {
  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  void open() {

    loco.open(loco_t::properties_t());
    fan::graphics::open_matrices(
      loco.get_context(),
      &matrices,
      ortho_x,
      ortho_y
    );
    //loco.get_window()->add_resize_callback([&](fan::window_t* window, const fan::vec2i& size) {
    //  fan::vec2 window_size = window->get_size();
    //  fan::vec2 ratio = window_size / window_size.max();
    //  std::swap(ratio.x, ratio.y);
    //  matrices.set_ortho(
    //    ortho_x * ratio.x,
    //    ortho_y * ratio.y
    //  );
    //  viewport.set(loco.get_context(), 0, size, size);
    //});*/
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, loco.get_window()->get_size(), loco.get_window()->get_size());
  }

  loco_t loco;
  fan::graphics::matrices_t matrices;
  fan::graphics::viewport_t viewport;
};

int main() {
  pile_t* pile = new pile_t;
  pile->open();

  auto context = pile->loco.get_context();

  context->set_vsync(pile->loco.get_window(), 0);

  fan::vulkan::shader_t shader;

  shader.open(pile->loco.get_context());
  shader.set_vertex(pile->loco.get_context(), "include/fan/graphics/glsl/vulkan/2D/objects/ssbo_test.vert.spv");
  shader.set_fragment(pile->loco.get_context(), "include/fan/graphics/glsl/vulkan/2D/objects/ssbo_test.frag.spv");

  struct test_struct_t {
    fan::mat4 m;
  };

  fan::vulkan::core::ssbo_t< test_struct_t> ssbo;
  ssbo.open(context);
  test_struct_t st;
  st.m = fan::mat4(1);
  st.m[0].x = 0.5;

 // ssbo.push_ram_instance(context, st);
  ssbo.common.edit(context, &pile->loco.m_write_queue, 0, ssbo.size() + sizeof(test_struct_t));

  fan::vulkan::descriptor_set_layout_t<1> dsl_properties;

  dsl_properties.write_descriptor_sets[0].binding = 0;
  dsl_properties.write_descriptor_sets[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsl_properties.write_descriptor_sets[0].flags = VK_SHADER_STAGE_VERTEX_BIT;
  dsl_properties.write_descriptor_sets[0].block_common = &ssbo.common;
  dsl_properties.write_descriptor_sets[0].range = VK_WHOLE_SIZE;

  auto descriptor_layout_nr = context->descriptor_sets.push_layout(pile->loco.get_context(), dsl_properties);
  auto desc_nr = context->descriptor_sets.push(context, descriptor_layout_nr, dsl_properties);

  fan::vulkan::pipelines_t::properties_t p;
  p.descriptor_set_layout = &context->descriptor_sets.get(descriptor_layout_nr).layout;
  p.shader = &shader;
  auto pipeline_nr = context->pipelines.push(pile->loco.get_context(), p);

  pile->loco.draw_queue = [&] {
    pile->viewport.set(pile->loco.get_context(), 0, pile->loco.get_window()->get_size(), pile->loco.get_window()->get_size());

    context->draw(6, 1, 0, pipeline_nr, &context->descriptor_sets.descriptor_list[desc_nr].descriptor_set[context->currentFrame]);
  };

  pile->loco.loop([&] {
    pile->loco.get_window()->get_fps();
  });

  return 0;
}