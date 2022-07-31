#pragma once

#include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)

namespace fan_2d {
  namespace opengl {

    struct sprite1_t : fan_2d::opengl::sprite_t{

      using inherit_t = fan_2d::opengl::sprite_t;

      void open(fan::opengl::context_t* context, fan::opengl::image_t light_map) {
        sprite_t::open(context);
        set_vertex(
          context,
            #include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.vs)
        );
        set_fragment(
          context,
          R"(
          #version 330
          in vec2 texture_coordinate;
          in vec4 instance_color;
          out vec4 o_color;
          uniform sampler2D texture_sampler;
          uniform sampler2D texture_light_map;
          uniform vec2 viewport_size;
          void main() {
            vec2 flipped_y = vec2(texture_coordinate.x, texture_coordinate.y);
            vec4 texture_color = texture(texture_sampler, flipped_y);
            vec2 p = gl_FragCoord.xy / viewport_size;
            vec4 light = texture(texture_light_map, vec2(p.x, 1.0 - p.y));
            o_color = texture_color * vec4(light.xyz, 1) * instance_color;
          }
          )"
        );
        sprite_t::compile(context);
        store = {};
        sprite_t::set_draw_cb(context, sprite1_t::draw_cb, &store);
        store.light_map = light_map;
      }
      struct store_t{
        fan::opengl::image_t light_map;
      }store;

      static void draw_cb(fan::opengl::context_t* context, inherit_t* sprite, void* userptr) {
        sprite1_t::store_t& store = *(sprite1_t::store_t*)userptr;
        sprite->m_shader.set_int(context, "texture_light_map", 1);
        sprite->m_shader.set_vec2(context, "viewport_size", context->get_viewport_size());
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, *store.light_map.get_texture(context));
      }
    };
  }
}