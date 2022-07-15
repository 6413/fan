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
        #include _FAN_PATH(graphics/glsl/opengl/2D/objects/sprite.fs)
        );
        sprite_t::compile(context);
        store = {};
        sprite_t::set_draw_cb(context, sprite_t::draw_cb, &store);
        store.light_map = light_map;
      }
      struct store_t{
        fan::opengl::image_t light_map;
      }store;

      static void draw_cb(fan::opengl::context_t* context, inherit_t* sprite, void* userptr) {
        sprite1_t::store_t& store = *(sprite1_t::store_t*)userptr;
        sprite->m_shader.set_int(context, "texture_light_map", 1);
        context->opengl.glActiveTexture(fan::opengl::GL_TEXTURE1);
        context->opengl.glBindTexture(fan::opengl::GL_TEXTURE_2D, store.light_map.texture);
      }
    };
  }
}