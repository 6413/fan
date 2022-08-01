#pragma once

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)

namespace fan_2d {
  namespace opengl {
    struct post_process_t {
      bool open(fan::opengl::context_t* context, const fan::opengl::core::renderbuffer_t::properties_t& p) {
        framebuffer.open(context);
        framebuffer.bind(context);

        fan::webp::image_info_t image_info;
        image_info.data = 0;
        image_info.size = p.size;
        fan::opengl::image_t::load_properties_t lp;
        lp.format = fan::opengl::GL_RGB;
        lp.internal_format = fan::opengl::GL_RGB;
        lp.filter = fan::opengl::GL_LINEAR;
        texture_colorbuffer.load(context, image_info);

        framebuffer.bind_to_texture(context, *texture_colorbuffer.get_texture(context));

        renderbuffer.open(context, p);
        framebuffer.bind_to_renderbuffer(context, renderbuffer.renderbuffer);

        bool ret = !framebuffer.ready(context);
        framebuffer.unbind(context);

        sprite.open(context);

        sprite.set_vertex(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/effects/post_process.vs)
        );
        sprite.set_fragment(
          context,
          #include _FAN_PATH(graphics/glsl/opengl/2D/effects/post_process.fs)
        );
        sprite.compile(context);

        fan::opengl::cid_t cid;
        sprite_t::properties_t sp;
        sp.position = 0;
        sp.image = texture_colorbuffer;
        sp.size = 1;
        assert(0);
        //sprite.push_back(context, &cid, sp);

        return ret;
      }
      void close(fan::opengl::context_t* context) {
        texture_colorbuffer.unload(context);
        framebuffer.close(context);
        renderbuffer.close(context);
      }

      void update_renderbuffer(fan::opengl::context_t* context, const fan::opengl::core::renderbuffer_t::properties_t& p) {
        renderbuffer.set_storage(context, p);
      }

      void start_capture(fan::opengl::context_t* context) {
        draw_nodereference = context->enable_draw(this, [](fan::opengl::context_t* context, void* d) { 
          post_process_t* post = (post_process_t*)d;
          post->framebuffer.bind(context);
          context->opengl.call(context->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT | fan::opengl::GL_DEPTH_BUFFER_BIT);
          context->set_depth_test(true);
          // probably want to glclear here if trash comes
        });
      }

      void draw(fan::opengl::context_t* context) {
        framebuffer.unbind(context); // not sure if necessary
        context->opengl.call(context->opengl.glClear, fan::opengl::GL_COLOR_BUFFER_BIT);
        context->set_depth_test(false);
        sprite.draw(context);
      }

      fan::opengl::core::renderbuffer_t renderbuffer;
      fan::opengl::core::framebuffer_t framebuffer;

      fan::opengl::image_t texture_colorbuffer;

      // for test purposes
      sprite_t sprite;

      uint32_t draw_nodereference;
    };
  }
}