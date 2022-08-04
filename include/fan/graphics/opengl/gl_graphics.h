#pragma once

#include _FAN_PATH(types/types.h)

#include _FAN_PATH(graphics/renderer.h)

#if fan_renderer == fan_renderer_opengl

#include _FAN_PATH(graphics/opengl/gl_core.h)
#include _FAN_PATH(graphics/opengl/gl_shader.h)

#include _FAN_PATH(graphics/shared_graphics.h)

#include _FAN_PATH(graphics/webp.h)

//#include _FAN_PATH(graphics/opengl/2D/objects/line.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/rectangle.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/circle.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/sprite.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/sprite1.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/sprite0.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/sprite1.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/yuv420p_renderer.h)
//

#include _FAN_PATH(font.h)
#include _FAN_PATH(graphics/opengl/font.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/letter_renderer.h)
//#include _FAN_PATH(graphics/opengl/2D/objects/text_renderer.h)

//#include _FAN_PATH(graphics/opengl/2D/effects/post_process.h)
//
//
//#include _FAN_PATH(graphics/opengl/2D/objects/depth/depth_rectangle.h)
//
//#include _FAN_PATH(graphics/opengl/2D/effects/particles.h)
//#include _FAN_PATH(graphics/opengl/2D/effects/flame.h)
//
//#include _FAN_PATH(graphics/opengl/3D/objects/model.h)
//#include _FAN_PATH(graphics/opengl/3D/objects/skybox.h)
//
//#include _FAN_PATH(graphics/shared_inline_graphics.h)

#endif