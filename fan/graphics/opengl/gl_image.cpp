#include "gl_image.h"

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
#include <fan/graphics/loco.h>

fan::graphics::image_t::image_t() {
    texture_reference.NRI = -1;
}

fan::graphics::image_t::image_t(const fan::webp::image_info_t image_info) {
    load(image_info, load_properties_t());
}

fan::graphics::image_t::image_t(const fan::webp::image_info_t image_info, load_properties_t p) {
    load(image_info, p);
}

fan::graphics::image_t::image_t(const char* path) {
    load(path);
}

fan::graphics::image_t::image_t(const char* path, load_properties_t p) {
    load(path, p);
}

bool fan::graphics::image_t::is_invalid() const {
    return texture_reference.NRI == (decltype(texture_reference.NRI))-1;
}

void fan::graphics::image_t::create_texture() {
    auto& context = gloco->get_context();
    texture_reference = gloco->image_list.NewNode();
    gloco->image_list[texture_reference].image = this;
    context.opengl.call(context.opengl.glGenTextures, 1, &get_texture());
}

void fan::graphics::image_t::erase_texture() {
    auto& context = gloco->get_context();
    context.opengl.glDeleteTextures(1, &get_texture());
    gloco->image_list.Recycle(texture_reference);
    texture_reference.NRI = -1;
}

void fan::graphics::image_t::bind_texture() {
    auto& context = gloco->get_context();
    context.opengl.call(context.opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, get_texture());
}

void fan::graphics::image_t::unbind_texture() {
    auto& context = gloco->get_context();
    context.opengl.call(context.opengl.glBindTexture, fan::opengl::GL_TEXTURE_2D, 0);
}

fan::opengl::GLuint& fan::graphics::image_t::get_texture() {
    return gloco->image_list[texture_reference].texture_id;
}

bool fan::graphics::image_t::load(fan::webp::image_info_t image_info) {
    return load(image_info, load_properties_t());
}

bool fan::graphics::image_t::load(fan::webp::image_info_t image_info, load_properties_t p) {

    auto& context = gloco->get_context();

    create_texture();
    bind_texture();

    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);

    size = image_info.size;

    context.opengl.call(context.opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, image_info.data);

    switch (p.min_filter) {
        case fan::opengl::GL_LINEAR_MIPMAP_LINEAR:
        case fan::opengl::GL_NEAREST_MIPMAP_LINEAR:
        case fan::opengl::GL_LINEAR_MIPMAP_NEAREST:
        case fan::opengl::GL_NEAREST_MIPMAP_NEAREST: {
            context.opengl.call(context.opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
            break;
        }
    }

    return 0;
}

// returns 0 on success

bool fan::graphics::image_t::load(const fan::string& path) {
    return load(path, load_properties_t());
}

bool fan::graphics::image_t::load(const fan::string& path, const load_properties_t& p) {

    #if fan_assert_if_same_path_loaded_multiple_times

    static std::unordered_map<fan::string, bool> existing_images;

    if (existing_images.find(path) != existing_images.end()) {
        fan::throw_error("image already existing " + path);
    }

    existing_images[path] = 0;

    #endif

    fan::webp::image_info_t image_info;
    if (fan::webp::load(path, &image_info)) {
        return true;
    }
    bool ret = load(image_info, p);
    fan::webp::free_image(image_info.data);
    return ret;
}

bool fan::graphics::image_t::load(fan::color* colors, const fan::vec2ui& size_) {
    return load(colors, size_, load_properties_t());
}

bool fan::graphics::image_t::load(fan::color* colors, const fan::vec2ui& size_, load_properties_t p) {

    auto& context = gloco->get_context();

    create_texture();
    bind_texture();

    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);

    size = size_;

    context.opengl.call(context.opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, fan::opengl::GL_RGBA32F, size.x, size.y, 0, p.format, fan::opengl::GL_FLOAT, (uint8_t*)colors);

    return 0;
}

void fan::graphics::image_t::reload_pixels(const fan::webp::image_info_t& image_info) {
    reload_pixels(image_info, load_properties_t());
}

void fan::graphics::image_t::reload_pixels(const fan::webp::image_info_t& image_info, const load_properties_t& p) {

    auto& context = gloco->get_context();

    bind_texture();

    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);

    size = image_info.size;
    context.opengl.call(context.opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, image_info.data);

}

void fan::graphics::image_t::unload() {
    erase_texture();
}

void fan::graphics::image_t::create(const fan::color& color, const fan::vec2& size_) {
    create(color, size_, load_properties_t());
}

// creates single colored text size.x*size.y sized

void fan::graphics::image_t::create(const fan::color& color, const fan::vec2& size_, load_properties_t p) {
    auto& context = gloco->get_context();


    size = size_;

    uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (size.x * size.y * fan::color::size()));
    for (int y = 0; y < size_.y; y++) {
        for (int x = 0; x < size_.x; x++) {
            for (int p = 0; p < fan::color::size(); p++) {
                *pixels = color[p] * 255;
                pixels++;
            }
        }
    }

    pixels -= (int)size.x * (int)size.y * fan::color::size();

    create_texture();
    bind_texture();

    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);

    context.opengl.call(context.opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, size.x, size.y, 0, p.format, p.type, pixels);

    free(pixels);

    context.opengl.call(context.opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
}

void fan::graphics::image_t::create_missing_texture() {
    create_missing_texture(load_properties_t());
}

void fan::graphics::image_t::create_missing_texture(load_properties_t p) {
    auto& context = gloco->get_context();

    uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (2 * 2 * fan::color::size()));
    uint32_t pixel = 0;

    pixels[pixel++] = 0;
    pixels[pixel++] = 0;
    pixels[pixel++] = 0;
    pixels[pixel++] = 255;

    pixels[pixel++] = 255;
    pixels[pixel++] = 0;
    pixels[pixel++] = 220;
    pixels[pixel++] = 255;

    pixels[pixel++] = 255;
    pixels[pixel++] = 0;
    pixels[pixel++] = 220;
    pixels[pixel++] = 255;

    pixels[pixel++] = 0;
    pixels[pixel++] = 0;
    pixels[pixel++] = 0;
    pixels[pixel++] = 255;

    p.visual_output = fan::opengl::GL_REPEAT;

    create_texture();
    bind_texture();

    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);

    size = fan::vec2i(2, 2);

    context.opengl.call(context.opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, 2, 2, 0, p.format, p.type, pixels);

    free(pixels);

    context.opengl.call(context.opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
}

void fan::graphics::image_t::create_transparent_texture() {
    load_properties_t p;
    auto& context = gloco->get_context();

    uint8_t* pixels = (uint8_t*)malloc(sizeof(uint8_t) * (2 * 2 * fan::color::size()));
    uint32_t pixel = 0;

    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 255;

    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 255;

    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 40;
    pixels[pixel++] = 255;

    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 60;
    pixels[pixel++] = 255;

    p.visual_output = fan::opengl::GL_REPEAT;

    create_texture();
    bind_texture();

    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_S, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_WRAP_T, p.visual_output);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MIN_FILTER, p.min_filter);
    context.opengl.call(context.opengl.glTexParameteri, fan::opengl::GL_TEXTURE_2D, fan::opengl::GL_TEXTURE_MAG_FILTER, p.mag_filter);

    size = fan::vec2i(2, 2);

    context.opengl.call(context.opengl.glTexImage2D, fan::opengl::GL_TEXTURE_2D, 0, p.internal_format, 2, 2, 0, p.format, p.type, pixels);

    free(pixels);

    context.opengl.call(context.opengl.glGenerateMipmap, fan::opengl::GL_TEXTURE_2D);
}

fan::vec4_wrap_t<fan::vec2> fan::graphics::image_t::calculate_aspect_ratio(const fan::vec2& size, f32_t scale) {

    fan::vec4_wrap_t<fan::vec2> tc = {
        fan::vec2(0, 1),
        fan::vec2(1, 1),
        fan::vec2(1, 0),
        fan::vec2(0, 0)
    };

    f32_t a = size.x / size.y;
    fan::vec2 n = size.normalize();

    for (uint32_t i = 0; i < 8; i++) {
        if (size.x < size.y) {
            tc[i % 4][i / 4] *= n[i / 4] / a * scale;
        }
        else {
            tc[i % 4][i / 4] *= n[i / 4] * a * scale;
        }
    }
    return tc;
}

void fan::graphics::image_t::get_pixel_data(void* data, fan::opengl::GLenum format) {
    auto& context = gloco->get_context();

    bind_texture();

    context.opengl.call(
        context.opengl.glGetTexImage,
        fan::opengl::GL_TEXTURE_2D,
        0,
        format,
        fan::opengl::GL_UNSIGNED_BYTE,
        data
    );
}

// slow

std::unique_ptr<uint8_t[]> fan::graphics::image_t::get_pixel_data(fan::opengl::GLenum format, fan::vec2 uvp, fan::vec2 uvs) {
    auto& context = gloco->get_context();

    bind_texture();

    fan::vec2ui uv_size = {
        (uint32_t)(size.x * uvs.x),
        (uint32_t)(size.y * uvs.y)
    };

    auto full_ptr = std::make_unique<uint8_t[]>(size.x * size.y * 4); // assuming rgba

    context.opengl.call(
        context.opengl.glGetTexImage,
        fan::opengl::GL_TEXTURE_2D,
        0,
        format,
        fan::opengl::GL_UNSIGNED_BYTE,
        full_ptr.get()
    );

    auto ptr = std::make_unique<uint8_t[]>(uv_size.x * uv_size.y * 4); // assuming rgba

    for (uint32_t y = 0; y < uv_size.y; ++y) {
        for (uint32_t x = 0; x < uv_size.x; ++x) {
            uint32_t full_index = ((y + uvp.y * size.y) * size.x + (x + uvp.x * size.x)) * 4;
            uint32_t index = (y * uv_size.x + x) * 4;
            ptr[index + 0] = full_ptr[full_index + 0];
            ptr[index + 1] = full_ptr[full_index + 1];
            ptr[index + 2] = full_ptr[full_index + 2];
            ptr[index + 3] = full_ptr[full_index + 3];
        }
    }

    return ptr;
}

fan::graphics::gl_image_impl::image_list_NodeReference_t::image_list_NodeReference_t(fan::graphics::image_t* image) {
  NRI = image->texture_reference.NRI;
}