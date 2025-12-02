module;

#if defined(fan_opengl)
  #include <fan/graphics/opengl/init.h>
#endif

#include <source_location>
#include <cmath>
#include <string>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#define shaper_get_key_safe(return_type, kps_type, variable) \
	[key_pack] ()-> auto& { \
		auto o = g_shapes->shaper.GetKeyOffset( \
			offsetof(fan::graphics::kps_t::CONCAT(_, kps_type), variable), \
			offsetof(fan::graphics::kps_t::kps_type, variable) \
		);\
		static_assert(std::is_same_v<decltype(fan::graphics::kps_t::kps_type::variable), fan::graphics::return_type>, "possibly unwanted behaviour"); \
		return *(fan::graphics::return_type*)&key_pack[o];\
	}()

#define shape_get_vi(shape) (*(fan::graphics::shapes::shape##_t::vi_t*)GetRenderData(fan::graphics::g_shapes->shaper))
#define shape_get_ri(shape) (*(fan::graphics::shapes::shape##_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))

module fan.graphics.shapes;

std::uint8_t* A_resize(void* ptr, std::uintptr_t size) {
	if (ptr) {
		if (size) {
			void* rptr = (void*)__generic_realloc(ptr, size);
			if (rptr == 0) {
				fan::throw_error_impl();
			}
			return (std::uint8_t*)rptr;
		}
		else {
			__generic_free(ptr);
			return 0;
		}
	}
	else {
		if (size) {
			void* rptr = (void*)__generic_malloc(size);
			if (rptr == 0) {
				fan::throw_error_impl();
			}
			return (std::uint8_t*)rptr;
		}
		else {
			return 0;
		}
	}
}

namespace fan::graphics {
	context_shader_t::context_shader_t() {}
	context_shader_t::~context_shader_t() {}
	context_image_t::context_image_t() {}
	context_image_t::~context_image_t() {}
	context_t::context_t() {}
	context_t::~context_t() {}


  // warning does deep copy, addresses can die
	fan::graphics::context_shader_t shader_get(fan::graphics::shader_nr_t nr) {
		fan::graphics::context_shader_t context_shader;
		if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {
			context_shader.gl = *(fan::opengl::context_t::shader_t*)fan::graphics::g_render_context_handle->shader_get(fan::graphics::g_render_context_handle, nr);
		}
	#if defined(fan_vulkan)
		else if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::vulkan) {
			context_shader.vk = *(fan::vulkan::context_t::shader_t*)fan::graphics::g_render_context_handle->shader_get(fan::graphics::g_render_context_handle, nr);
		}
	#endif
		return context_shader;
	}


  #if defined(fan_json)
	fan::json image_to_json(const fan::graphics::image_t& image) {
		fan::json image_json;
		if (image.iic()) {
			return image_json;
		}

		auto shape_data = (*fan::graphics::g_render_context_handle.image_list)[image];
		if (shape_data.image_path.size()) {
			image_json["image_path"] = shape_data.image_path;
		}
		else {
			return image_json;
		}

		auto lp = fan::graphics::g_render_context_handle->image_get_settings(fan::graphics::g_render_context_handle, image);
		fan::graphics::image_load_properties_t defaults;
		if (lp.visual_output != defaults.visual_output) {
			image_json["image_visual_output"] = lp.visual_output;
		}
		if (lp.format != defaults.format) {
			image_json["image_format"] = lp.format;
		}
		if (lp.type != defaults.type) {
			image_json["image_type"] = lp.type;
		}
		if (lp.min_filter != defaults.min_filter) {
			image_json["image_min_filter"] = lp.min_filter;
		}
		if (lp.mag_filter != defaults.mag_filter) {
			image_json["image_mag_filter"] = lp.mag_filter;
		}

		return image_json;
	}

	fan::graphics::image_t json_to_image(const fan::json& image_json, const std::source_location& callers_path) {
		if (!image_json.contains("image_path")) {
			return fan::graphics::g_render_context_handle.default_texture;
		}

		std::string path = image_json["image_path"];
    std::string relative_path = fan::io::file::find_relative_path(
      path, 
      callers_path
    ).generic_string();
		if (!fan::io::file::exists(relative_path))
    {
			return fan::graphics::g_render_context_handle.default_texture;
		}
    path = std::filesystem::absolute(relative_path).generic_string();

		fan::graphics::image_load_properties_t lp;

		if (image_json.contains("image_visual_output")) {
			lp.visual_output = image_json["image_visual_output"];
		}
		if (image_json.contains("image_format")) {
			lp.format = image_json["image_format"];
		}
		if (image_json.contains("image_type")) {
			lp.type = image_json["image_type"];
		}
		if (image_json.contains("image_min_filter")) {
			lp.min_filter = image_json["image_min_filter"];
		}
		if (image_json.contains("image_mag_filter")) {
			lp.mag_filter = image_json["image_mag_filter"];
		}

		fan::graphics::image_nr_t image = fan::graphics::g_render_context_handle->image_load_path_props(
			fan::graphics::g_render_context_handle,
			path,
			lp,
			callers_path
		);
		(*fan::graphics::g_render_context_handle.image_list)[image].image_path = path;
		return image;
	}
#endif


#if defined(fan_json)
  sprite_sheet_animation_t::image_t::operator fan::json() const {
    fan::json j;
    image_t defaults;
    if (hframes != defaults.hframes) {
      j["hframes"] = hframes;
    }
    if (hframes != defaults.vframes) {
      j["vframes"] = vframes;
    }
    j.update(fan::graphics::image_to_json(image), true);
    return j;
  }

  sprite_sheet_animation_t::image_t& sprite_sheet_animation_t::image_t::assign(const fan::json& j, const std::source_location& callers_path) {
    image = fan::graphics::json_to_image(j, callers_path);
    if (j.contains("hframes")) {
      hframes = j.at("hframes");
    }
    if (j.contains("vframes")) {
      vframes = j.at("vframes");
    }
    return *this;
  }
#endif

  sprite_sheet_animation_t::sprite_sheet_animation_t() {
  }

  sprite_sheet_animation_t::~sprite_sheet_animation_t() {
  }

  animation_nr_t::animation_nr_t() = default;

  animation_nr_t::animation_nr_t(uint32_t id) {
    this->id = id;
  }

  animation_nr_t::operator uint32_t() const {
    return id;
  }

  animation_nr_t::operator bool() const {
    return id != (decltype(id))-1;
  }

  animation_nr_t animation_nr_t::operator++(int) {
    animation_nr_t temp(*this);
    ++id;
    return temp;
  }

  bool animation_nr_t::operator==(const animation_nr_t& other) const {
    return id == other.id;
  }

  bool animation_nr_t::operator!=(const animation_nr_t& other) const {
    return id != other.id;
  }

  size_t animation_nr_hash_t::operator()(const animation_nr_t& anim_nr) const noexcept {
    return std::hash<uint32_t>()(anim_nr.id);
  }

  std::size_t animation_pair_hash_t::operator()(const std::pair<animation_nr_t, std::string>& p) const noexcept {
    std::size_t h1 = animation_nr_hash_t{}(p.first);
    std::size_t h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 << 1);
  }

  std::unordered_map<animation_nr_t, sprite_sheet_animation_t, animation_nr_hash_t> all_animations;
  animation_nr_t all_animations_counter = 0;
  std::unordered_map<std::pair<animation_shape_nr_t, std::string>, animation_nr_t, animation_pair_hash_t> shape_animation_lookup_table;
  std::unordered_map<animation_shape_nr_t, std::vector<animation_nr_t>, animation_nr_hash_t> shape_animations;
  animation_nr_t shape_animation_counter = 0;

  sprite_sheet_animation_t& get_sprite_sheet_animation(animation_nr_t nr) {
    auto found_anim = fan::graphics::all_animations.find(nr);
    if (found_anim == fan::graphics::all_animations.end()) {
      fan::throw_error("Animation not found");
    }
    return found_anim->second;
  }

  sprite_sheet_animation_t& get_sprite_sheet_animation(animation_nr_t shape_animation_id, const std::string& anim_name) {
    auto found = shape_animation_lookup_table.find(std::make_pair(shape_animation_id, anim_name));
    if (found == shape_animation_lookup_table.end()) {
      fan::throw_error("Failed to find sprite sheet animation:" + anim_name);
    }
    return get_sprite_sheet_animation(found->second);
  }

  std::vector<fan::graphics::animation_nr_t>& get_sprite_sheet_shape_animation(animation_nr_t shape_animation_id) {
    auto found = shape_animations.find(shape_animation_id);
    if (found == shape_animations.end()) {
      fan::throw_error("Failed to find sprite sheet animation:" + std::to_string((uint32_t)shape_animation_id));
    }
    return found->second;
  }

  void rename_sprite_sheet_shape_animation(animation_nr_t shape_animation_id, const std::string& old_name, const std::string& new_name) {
    auto& previous_anims = shape_animations[shape_animation_id];
    auto found = std::find_if(previous_anims.begin(), previous_anims.end(), [old_name](const animation_nr_t nr) {
      auto found = fan::graphics::all_animations.find(nr);
      if (found == fan::graphics::all_animations.end()) {
        fan::throw_error("Animation nr expired (bug)");
      }
      return found->second.name == old_name;
      });
    if (found == previous_anims.end()) {
      fan::throw_error("Animation:" + old_name, ", not found");
    }
    animation_nr_t previous_anim_nr = *found;
    auto prev_found = fan::graphics::all_animations.find(previous_anim_nr);
    if (prev_found == fan::graphics::all_animations.end()) {
      fan::throw_error("Animation nr expired (bug)");
    }
    auto& previous_anim = prev_found->second;
    {
      auto found = fan::graphics::shape_animation_lookup_table.find(std::make_pair(shape_animation_id, previous_anim.name));
      if (found != fan::graphics::shape_animation_lookup_table.end()) {
        fan::graphics::shape_animation_lookup_table.erase(found);
      }
    }
    previous_anim.name = new_name;
    fan::graphics::shape_animation_lookup_table[std::make_pair(shape_animation_id, new_name)] = previous_anim_nr;
  }

  animation_nr_t add_sprite_sheet_shape_animation(animation_nr_t new_anim) {
    shape_animations[shape_animation_counter].emplace_back(new_anim);
    return shape_animation_counter++;
  }

  animation_nr_t add_existing_sprite_sheet_shape_animation(animation_nr_t existing_anim, animation_nr_t shape_animation_id, const sprite_sheet_animation_t& new_anim) {
    animation_nr_t new_anim_nr = existing_anim;
    all_animations[existing_anim] = new_anim;
    if (!shape_animation_id) {
      shape_animation_id = add_sprite_sheet_shape_animation(new_anim_nr);
      shape_animation_lookup_table[std::make_pair(shape_animation_id, new_anim.name)] = new_anim_nr;
      return shape_animation_id;
    }
    shape_animation_lookup_table[std::make_pair(shape_animation_id, new_anim.name)] = new_anim_nr;
    auto found = shape_animations.find(shape_animation_id);
    if (found == shape_animations.end()) {
      fan::throw_error("add_sprite_sheet_shape_animation:given shape animation id not found");
    }
    found->second.emplace_back(new_anim_nr);
    return shape_animation_id;
  }

  animation_nr_t add_sprite_sheet_shape_animation(animation_nr_t shape_animation_id, const sprite_sheet_animation_t& new_anim) {
    animation_nr_t new_anim_nr = all_animations_counter++;
    return add_existing_sprite_sheet_shape_animation(new_anim_nr, shape_animation_id, new_anim);
  }

  bool is_animation_finished(animation_nr_t nr, const fan::graphics::sprite_sheet_data_t& sd) {
    auto& animation = fan::graphics::get_sprite_sheet_animation(nr);
    return sd.current_frame == animation.selected_frames.size() - 1;
  }

#if defined(fan_json)
  fan::json sprite_sheet_serialize() {
    fan::json result = fan::json::object();
    fan::json animations_arr = fan::json::array();

    for (auto& anim : all_animations) {
      fan::json ss;
      ss["name"] = anim.second.name;
      ss["selected_frames"] = anim.second.selected_frames;
      ss["fps"] = anim.second.fps;
      ss["id"] = anim.first.id;

      if (!anim.second.images.empty()) {
        fan::json images_arr = fan::json::array();
        for (const auto& img : anim.second.images) {
          images_arr.push_back(img);
        }
        ss["images"] = images_arr;
      }

      animations_arr.push_back(ss);
    }

    if (!animations_arr.empty()) {
      result["animations"] = animations_arr;
    }

    return result;
  }

  void sprite_sheet_deserialize(fan::json& json, const std::source_location& callers_path) {
    animation_nr_t counter_offset = all_animations_counter;

    if (json.contains("animations")) {
      for (const auto& item : json["animations"]) {
        sprite_sheet_animation_t anim;
        anim.name = item.value("name", std::string{});
        if (item.contains("selected_frames") && item["selected_frames"].is_array()) {
          anim.selected_frames.clear();
          for (const auto& frame_json : item["selected_frames"]) {
            anim.selected_frames.push_back(frame_json.get<int>());
          }
          if (item.contains("images")) {
            for (const auto& frame_json : item["images"]) {
              sprite_sheet_animation_t::image_t img;
              img.assign(frame_json, callers_path);
              anim.images.push_back(img);
            }
          }
        }
        else {
          anim.selected_frames.clear();
        }
        anim.fps = item.value("fps", 0.0f);

        animation_nr_t original_id = item.value("id", uint32_t());
        animation_nr_t new_id = original_id.id + counter_offset.id;
        all_animations[new_id] = anim;
        all_animations_counter = std::max(all_animations_counter.id, static_cast<uint32_t>(new_id.id + 1));
      }
    }

    // update animation id table
    if (json.contains("shapes")) {
      for (auto& shape : json["shapes"]) {
        // Update animation references in each shape
        if (shape.contains("animations")) {
          for (auto& anim_id : shape["animations"]) {
            animation_nr_t original_id = anim_id.get<uint32_t>();
            anim_id = original_id.id + counter_offset.id;
          }
        }
      }
    }
  }
#endif // fan_json


  std::string read_shader(const std::string& path, const std::source_location& callers_path) {
    std::string code;
    fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &code);
    return code;
  }
}

namespace fan::graphics{

  void shapes::shaper_deep_copy(shape_t* dst, const shape_t* const src, shaper_t::ShapeTypeIndex_t sti) {
    // alloc can be avoided inside switch
    uint8_t* KeyPack = new uint8_t[shaper.GetKeysSize(*src)];
    shaper.WriteKeys(*src, KeyPack);

    auto _vi = src->GetRenderData(shaper);
    auto vlen = shaper.GetRenderDataSize(sti);
    uint8_t* vi = new uint8_t[vlen];
    std::memcpy(vi, _vi, vlen);

    auto _ri = src->GetData(shaper);
    auto rlen = shaper.GetDataSize(sti);

    uint8_t* ri = new uint8_t[rlen];
    std::memcpy(ri, _ri, rlen);

    if (sti == shape_type_t::sprite) {
      if (((sprite_t::ri_t*)_ri)->sprite_sheet_data.frame_update_nr) {
        ((sprite_t::ri_t*)ri)->sprite_sheet_data.frame_update_nr = fan::graphics::g_render_context_handle.update_callback->NewNodeLast(); // since hard copy, let it leak
      }
    }

    *dst = fan::graphics::g_shapes->shaper.add(
      sti,
      KeyPack,
      fan::graphics::g_shapes->shaper.GetKeysSize(*src),
      vi,
      ri
    );

  #if defined(debug_shape_t)
    fan::print("+", NRI);
  #endif

    delete[] KeyPack;
    delete[] vi;
    delete[] ri;
  }



	shapes::shape_t::shape_t() {
		sic();
	}

	shapes::shape_t::shape_t(shaper_t::ShapeID_t&& s) {
		NRI = s.NRI;

		if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
			auto& ri = *(sprite_t::ri_t*)s.GetData(g_shapes->shaper);
			if (ri.sprite_sheet_data.frame_update_nr) {
				(*fan::graphics::g_render_context_handle.update_callback)[ri.sprite_sheet_data.frame_update_nr] = [nr = NRI](void* ptr) -> void {
					fan::graphics::shapes::shape_t::sprite_sheet_frame_update_cb(fan::graphics::g_shapes->shaper, (fan::graphics::shapes::shape_t*)&nr);
				};
			}
		}

		s.sic();
	}

	shapes::shape_t::shape_t(const shaper_t::ShapeID_t& s) : shape_t() {

		if (s.iic()) {
			return;
		}

		auto sti = g_shapes->shaper.ShapeList[s].sti;
		{
			fan::graphics::g_shapes->shaper_deep_copy(this, (const fan::graphics::shapes::shape_t*)&s, sti);
		}
		if (sti == shape_type_t::polygon) {
			polygon_t::ri_t* src_data = (polygon_t::ri_t*)s.GetData(fan::graphics::g_shapes->shaper);
			polygon_t::ri_t* dst_data = (polygon_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
			if (fan::graphics::g_render_context_handle.get_renderer() == fan::window_t::renderer_t::opengl) {
				auto& gl_ctx = (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle)));
				dst_data->vao.open(gl_ctx);
				dst_data->vbo.open(gl_ctx, src_data->vbo.m_target);

				auto& shape_data = g_shapes->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
				fan::graphics::context_shader_t shader;
				if (!shape_data.shader.iic()) {
					shader = fan::graphics::shader_get(shape_data.shader);
				}
				dst_data->vao.bind(gl_ctx);
				dst_data->vbo.bind(gl_ctx);
				uint64_t ptr_offset = 0;
				for (shape_gl_init_t& location : g_shapes->polygon.locations) {
					if ((gl_ctx.opengl.major == 2 && gl_ctx.opengl.minor == 1) && !shape_data.shader.iic()) {
						location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
					}
					fan_opengl_call(glEnableVertexAttribArray(location.index.first));
					switch (location.type) {
					case GL_UNSIGNED_INT:
					case GL_INT: {
						fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
						break;
					}
					default: {
						fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
					}
					}
					// instancing
					if ((gl_ctx.opengl.major > 3) || (gl_ctx.opengl.major == 3 && gl_ctx.opengl.minor >= 3)) {
						if (shape_data.instanced) {
							fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
						}
					}
					switch (location.type) {
					case GL_FLOAT: {
						ptr_offset += location.size * sizeof(GLfloat);
						break;
					}
					case GL_UNSIGNED_INT: {
						ptr_offset += location.size * sizeof(GLuint);
						break;
					}
					default: {
						fan::throw_error_impl();
					}
					}
				}
				fan::opengl::core::write_glbuffer(gl_ctx, dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
				glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
				glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
				glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
				polygon_vertex_t* ri = new polygon_vertex_t[dst_data->buffer_size / sizeof(polygon_vertex_t)];
				polygon_vertex_t* ri2 = new polygon_vertex_t[dst_data->buffer_size / sizeof(polygon_vertex_t)];
				fan::opengl::core::get_glbuffer(gl_ctx, ri, dst_data->vbo.m_buffer, dst_data->buffer_size, 0, dst_data->vbo.m_target);
				fan::opengl::core::get_glbuffer(gl_ctx, ri2, src_data->vbo.m_buffer, src_data->buffer_size, 0, src_data->vbo.m_target);
				delete[] ri;
			}
			else {
				fan::throw_error_impl();
			}
		}
	}

	shapes::shape_t::shape_t(shape_t&& s) : shape_t(std::move(*dynamic_cast<shaper_t::ShapeID_t*>(&s))) {

	}

	shapes::shape_t::shape_t(const fan::graphics::shapes::shape_t& s) : shape_t(*dynamic_cast<const shaper_t::ShapeID_t*>(&s)) {
		//NRI = s.NRI;
	}

	fan::graphics::shapes::shape_t& shapes::shape_t::operator=(const fan::graphics::shapes::shape_t& s) {
		if (iic() == false) {
			remove();
		}
		if (s.iic()) {
			return *this;
		}
		if (this != &s) {
			auto sti = g_shapes->shaper.ShapeList[s].sti;
			{
				g_shapes->shaper_deep_copy(this, (const fan::graphics::shapes::shape_t*)&s, sti);
			}
			if (sti == shape_type_t::polygon) {
				polygon_t::ri_t* src_data = (polygon_t::ri_t*)s.GetData(fan::graphics::g_shapes->shaper);
				polygon_t::ri_t* dst_data = (polygon_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
				if (fan::graphics::g_render_context_handle.get_renderer() == fan::window_t::renderer_t::opengl) {
					dst_data->vao.open((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
					dst_data->vbo.open((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))), src_data->vbo.m_target);

					auto& shape_data = g_shapes->shaper.GetShapeTypes(shape_type_t::polygon).renderer.gl;
					fan::graphics::context_shader_t shader;
					if (!shape_data.shader.iic()) {
						shader = fan::graphics::shader_get(shape_data.shader);
					}
					dst_data->vao.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
					dst_data->vbo.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
					uint64_t ptr_offset = 0;
					for (shape_gl_init_t& location : g_shapes->polygon.locations) {
						if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major == 2 && (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.minor == 1) && !shape_data.shader.iic()) {
							location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
						}
						fan_opengl_call(glEnableVertexAttribArray(location.index.first));
						switch (location.type) {
						case GL_UNSIGNED_INT:
						case GL_INT: {
							fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
							break;
						}
						default: {
							fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
						}
						}
						// instancing
						if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major > 3) || ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major == 3 && (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.minor >= 3)) {
							if (shape_data.instanced) {
								fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
							}
						}
						switch (location.type) {
						case GL_FLOAT: {
							ptr_offset += location.size * sizeof(GLfloat);
							break;
						}
						case GL_UNSIGNED_INT: {
							ptr_offset += location.size * sizeof(GLuint);
							break;
						}
						default: {
							fan::throw_error_impl();
						}
						}
						fan::opengl::core::write_glbuffer((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))), dst_data->vbo.m_buffer, 0, dst_data->buffer_size, dst_data->vbo.m_usage, dst_data->vbo.m_target);
						glBindBuffer(GL_COPY_READ_BUFFER, src_data->vbo.m_buffer);
						glBindBuffer(GL_COPY_WRITE_BUFFER, dst_data->vbo.m_buffer);
						glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, dst_data->buffer_size);
					}
				}
				else {
					fan::throw_error_impl();
				}
			}
			else if (sti == fan::graphics::shapes::shape_type_t::sprite) {
				// handle sprite sheet specific updates
				sprite_t::ri_t* ri = (sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
				sprite_t::ri_t* _ri = (sprite_t::ri_t*)s.GetData(fan::graphics::g_shapes->shaper);
				//if (((sprite_t::ri_t*)_ri)->sprite_sheet_data.frame_update_nr) {
				ri->sprite_sheet_data.frame_update_nr = fan::graphics::g_render_context_handle.update_callback->NewNodeLast(); // since hard copy, let it leak
				(*fan::graphics::g_render_context_handle.update_callback)[ri->sprite_sheet_data.frame_update_nr] = [nr = NRI](void* ptr) {
					fan::graphics::shapes::shape_t::sprite_sheet_frame_update_cb(g_shapes->shaper, (fan::graphics::shapes::shape_t*)&nr);
					};
				// }
			}
			//fan::print("i dont know what to do");
			//NRI = s.NRI;
		}
		return *this;
	}

	fan::graphics::shapes::shape_t& shapes::shape_t::operator=(fan::graphics::shapes::shape_t&& s) {
		if (NRI == s.NRI) {
			s.sic();
			return *this;
		}
		if (iic() == false) {
			remove();
		}
		if (s.iic()) {
			return *this;
		}

		if (this != &s) {
			if (s.iic() == false) {

			}
			NRI = s.NRI;

			if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
				auto& ri = *(sprite_t::ri_t*)s.GetData(fan::graphics::g_shapes->shaper);
				if (ri.sprite_sheet_data.frame_update_nr) {
					(*fan::graphics::g_render_context_handle.update_callback)[ri.sprite_sheet_data.frame_update_nr] = [nr = NRI](void* ptr) {
						fan::graphics::shapes::shape_t::sprite_sheet_frame_update_cb(g_shapes->shaper, (fan::graphics::shapes::shape_t*)&nr);
						};
				}
			}

			s.sic();
		}
		return *this;
	}

	shapes::shape_t::~shape_t() {
		remove();
	}

	shapes::shape_t::operator bool() const {
		return !iic();
	}

	bool shapes::shape_t::operator==(const shape_t& shape) const {
		return NRI == shape.NRI;
	}

	void shapes::shape_t::remove() {
		if (iic()) {
			return;
		}
	#if defined(debug_shape_t)
		fan::print("-", NRI);
	#endif
		if (g_shapes->shaper.ShapeList.Usage() == 0) {
			return;
		}
		auto shape_type = get_shape_type();
		if (shape_type == fan::graphics::shapes::shape_type_t::vfi) {
			g_shapes->vfi.erase(*this);
			sic();
			return;
		}
		if (shape_type == fan::graphics::shapes::shape_type_t::polygon) {
			auto ri = (polygon_t::ri_t*)GetData(g_shapes->shaper);
			ri->vbo.close((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
			ri->vao.close((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
		}
		else if (shape_type == fan::graphics::shapes::shape_type_t::sprite) {
			auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
			if (ri.sprite_sheet_data.frame_update_nr) {
				fan::graphics::g_render_context_handle.update_callback->unlrec(ri.sprite_sheet_data.frame_update_nr);
				ri.sprite_sheet_data.frame_update_nr.sic();
			}
		}
		g_shapes->shaper.remove(*this);
		sic();
	}

	void shapes::shape_t::erase() {
		remove();
	}

	// many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t
	uint16_t shapes::shape_t::get_shape_type() const {
		return g_shapes->shaper.ShapeList[*this].sti;
	}

	void shapes::shape_t::set_position(const fan::vec2& position) {
		g_shapes->shape_functions[get_shape_type()].set_position2(this, position);
	}

	void shapes::shape_t::set_position(const fan::vec3& position) {
		g_shapes->shape_functions[get_shape_type()].set_position3(this, position);
	}

	void shapes::shape_t::set_x(f32_t x) {
		set_position(fan::vec2(x, get_position().y));
	}

	void shapes::shape_t::set_y(f32_t y) {
		set_position(fan::vec2(get_position().x, y));
	}

	void shapes::shape_t::set_z(f32_t z) {
		set_position(fan::vec3(get_position().x, get_position().y, z));
	}

	fan::vec3 shapes::shape_t::get_position() const {
		auto shape_type = get_shape_type();
		return g_shapes->shape_functions[shape_type].get_position(this);
	}

	f32_t shapes::shape_t::get_x() const {
		return get_position().x;
	}

	f32_t shapes::shape_t::get_y() const {
		return get_position().y;
	}

	f32_t shapes::shape_t::get_z() const {
		return get_position().z;
	}

	void shapes::shape_t::set_size(const fan::vec2& size) {
		g_shapes->shape_functions[get_shape_type()].set_size(this, size);
	}

	void shapes::shape_t::set_size3(const fan::vec3& size) {
		g_shapes->shape_functions[get_shape_type()].set_size3(this, size);
	}

	// returns half extents of draw
	fan::vec2 shapes::shape_t::get_size() const {
		return fan::graphics::g_shapes->shape_functions[get_shape_type()].get_size(this);
	}

	fan::vec3 shapes::shape_t::get_size3() {
		return fan::graphics::g_shapes->shape_functions[get_shape_type()].get_size3(this);
	}

	void shapes::shape_t::set_rotation_point(const fan::vec2& rotation_point) {
		fan::graphics::g_shapes->shape_functions[get_shape_type()].set_rotation_point(this, rotation_point);
	}

	fan::vec2 shapes::shape_t::get_rotation_point() const {
		return fan::graphics::g_shapes->shape_functions[get_shape_type()].get_rotation_point(this);
	}

	void shapes::shape_t::set_color(const fan::color& color) {
		g_shapes->shape_functions[get_shape_type()].set_color(this, color);
	}

	fan::color shapes::shape_t::get_color() {
		return g_shapes->shape_functions[get_shape_type()].get_color(this);
	}

	void shapes::shape_t::set_angle(const fan::vec3& angle) {
		g_shapes->shape_functions[get_shape_type()].set_angle(this, angle);
	}

	fan::vec3 shapes::shape_t::get_angle() const {
		return g_shapes->shape_functions[get_shape_type()].get_angle(this);
	}

	fan::basis shapes::shape_t::get_basis() const {
		auto zangle = get_angle().z;
		auto c = std::cos(zangle);
		auto s = std::sin(zangle);

		return fan::basis{
			.right = { c, s, 0 },
			.forward = { s, -c, 0 },
			.up = { 0, 0, 1 }
		};
	}

	fan::vec3 shapes::shape_t::get_forward() const {
		return get_basis().forward;
	}

	fan::vec3 shapes::shape_t::get_right() const {
		return get_basis().right;
	}

	fan::vec3 shapes::shape_t::get_up() const {
		return get_basis().up;
	}

	fan::mat3 shapes::shape_t::get_rotation_matrix() const {
		return get_basis();
	}

	fan::vec3 shapes::shape_t::transform(const fan::vec3& local) const {
		// sign conflict, when forward y is -1, then moving y by -1 would move it down when we want it up
		// so flip the y sign since coordinate system is +y down
		fan::vec3 flipped_y{ local.x, -local.y, local.z };
		return get_position() + get_basis() * flipped_y;
	}

	fan::mat4 shapes::shape_t::get_transform() const {
		fan::mat4 m = fan::mat4::identity();
		m = m.translate(get_position());
		m = m * get_rotation_matrix();
		m = m.scale(get_size());
		return m;
	}

#if defined(fan_physics)
	fan::physics::aabb_t shapes::shape_t::get_aabb() const {
		fan::vec2 pos = get_position();
		fan::vec2 he = get_size(); // half extents
		f32_t cs = std::cos(get_angle().z);
		f32_t sn = std::sin(get_angle().z);
		fan::vec2 pivot = get_rotation_point();
    static constexpr auto flt_max = std::numeric_limits<f32_t>::max();
		fan::vec2 minp(flt_max, flt_max), maxp(-flt_max, -flt_max);

		for (int i = -1; i <= 1; i += 2) {
			for (int j = -1; j <= 1; j += 2) {
				fan::vec2 r = { i * he.x - pivot.x, j * he.y - pivot.y };
				r = { r.x * cs - r.y * sn, r.x * sn + r.y * cs };
				r += pos + pivot;
				minp.x = std::min(minp.x, r.x);
				minp.y = std::min(minp.y, r.y);
				maxp.x = std::max(maxp.x, r.x);
				maxp.y = std::max(maxp.y, r.y);
			}
		}

		return { minp, maxp };
	}
#endif

	fan::vec2 shapes::shape_t::get_tc_position() const {
		return g_shapes->shape_functions[get_shape_type()].get_tc_position(this);
	}

	void shapes::shape_t::set_tc_position(const fan::vec2& tc_position) {
		auto st = get_shape_type();
		g_shapes->shape_functions[st].set_tc_position(this, tc_position);
	}

	fan::vec2 shapes::shape_t::get_tc_size() const {
		return g_shapes->shape_functions[get_shape_type()].get_tc_size(this);
	}

	void shapes::shape_t::set_tc_size(const fan::vec2& tc_size) {
		g_shapes->shape_functions[get_shape_type()].set_tc_size(this, tc_size);
	}

	bool shapes::shape_t::load_tp(fan::graphics::texture_pack::ti_t* ti) {
		auto st = get_shape_type();
		if (st == fan::graphics::shapes::shape_type_t::sprite ||
			st == fan::graphics::shapes::shape_type_t::unlit_sprite) {
			auto image = ti->image;
			set_image(image);
			set_tc_position(ti->position / image.get_size());
			set_tc_size(ti->size / image.get_size());
			if (st == fan::graphics::shapes::shape_type_t::sprite) {
				sprite_t::ri_t* ram_data = (sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
				ram_data->texture_pack_unique_id = ti->unique_id;
			}
			else if (st == fan::graphics::shapes::shape_type_t::unlit_sprite) {
				unlit_sprite_t::ri_t* ram_data = (unlit_sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
				ram_data->texture_pack_unique_id = ti->unique_id;
			}
			return false;
		}
		fan::throw_error("invalid function call for current shape:"_str + shape_names[st]);
		return true;
	}

	fan::graphics::texture_pack::ti_t shapes::shape_t::get_tp() {
		fan::graphics::texture_pack::ti_t ti;
    ti.unique_id = get_tp_unique();
		ti.image = get_image();
		ti.position = get_tc_position() * ti.image.get_size();
		ti.size = get_tc_size() * ti.image.get_size();
		return ti;
		//return g_shapes->shape_functions[g_shapes->shaper.GetSTI(*this)].get_tp(this);
	}

	bool shapes::shape_t::set_tp(fan::graphics::texture_pack::ti_t* ti) {
		return load_tp(ti);
	}

  fan::graphics::texture_pack::unique_t shapes::shape_t::get_tp_unique() const {
    return ((shapes::sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))->texture_pack_unique_id;
  }

	fan::graphics::camera_t shapes::shape_t::get_camera() const {
		return g_shapes->shape_functions[get_shape_type()].get_camera(this);
	}

	void shapes::shape_t::set_camera(fan::graphics::camera_t camera) {
		g_shapes->shape_functions[get_shape_type()].set_camera(this, camera);
	}

	fan::graphics::viewport_t shapes::shape_t::get_viewport() const {
		return g_shapes->shape_functions[get_shape_type()].get_viewport(this);
	}

	void shapes::shape_t::set_viewport(fan::graphics::viewport_t viewport) {
		g_shapes->shape_functions[get_shape_type()].set_viewport(this, viewport);
	}

	render_view_t shapes::shape_t::get_render_view() const {
		render_view_t r;
		r.camera = get_camera();
		r.viewport = get_viewport();
		return r;
	}

	void shapes::shape_t::set_render_view(const fan::graphics::render_view_t& render_view) {
		set_camera(render_view.camera);
		set_viewport(render_view.viewport);
	}

	fan::vec2 shapes::shape_t::get_grid_size() {
		return g_shapes->shape_functions[get_shape_type()].get_grid_size(this);
	}

	void shapes::shape_t::set_grid_size(const fan::vec2& grid_size) {
		g_shapes->shape_functions[get_shape_type()].set_grid_size(this, grid_size);
	}

	fan::graphics::image_t shapes::shape_t::get_image() const {
		if (g_shapes->shape_functions[get_shape_type()].get_image) {
			return g_shapes->shape_functions[get_shape_type()].get_image(this);
		}
		return fan::graphics::g_render_context_handle.default_texture;
	}

	void shapes::shape_t::set_image(fan::graphics::image_t image) {
		g_shapes->shape_functions[get_shape_type()].set_image(this, image);
	}

	fan::graphics::image_data_t& shapes::shape_t::get_image_data() {
		return g_shapes->shape_functions[get_shape_type()].get_image_data(this);
	}

	std::array<fan::graphics::image_t, 30> shapes::shape_t::get_images() const {
		auto shape_type = get_shape_type();
		if (shape_type == shape_type_t::sprite) {
			return ((sprite_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images;
		}
		else if (shape_type == shape_type_t::unlit_sprite) {
			return ((unlit_sprite_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images;
		}
		else if (shape_type == shape_type_t::universal_image_renderer) {
			std::array<fan::graphics::image_t, 30> ret;
			auto uni_images = ((universal_image_renderer_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images_rest;
			std::copy(uni_images.begin(), uni_images.end(), ret.begin());

			return ret;
		}
	#if fan_debug >= fan_debug_medium
		fan::throw_error("only for sprite and unlit_sprite");
	#endif
		return {};
	}

	void shapes::shape_t::set_images(const std::array<fan::graphics::image_t, 30>& images) {
		auto shape_type = get_shape_type();
		if (shape_type == shape_type_t::sprite) {
			((sprite_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images = images;
		}
		else if (shape_type == shape_type_t::unlit_sprite) {
			((unlit_sprite_t::ri_t*)ShapeID_t::GetData(fan::graphics::g_shapes->shaper))->images = images;
		}
	#if fan_debug >= fan_debug_medium
		else {
			fan::throw_error("only for sprite and unlit_sprite");
		}
	#endif
	}

	f32_t shapes::shape_t::get_parallax_factor() const {
		return g_shapes->shape_functions[get_shape_type()].get_parallax_factor(this);
	}

	void shapes::shape_t::set_parallax_factor(f32_t parallax_factor) {
		g_shapes->shape_functions[get_shape_type()].set_parallax_factor(this, parallax_factor);
	}

	uint32_t shapes::shape_t::get_flags() const {
		auto f = g_shapes->shape_functions[get_shape_type()].get_flags;
		if (f) {
			return f(this);
		}
		return 0;
	}

	void shapes::shape_t::set_flags(uint32_t flag) {
		auto st = get_shape_type();
		return g_shapes->shape_functions[st].set_flags(this, flag);
	}

	f32_t shapes::shape_t::get_radius() const {
		return g_shapes->shape_functions[get_shape_type()].get_radius(this);
	}

	fan::vec3 shapes::shape_t::get_src() const {
		return g_shapes->shape_functions[get_shape_type()].get_src(this);
	}

	fan::vec3 shapes::shape_t::get_dst() const {
		return g_shapes->shape_functions[get_shape_type()].get_dst(this);
	}

	f32_t shapes::shape_t::get_outline_size() const {
		return g_shapes->shape_functions[get_shape_type()].get_outline_size(this);
	}

	fan::color shapes::shape_t::get_outline_color() const {
		return g_shapes->shape_functions[get_shape_type()].get_outline_color(this);
	}

	void shapes::shape_t::set_outline_color(const fan::color& color) {
		return g_shapes->shape_functions[get_shape_type()].set_outline_color(this, color);
	}

	void shapes::shape_t::reload(uint8_t format, void** image_data, const fan::vec2& image_size) {
		auto& settings = fan::graphics::g_render_context_handle->image_get_settings(fan::graphics::g_render_context_handle, get_image());
		uint32_t filter = settings.min_filter;
		universal_image_renderer_t::ri_t& ri = *(universal_image_renderer_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
		uint8_t image_count_new = fan::graphics::get_channel_amount(format);
		if (format != ri.format) {
			auto sti = get_shape_type();
			uint8_t* key_pack = g_shapes->shaper.GetKeys(*this);
			fan::graphics::image_t vi_image = shaper_get_key_safe(image_t, texture_t, image);

			auto shader = g_shapes->shaper.GetShader(sti);
			fan::graphics::g_render_context_handle->shader_set_vertex(
				fan::graphics::g_render_context_handle,
				shader,
				read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
			);
			{
				std::string fs;
				switch (format) {
				case fan::graphics::image_format::yuv420p: {
					fs = read_shader("shaders/opengl/2D/objects/yuv420p.fs");
					break;
				}
				case fan::graphics::image_format::nv12: {
					fs = read_shader("shaders/opengl/2D/objects/nv12.fs");
					break;
				}
				default: {
					fan::throw_error("unimplemented format");
				}
				}
				fan::graphics::g_render_context_handle->shader_set_fragment(fan::graphics::g_render_context_handle, shader, fs);
				fan::graphics::g_render_context_handle->shader_compile(fan::graphics::g_render_context_handle, shader);
			}

			uint8_t image_count_old = fan::graphics::get_channel_amount(ri.format);
			if (image_count_new < image_count_old) {
				uint8_t textures_to_remove = image_count_old - image_count_new;
				if (vi_image.iic() || vi_image == fan::graphics::g_render_context_handle.default_texture) { // uninitialized
					textures_to_remove = 0;
				}
				for (int i = 0; i < textures_to_remove; ++i) {
					int index = image_count_old - i - 1; // not tested
					if (index == 0) {
						fan::graphics::g_render_context_handle->image_erase(fan::graphics::g_render_context_handle, vi_image);
								
						set_image(fan::graphics::g_render_context_handle.default_texture);
					}
					else {
						fan::graphics::g_render_context_handle->image_erase(fan::graphics::g_render_context_handle, ri.images_rest[index - 1]);
						ri.images_rest[index - 1] = fan::graphics::g_render_context_handle.default_texture;
					}
				}
			}
			else if (image_count_new > image_count_old) {
				fan::graphics::image_t images[4];
				for (uint32_t i = image_count_old; i < image_count_new; ++i) {
					images[i] = fan::graphics::g_render_context_handle->image_create(fan::graphics::g_render_context_handle);
				}
				set_image(images[0]);
				std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
			}
		}

		auto vi_image = get_image();

		for (uint32_t i = 0; i < image_count_new; ++i) {
			if (i == 0) {
				if (vi_image.iic() || vi_image == fan::graphics::g_render_context_handle.default_texture) {
					vi_image = fan::graphics::g_render_context_handle->image_create(fan::graphics::g_render_context_handle);
					set_image(vi_image);
				}
			}
			else {
				if (ri.images_rest[i - 1].iic() || ri.images_rest[i - 1] == fan::graphics::g_render_context_handle.default_texture) {
					ri.images_rest[i - 1] = fan::graphics::g_render_context_handle->image_create(fan::graphics::g_render_context_handle);
				}
			}
		}

		for (uint32_t i = 0; i < image_count_new; i++) {
			fan::image::info_t image_info;
			image_info.data = image_data[i];
			image_info.size = fan::graphics::get_image_sizes(format, image_size)[i];
			auto lp = fan::graphics::get_image_properties<image_load_properties_t>(format)[i];
			lp.min_filter = filter;
			if (filter == fan::graphics::image_filter::linear ||
				filter == fan::graphics::image_filter::nearest) {
				lp.mag_filter = filter;
			}
			else {
				lp.mag_filter = fan::graphics::image_filter::linear;
			}
			if (i == 0) {
						
				fan::graphics::g_render_context_handle->image_reload_image_info_props(fan::graphics::g_render_context_handle, 
					vi_image,
					image_info,
					lp
				);
			}
			else {
						
				fan::graphics::g_render_context_handle->image_reload_image_info_props(fan::graphics::g_render_context_handle, 
					ri.images_rest[i - 1],
					image_info,
					lp
				);
			}
		}
		ri.format = format;
	}

	void shapes::shape_t::reload(uint8_t format, const fan::vec2& image_size) {
				
		auto& settings = fan::graphics::g_render_context_handle->image_get_settings(fan::graphics::g_render_context_handle, get_image());
		void* data[4]{};
		reload(format, data, image_size);
	}

	// universal image specific
	void shapes::shape_t::reload(uint8_t format, fan::graphics::image_t images[4]) {
		universal_image_renderer_t::ri_t& ri = *(universal_image_renderer_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
		uint8_t image_count_new = fan::graphics::get_channel_amount(format);
		if (format != ri.format) {
			auto sti = g_shapes->shaper.ShapeList[*this].sti;
			uint8_t* key_pack = g_shapes->shaper.GetKeys(*this);
			fan::graphics::image_t vi_image = shaper_get_key_safe(image_t, texture_t, image);


			auto shader = g_shapes->shaper.GetShader(sti);
					
			fan::graphics::g_render_context_handle->shader_set_vertex(fan::graphics::g_render_context_handle, 
				shader,
				read_shader("shaders/opengl/2D/objects/pixel_format_renderer.vs")
			);
			{
				std::string fs;
				switch (format) {
				case fan::graphics::image_format::yuv420p: {
					fs = read_shader("shaders/opengl/2D/objects/yuv420p.fs");
					break;
				}
				case fan::graphics::image_format::nv12: {
					fs = read_shader("shaders/opengl/2D/objects/nv12.fs");
					break;
				}
				default: {
					fan::throw_error("unimplemented format");
				}
				}
				fan::graphics::g_render_context_handle->shader_set_fragment(fan::graphics::g_render_context_handle, shader, fs);
						
				fan::graphics::g_render_context_handle->shader_compile(fan::graphics::g_render_context_handle, shader);
			}
			set_image(images[0]);
			std::copy(&images[1], &images[0] + ri.images_rest.size(), ri.images_rest.data());
			ri.format = format;
		}
	}

	void shapes::shape_t::set_line(const fan::vec2& src, const fan::vec2& dst) {
		auto st = get_shape_type();
		if (st == fan::graphics::shapes::shape_type_t::line) {
			auto data = reinterpret_cast<line_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper));
			data->src = fan::vec3(src.x, src.y, 0);
			data->dst = fan::vec3(dst.x, dst.y, 0);
			if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {
				auto& data = g_shapes->shaper.ShapeList[*this];
				g_shapes->shaper.ElementIsPartiallyEdited(
					data.sti,
					data.blid,
					data.ElementIndex,
					fan::member_offset(&line_t::vi_t::src),
					sizeof(line_t::vi_t::src)
				);
				g_shapes->shaper.ElementIsPartiallyEdited(
					data.sti,
					data.blid,
					data.ElementIndex,
					fan::member_offset(&line_t::vi_t::dst),
					sizeof(line_t::vi_t::dst)
				);
			}
		}
	#if defined(fan_3D)
		if (st == fan::graphics::shapes::shape_type_t::line3d) {
			auto data = reinterpret_cast<line3d_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper));
			data->src = fan::vec3(src.x, src.y, 0);
			data->dst = fan::vec3(dst.x, dst.y, 0);
			if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {
				auto& data = g_shapes->shaper.ShapeList[*this];
				g_shapes->shaper.ElementIsPartiallyEdited(
					data.sti,
					data.blid,
					data.ElementIndex,
					fan::member_offset(&line3d_t::vi_t::src),
					sizeof(line3d_t::vi_t::src)
				);
				g_shapes->shaper.ElementIsPartiallyEdited(
					data.sti,
					data.blid,
					data.ElementIndex,
					fan::member_offset(&line3d_t::vi_t::dst),
					sizeof(line3d_t::vi_t::dst)
				);
			}
		}
	#endif
	}

	bool shapes::shape_t::is_mouse_inside() {
		switch (get_shape_type()) {
		case shape_type_t::rectangle: {
			return fan_2d::collision::rectangle::point_inside_no_rotation(
				get_mouse_position(get_camera(), get_viewport()),
				get_position(),
				get_size()
			);
		}
		default: {
			break;
		}
		}
    return false;
	}

#if defined(fan_physics)
	bool shapes::shape_t::intersects(const fan::graphics::shapes::shape_t& shape) const {
		switch (get_shape_type()) {
		case shape_type_t::capsule: // inaccurate
		case shape_type_t::shader_shape:
		case shape_type_t::unlit_sprite:
		case shape_type_t::sprite:
		case shape_type_t::rectangle: {
			fan::physics::aabb_t aabb = get_aabb();
			fan::physics::aabb_t aabb2 = shape.get_aabb();
			return fan_2d::collision::rectangle::check_collision(
				aabb.min + (aabb.max - aabb.min) / 2.f,
				(aabb.max - aabb.min) / 2.f,
				aabb2.min + (aabb2.max - aabb2.min) / 2.f,
				(aabb2.max - aabb2.min) / 2.f
			);
		}
		}
		fan::throw_error("todo");
		return true;
	}

	bool shapes::shape_t::collides(const fan::graphics::shapes::shape_t& shape) const {
		return intersects(shape);
	}

	bool shapes::shape_t::point_inside(const fan::vec2& point) const {
		switch (get_shape_type()) {
		case shape_type_t::capsule: // inaccurate
		case shape_type_t::shader_shape:
		case shape_type_t::unlit_sprite:
		case shape_type_t::sprite:
		case shape_type_t::rectangle: {
			fan::physics::aabb_t aabb = get_aabb();
			fan::vec2 size = aabb.max - aabb.min;
			return fan_2d::collision::rectangle::point_inside(
				aabb.min,
				fan::vec2(aabb.min.x + size.x, aabb.min.y),
				aabb.max,
				fan::vec2(aabb.min.x, aabb.min.y + size.y),
				point
			);
		}
		case shape_type_t::circle: {
			return fan_2d::collision::circle::point_inside(point, get_position(), get_radius());
		}
		}
		fan::throw_error("todo");
		return true;
	}

	bool shapes::shape_t::collides(const fan::vec2& point) const {
		return point_inside(point);
	}
#endif

  fan::vec2 shapes::shape_t::get_image_sign() const {
    return get_tc_size().sign();
  }

	void shapes::shape_t::add_existing_animation(animation_nr_t nr) {
		if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
			auto& ri = shape_get_ri(sprite);
			auto& animation = fan::graphics::get_sprite_sheet_animation(nr);
			ri.shape_animations = fan::graphics::add_existing_sprite_sheet_shape_animation(nr, ri.shape_animations, animation);
			ri.current_animation = fan::graphics::shape_animations[ri.shape_animations].back();
		}
		else {
			fan::throw_error("Unimplemented for this shape");
		}
	}

  bool shapes::shape_t::is_animation_finished() const {
    return is_animation_finished(get_current_animation_id());
  }
	bool shapes::shape_t::is_animation_finished(animation_nr_t nr) const {
		auto& animation = fan::graphics::get_sprite_sheet_animation(nr);
		auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
		fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
		return sheet_data.current_frame == animation.selected_frames.size() - 1;
	}
  void shapes::shape_t::set_animation_loop(animation_nr_t nr, bool flag) {
    auto& animation = fan::graphics::get_sprite_sheet_animation(nr);
    animation.loop = flag;
  }
  void shapes::shape_t::reset_current_sprite_sheet_animation_frame() {
    auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
    ri.sprite_sheet_data.current_frame = 0;
  }
	void shapes::shape_t::reset_current_sprite_sheet_animation() {
		auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
		ri.sprite_sheet_data.current_frame = 0;
		ri.sprite_sheet_data.update_timer.restart();
	}

	// sprite sheet - sprite specific
	void shapes::shape_t::set_sprite_sheet_next_frame(int advance) {
		if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
			auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
			auto found = fan::graphics::all_animations.find(ri.current_animation);
			if (found == fan::graphics::all_animations.end()) {
				fan::throw_error("current_animation not found");
			}

			auto& animation = found->second;
			fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
      sheet_data.current_frame += advance;

      if (animation.loop) {
			  if (sheet_data.current_frame >= animation.selected_frames.size()) {
  				sheet_data.current_frame = 0;
			  }
      }
      else {
        sheet_data.current_frame = std::min(sheet_data.current_frame, (int)animation.selected_frames.size() - 1);
      }
			int actual_frame = animation.selected_frames[sheet_data.current_frame];

			// Find which image this frame belongs to and the local frame within that image
			int image_index = 0;
			int local_frame = actual_frame;
			int frame_count = 0;

			for (int i = 0; i < animation.images.size(); ++i) {
				int frames_in_this_image = animation.images[i].hframes * animation.images[i].vframes;
				if (actual_frame < frame_count + frames_in_this_image) {
					image_index = i;
					local_frame = actual_frame - frame_count;
					break;
				}
				frame_count += frames_in_this_image;
			}

			auto& current_image = animation.images[image_index];
			set_image(current_image.image);
			sheet_data.update_timer.restart();

			fan::vec2 tc_size = fan::vec2(1.0 / current_image.hframes, 1.0 / current_image.vframes);
			int frame_x = local_frame % current_image.hframes;
			int frame_y = local_frame / current_image.hframes;
      fan::vec2 pos(
        frame_x * tc_size.x,
        frame_y * tc_size.y
      );
      fan::vec2 sign = get_image_sign();
      if (sign.x < 0) {
        pos.x += tc_size.x;
      }
      if (sign.y < 0) {
        pos.y += tc_size.y;
      }
			set_tc_position(pos);
			set_tc_size(tc_size * sign);
		}
		else {
			fan::throw_error("Unimplemented for this shape");
		}
	}

	animation_shape_nr_t shapes::shape_t::get_shape_animations_id() const {
		return ((sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))->shape_animations;
	}

	std::unordered_map<std::string, fan::graphics::animation_nr_t> shapes::shape_t::get_all_animations() const {
		std::unordered_map<std::string, fan::graphics::animation_nr_t> result;

		for (auto& animation_nrs : fan::graphics::shape_animations[get_shape_animations_id()]) {
			auto& anim = fan::graphics::all_animations[animation_nrs];
			result[anim.name] = animation_nrs;
		}
		return result;
	}

	// Takes in seconds
	void shapes::shape_t::set_sprite_sheet_fps(f32_t fps) {
		if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
			auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
			fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
			for (auto& animation_nrs : shape_animations[ri.shape_animations]) {
				::fan::graphics::get_sprite_sheet_animation(animation_nrs).fps = fps;
			}
			if (sheet_data.update_timer.m_time == (uint64_t)-1) {
				sheet_data.update_timer.start(1.0 / fps * 1e+9);
			}
			else {
				sheet_data.update_timer.set_time(1.0 / fps * 1e+9);
			}
		}
		else {
			fan::throw_error("Unimplemented for this shape");
		}
	}

	bool shapes::shape_t::has_animation() {
		if (get_shape_type() != fan::graphics::shapes::shape_type_t::sprite) {
			return false;
		}

		auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
		fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
		return sheet_data.update_timer.started();
	}

	void shapes::shape_t::set_sprite_sheet_frames(uint32_t image_index, int horizontal_frames, int vertical_frames) {
		if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
			auto& current_anim = get_sprite_sheet_animation();
			current_anim.images[image_index].hframes = horizontal_frames;
			current_anim.images[image_index].vframes = vertical_frames;
			start_sprite_sheet_animation();
		}
		else {
			fan::throw_error("Unimplemented for this shape");
		}
	}

	animation_nr_t& shapes::shape_t::get_current_animation_id() const {
		return shape_get_ri(sprite).current_animation;
	}

  bool shapes::shape_t::animation_on(const std::string& name, int frame_index) {
    auto cur_frame = get_current_animation_frame();
    return cur_frame == frame_index && name == get_current_animation().name;
  }

  bool shapes::shape_t::animation_on(const std::string& name, const std::initializer_list<int>& arr) {
    auto cur_frame = get_current_animation_frame();
    bool str_eq = name == get_current_animation().name;
    for (auto& i : arr) {
      if (cur_frame == i && str_eq) {
        return true;
      }
    }
    return false;
  }

	void shapes::shape_t::set_current_animation_id(animation_nr_t animation_id) {
	#if fan_debug >= fan_debug_medium
		if (!animation_id) {
			fan::throw_error("invalid animation id");
		}
	#endif
		get_current_animation_id() = animation_id;
	}

	sprite_sheet_animation_t& shapes::shape_t::get_current_animation() {
		auto found = all_animations.find(get_current_animation_id());
		#if fan_debug >= fan_debug_medium
		if (found == all_animations.end()) {
			fan::throw_error("animation not found");
		}
		#endif
		return found->second;
	}

	int shapes::shape_t::get_current_animation_frame() const {
		auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
		return ri.sprite_sheet_data.current_frame;
	}
  void shapes::shape_t::set_current_animation_frame(int frame_id) {
    auto& ri = *(sprite_t::ri_t*)GetData(fan::graphics::g_shapes->shaper);
    ri.sprite_sheet_data.current_frame = frame_id;
  }

	// dont store the pointer
	sprite_sheet_animation_t* shapes::shape_t::get_animation(const std::string& name) {
		auto anims = get_all_animations();
		for (const auto& anim : anims) {
			if (anim.first == name) {
				auto found = all_animations.find(anim.second);
			#if fan_debug >= fan_debug_medium
				if (found == all_animations.end()) {
					fan::throw_error("animation not found, animation list (corruption)");
				}
			#endif
				return &found->second;
			}
		}
		return nullptr;
	}

	void shapes::shape_t::set_light_position(const fan::vec3& new_pos) {
		if (get_shape_type() != fan::graphics::shapes::shape_type_t::shadow) {
			fan::throw_error("invalid function call for current shape");
		}
		reinterpret_cast<shadow_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper))->light_position = new_pos;
		if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {
			auto& data = g_shapes->shaper.ShapeList[*this];
			g_shapes->shaper.ElementIsPartiallyEdited(
				data.sti,
				data.blid,
				data.ElementIndex,
				fan::member_offset(&shadow_t::vi_t::light_position),
				sizeof(shadow_t::vi_t::light_position)
			);
		}
	}

	void shapes::shape_t::set_light_radius(f32_t radius) {
		if (get_shape_type() != fan::graphics::shapes::shape_type_t::shadow) {
			fan::throw_error("invalid function call for current shape");
		}

		reinterpret_cast<shadow_t::vi_t*>(GetRenderData(fan::graphics::g_shapes->shaper))->light_radius = radius;
		if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {
			auto& data = g_shapes->shaper.ShapeList[*this];
			g_shapes->shaper.ElementIsPartiallyEdited(
				data.sti,
				data.blid,
				data.ElementIndex,
				fan::member_offset(&shadow_t::vi_t::light_radius),
				sizeof(shadow_t::vi_t::light_radius)
			);
		}
	}

	// for line
	void shapes::shape_t::set_thickness(f32_t new_thickness) {
	#if fan_debug >= 3
		if (get_shape_type() != fan::graphics::shapes::shape_type_t::line) {
			fan::throw_error("Invalid function call 'set_thickness', shape was not line");
		}
	#endif
		((line_t::vi_t*)GetRenderData(fan::graphics::g_shapes->shaper))->thickness = new_thickness;
		auto& data = g_shapes->shaper.ShapeList[*this];
		g_shapes->shaper.ElementIsPartiallyEdited(
			data.sti,
			data.blid,
			data.ElementIndex,
			fan::member_offset(&line_t::vi_t::thickness),
			sizeof(line_t::vi_t::thickness)
		);
	}

	void shapes::shape_t::apply_floating_motion(f32_t time, f32_t amplitude, f32_t speed, f32_t phase) {
		fan::throw_error("time todo");
		f32_t y = std::sin(time * speed + phase) * amplitude;
		set_y(y);
	} // shape_t


  //shapes
  shapes::shape_t shapes::light_t::push_back(const properties_t& properties) {
    vi_t vi;
    vi.position = properties.position;
    vi.parallax_factor = properties.parallax_factor;
    vi.size = properties.size;
    vi.rotation_point = properties.rotation_point;
    vi.color = properties.color;
    vi.flags = properties.flags;
    vi.angle = properties.angle;
    ri_t ri;

    return shape_add(shape_type, vi, ri,
      Key_e::light, (uint8_t)0,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  shapes::shape_t shapes::line_t::push_back(const properties_t& properties) {
    vi_t vi;
    vi.src = properties.src;
    vi.dst = properties.dst;
    vi.color = properties.color;
    vi.thickness = properties.thickness;
    ri_t ri;

    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.src.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  shapes::shape_t shapes::rectangle_t::push_back(const properties_t& properties) {
    vi_t vi;
    vi.position = properties.position;
    vi.size = properties.size;
    vi.color = properties.color;
    vi.outline_color = properties.outline_color;
    vi.angle = properties.angle;
    vi.rotation_point = properties.rotation_point;
    ri_t ri;

    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }

  shapes::shape_t shapes::sprite_t::push_back(const properties_t& properties) {

    bool uses_texture_pack = properties.texture_pack_unique_id.iic() == false && g_shapes->texture_pack;
    fan::graphics::texture_pack::ti_t ti;
    if (uses_texture_pack) {
      uses_texture_pack = !g_shapes->texture_pack->qti((*g_shapes->texture_pack)[properties.texture_pack_unique_id].name, &ti);
      if (uses_texture_pack) {
        auto& img = image_get_data(g_shapes->texture_pack->get_pixel_data(properties.texture_pack_unique_id).image);
        ti.position /= img.size;
        ti.size /= img.size;
      }
    }

    vi_t vi;
    vi.position = properties.position;
    vi.size = properties.size;
    vi.rotation_point = properties.rotation_point;
    vi.color = properties.color;
    vi.angle = properties.angle;
    vi.flags = properties.flags;
    vi.tc_position = uses_texture_pack ? ti.position : properties.tc_position;
    vi.tc_size = uses_texture_pack ? ti.size : properties.tc_size;
    vi.parallax_factor = properties.parallax_factor;
    vi.seed = properties.seed;

    ri_t ri;
    ri.images = properties.images;
    ri.sprite_sheet_data.current_frame = 0;
    ri.shape_animations = properties.shape_animations;
    ri.current_animation = properties.current_animation;

    if (uses_texture_pack) {
      ri.texture_pack_unique_id = properties.texture_pack_unique_id;
    }

    if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {

      if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major > 3) || ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major == 3 && ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.minor >= 3))) {
        return shape_add(
          shape_type, vi, ri,
          Key_e::depth,
          static_cast<uint16_t>(properties.position.z),
          Key_e::blending, static_cast<uint8_t>(properties.blending),
          Key_e::image, uses_texture_pack ? ti.image : properties.image,
          Key_e::viewport, properties.viewport,
          Key_e::camera, properties.camera,
          Key_e::ShapeType, shape_type,
          Key_e::draw_mode, properties.draw_mode,
          Key_e::vertex_count, properties.vertex_count
        );
      }
      else {
        // Legacy version requires array of 6 identical vertices
        vi_t vertices[6];
        for (int i = 0; i < 6; i++) {
          vertices[i] = vi;
        }

        return shape_add(
          shape_type, vertices[0], ri, Key_e::depth,
          static_cast<uint16_t>(properties.position.z),
          Key_e::blending, static_cast<uint8_t>(properties.blending),
          Key_e::image, uses_texture_pack ? ti.image : properties.image, Key_e::viewport,
          properties.viewport, Key_e::camera, properties.camera,
          Key_e::ShapeType, shape_type,
          Key_e::draw_mode, properties.draw_mode,
          Key_e::vertex_count, properties.vertex_count
        );
      }
    }
    else if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::vulkan) {
      return shape_add(
        shape_type, vi, ri, Key_e::depth,
        static_cast<uint16_t>(properties.position.z),
        Key_e::blending, static_cast<uint8_t>(properties.blending),
        Key_e::image, uses_texture_pack ? ti.image : properties.image, Key_e::viewport,
        properties.viewport, Key_e::camera, properties.camera,
        Key_e::ShapeType, shape_type,
        Key_e::draw_mode, properties.draw_mode,
        Key_e::vertex_count, properties.vertex_count
      );
    }

    return {};
  }


  shapes::shape_t shapes::unlit_sprite_t::push_back(const properties_t& properties) {

    bool uses_texture_pack = properties.texture_pack_unique_id.iic() == false && g_shapes->texture_pack;
    fan::graphics::texture_pack::ti_t ti;
    if (uses_texture_pack) {
      uses_texture_pack = !g_shapes->texture_pack->qti((*g_shapes->texture_pack)[properties.texture_pack_unique_id].name, &ti);
      if (uses_texture_pack) {
        auto img_size = g_shapes->texture_pack->get_pixel_data(properties.texture_pack_unique_id).image.get_size();
        ti.position /= img_size;
        ti.size /= img_size;
      }
    }

    vi_t vi;
    vi.position = properties.position;
    vi.size = properties.size;
    vi.rotation_point = properties.rotation_point;
    vi.color = properties.color;
    vi.angle = properties.angle;
    vi.flags = properties.flags;
    vi.tc_position = uses_texture_pack ? ti.position : properties.tc_position;
    vi.tc_size = uses_texture_pack ? ti.size : properties.tc_size;
    vi.parallax_factor = properties.parallax_factor;
    vi.seed = properties.seed;
    ri_t ri;
    ri.images = properties.images;
    if (uses_texture_pack) {
      ri.texture_pack_unique_id = properties.texture_pack_unique_id;
    }

    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::image, uses_texture_pack ? ti.image : properties.image,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  shapes::shape_t shapes::text_t::push_back(const properties_t& properties) {
    return g_shapes->shaper.add(shape_type_t::text, nullptr, 0, nullptr, nullptr);
  }
  fan::graphics::shapes::shape_t shapes::circle_t::push_back(const circle_t::properties_t& properties) {
    circle_t::vi_t vi;
    vi.position = properties.position;
    vi.radius = properties.radius;
    vi.rotation_point = properties.rotation_point;
    vi.color = properties.color;
    vi.angle = properties.angle;
    vi.flags = properties.flags;
    circle_t::ri_t ri;
    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  fan::graphics::shapes::shape_t shapes::capsule_t::push_back(const capsule_t::properties_t& properties) {
    capsule_t::vi_t vi;
    vi.position = properties.position;
    vi.center0 = properties.center0;
    vi.center1 = properties.center1;
    vi.radius = properties.radius;
    vi.rotation_point = properties.rotation_point;
    vi.color = properties.color;
    vi.outline_color = properties.outline_color;
    vi.angle = properties.angle;
    vi.flags = properties.flags;
    capsule_t::ri_t ri;
    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  fan::graphics::shapes::shape_t shapes::polygon_t::push_back(const properties_t& properties) {
    if (properties.vertices.empty()) {
      fan::throw_error("invalid vertices");
    }

    std::vector<polygon_vertex_t> polygon_vertices(properties.vertices.size());
    for (std::size_t i = 0; i < properties.vertices.size(); ++i) {
      polygon_vertices[i].position = properties.vertices[i].position;
      polygon_vertices[i].color = properties.vertices[i].color;
      polygon_vertices[i].offset = properties.position;
      polygon_vertices[i].angle = properties.angle;
      polygon_vertices[i].rotation_point = properties.rotation_point;
    }

    vi_t vis;
    ri_t ri;
    ri.buffer_size = sizeof(decltype(polygon_vertices)::value_type) * polygon_vertices.size();
    ri.vao.open((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
    ri.vao.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
    ri.vbo.open((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))), GL_ARRAY_BUFFER);
    fan::opengl::core::write_glbuffer(
      (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))),
      ri.vbo.m_buffer,
      polygon_vertices.data(),
      ri.buffer_size,
      GL_STATIC_DRAW,
      ri.vbo.m_target
    );

    auto& shape_data = g_shapes->shaper.GetShapeTypes(shape_type).renderer.gl;

    fan::graphics::context_shader_t shader;
    if (!shape_data.shader.iic()) {
      shader = fan::graphics::shader_get(shape_data.shader);
    }
    uint64_t ptr_offset = 0;
    for (shape_gl_init_t& location : locations) {
      if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major == 2 && (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.minor == 1) && !shape_data.shader.iic()) {
        location.index.first = fan_opengl_call(glGetAttribLocation(shader.gl.id, location.index.second));
      }
      fan_opengl_call(glEnableVertexAttribArray(location.index.first));
      switch (location.type) {
      case GL_UNSIGNED_INT:
      case GL_INT: {
        fan_opengl_call(glVertexAttribIPointer(location.index.first, location.size, location.type, location.stride, (void*)ptr_offset));
        break;
      }
      default: {
        fan_opengl_call(glVertexAttribPointer(location.index.first, location.size, location.type, GL_FALSE, location.stride, (void*)ptr_offset));
      }
      }
      // instancing
      if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major > 3) || ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major == 3 && (*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.minor >= 3)) {
        if (shape_data.instanced) {
          fan_opengl_call(glVertexAttribDivisor(location.index.first, 1));
        }
      }
      switch (location.type) {
      case GL_FLOAT: {
        ptr_offset += location.size * sizeof(GLfloat);
        break;
      }
      case GL_UNSIGNED_INT: {
        ptr_offset += location.size * sizeof(GLuint);
        break;
      }
      default: {
        fan::throw_error_impl();
      }
      }
    }

    return shape_add(shape_type, vis, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, (uint32_t)properties.vertices.size()
    );
  }

  shapes::shape_t shapes::grid_t::push_back(const properties_t& properties) {
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
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }

  shapes::shape_t shapes::particles_t::push_back(const properties_t& properties) {
    //KeyPack.ShapeType = shape_type;
    vi_t vi;
    ri_t ri;
    ri.position = properties.position;
    ri.size = properties.size;
    ri.color = properties.color;

    ri.begin_time = fan::time::now();
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
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  shapes::shape_t shapes::universal_image_renderer_t::push_back(const properties_t& properties) {
    vi_t vi;
    vi.position = properties.position;
    vi.size = properties.size;
    vi.tc_position = properties.tc_position;
    vi.tc_size = properties.tc_size;
    ri_t ri;
    // + 1
    std::copy(&properties.images[1], &properties.images[0] + properties.images.size(), ri.images_rest.data());
    shapes::shape_t shape = shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::image, properties.images[0],
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
    ((ri_t*)shape.GetData(fan::graphics::g_shapes->shaper))->format = shape.get_image_data().image_settings.format;

    return shape;
  }
  shapes::shape_t shapes::gradient_t::push_back(const properties_t& properties) {
    kps_t::common_t KeyPack;
    KeyPack.ShapeType = shape_type;
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

    return shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  shapes::shape_t shapes::shadow_t::push_back(const properties_t& properties) {
    vi_t vi;
    vi.position = properties.position;
    vi.shape = properties.shape;
    vi.size = properties.size;
    vi.rotation_point = properties.rotation_point;
    vi.color = properties.color;
    vi.flags = properties.flags;
    vi.angle = properties.angle;
    vi.light_position = properties.light_position;
    vi.light_radius = properties.light_radius;
    ri_t ri;

    return shape_add(shape_type, vi, ri,
      Key_e::shadow, (uint8_t)0,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
  shapes::shape_t shapes::shader_shape_t::push_back(const properties_t& properties) {
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
    fan::graphics::shapes::shape_t ret = shape_add(shape_type, vi, ri,
      Key_e::depth, (uint16_t)properties.position.z,
      Key_e::blending, (uint8_t)properties.blending,
      Key_e::image, properties.image,
      Key_e::viewport, properties.viewport,
      Key_e::camera, properties.camera,
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
    g_shapes->shaper.GetShader(shape_type) = properties.shader;
    return ret;
  }
#if defined(fan_3D)
  shapes::shape_t shapes::rectangle3d_t::push_back(const properties_t& properties) {
    vi_t vi;
    vi.position = properties.position;
    vi.size = properties.size;
    vi.color = properties.color;
    //vi.angle = properties.angle;
    ri_t ri;

    if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::opengl) {
      if (((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major > 3) || ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.major == 3 && ((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))).opengl.minor >= 3))) {
        // might not need depth
        return shape_add(shape_type, vi, ri,
          Key_e::depth, (uint16_t)properties.position.z,
          Key_e::blending, (uint8_t)properties.blending,
          Key_e::viewport, properties.viewport,
          Key_e::camera, properties.camera,
          Key_e::ShapeType, shape_type,
          Key_e::draw_mode, properties.draw_mode,
          Key_e::vertex_count, properties.vertex_count
        );
      }
      else {
        vi_t vertices[36];
        for (int i = 0; i < 36; i++) {
          vertices[i] = vi;
        }

        return shape_add(shape_type, vertices[0], ri,
          Key_e::depth, (uint16_t)properties.position.z,
          Key_e::blending, (uint8_t)properties.blending,
          Key_e::viewport, properties.viewport,
          Key_e::camera, properties.camera,
          Key_e::ShapeType, shape_type,
          Key_e::draw_mode, properties.draw_mode,
          Key_e::vertex_count, properties.vertex_count
        );
      }
    }
    else if (fan::graphics::g_render_context_handle.window->renderer == fan::window_t::renderer_t::vulkan) {

    }
    fan::throw_error();
    return{};
  }
  shapes::shape_t shapes::line3d_t::push_back(const properties_t& properties) {
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
      Key_e::ShapeType, shape_type,
      Key_e::draw_mode, properties.draw_mode,
      Key_e::vertex_count, properties.vertex_count
    );
  }
#endif

} // namespace fan::graphics



void fan::graphics::shapes::shape_t::sprite_sheet_frame_update_cb(fan::graphics::shaper_t& shaper, fan::graphics::shapes::shape_t* shape) {
	auto& ri = *(sprite_t::ri_t*)shape->GetData(shaper);
	fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
	if (sheet_data.update_timer) { // is it possible to just remove frame_udpate_cb if its not valid
		if (ri.current_animation) {
			auto& selected_frames = all_animations[ri.current_animation].selected_frames;
			if (selected_frames.empty()) {
				return;
			}
			shape->set_sprite_sheet_next_frame();
		}
		else {
			shape->set_sprite_sheet_next_frame();
		}
		sheet_data.update_timer.restart();
	}
}

fan::graphics::sprite_sheet_animation_t& fan::graphics::shapes::shape_t::get_sprite_sheet_animation() {
	return ::fan::graphics::get_sprite_sheet_animation(shape_get_ri(sprite).current_animation);
}


void fan::graphics::shapes::shape_t::start_sprite_sheet_animation() {
	auto& ri = shape_get_ri(sprite);
	auto& current_anim = get_sprite_sheet_animation();

	fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;
	sheet_data.current_frame = 0;
	sheet_data.update_timer.start(1.0 / current_anim.fps * 1e+9);
	int actual_frame = current_anim.selected_frames[sheet_data.current_frame];
	if (current_anim.images.empty()) {
		return;
	}
	auto& current_image = current_anim.images[actual_frame];

	set_tc_position(fan::vec2(0, 0));
	set_tc_size(fan::vec2(1.0 / current_image.hframes, 1.0 / current_image.vframes));
	// No frames to process, remove frame update function
	//if (current_image.vframes * current_image.hframes == 1) {
	//  if (sheet_data.frame_update_nr) {
	//    fan::graphics::g_render_context_handle.update_callback->unlrec(sheet_data.frame_update_nr);
	//    sheet_data.frame_update_nr.sic();
	//  }
	//}
	//else {
	if (sheet_data.frame_update_nr == false) {
		sheet_data.frame_update_nr = fan::graphics::g_render_context_handle.update_callback->NewNodeLast();
	}
	(*fan::graphics::g_render_context_handle.update_callback)[sheet_data.frame_update_nr] = [nr = NRI](void* ptr) {
		sprite_sheet_frame_update_cb(g_shapes->shaper, (fan::graphics::shapes::shape_t*)&nr);
		};
	//}
}

void fan::graphics::shapes::shape_t::stop_sprite_sheet_animation() {
  auto& ri = shape_get_ri(sprite);
  auto& current_anim = get_sprite_sheet_animation();
  fan::graphics::sprite_sheet_data_t& sheet_data = ri.sprite_sheet_data;

  if (sheet_data.frame_update_nr) {
    fan::graphics::g_render_context_handle.update_callback->unlrec(sheet_data.frame_update_nr);
    sheet_data.frame_update_nr.sic();
  }
}

void fan::graphics::shapes::shape_t::set_sprite_sheet_animation(const fan::graphics::sprite_sheet_animation_t& animation) {
	if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
		auto& ri = shape_get_ri(sprite);
		auto& previous_anim = ::fan::graphics::get_sprite_sheet_animation(ri.current_animation);
		{
			auto found = shape_animation_lookup_table.find(std::make_pair(ri.shape_animations, previous_anim.name));
			if (found != shape_animation_lookup_table.end()) {
				shape_animation_lookup_table.erase(found);
			}
		}
		previous_anim = animation;
		shape_animation_lookup_table[std::make_pair(ri.shape_animations, animation.name)] = ri.current_animation;

		start_sprite_sheet_animation();
	}
	else {
		fan::throw_error("Unimplemented for this shape");
	}
}

void fan::graphics::shapes::shape_t::add_sprite_sheet_animation(const fan::graphics::sprite_sheet_animation_t& animation) {
	if (get_shape_type() == fan::graphics::shapes::shape_type_t::sprite) {
		auto& ri = shape_get_ri(sprite);
		// adds animation to 
		ri.shape_animations = add_sprite_sheet_shape_animation(ri.shape_animations, animation);
		ri.current_animation = shape_animations[ri.shape_animations].back();
		start_sprite_sheet_animation();
	}
	else {
		fan::throw_error("Unimplemented for this shape");
	}
}

#define shaper_set_ShapeTypeChange \
  __builtin_memcpy(new_renderdata, old_renderdata, element_count * g_shapes->shaper.GetRenderDataSize(sti)); \
  __builtin_memcpy(new_data, old_data, element_count * g_shapes->shaper.GetDataSize(sti));
void fan::graphics::shaper_t::_ShapeTypeChange(
  ShapeTypeIndex_t sti,
  KeyPackSize_t keypack_size,
  uint8_t* keypack,
  MaxElementPerBlock_t element_count,
  const void* old_renderdata,
  const void* old_data,
  void* new_renderdata,
  void* new_data
) {
  shaper_set_ShapeTypeChange
}

#if defined(fan_json)
namespace fan::graphics {
  bool shape_to_json(fan::graphics::shapes::shape_t& shape, fan::json* json) {
    fan::json& out = *json;
    switch (shape.get_shape_type()) {
    case fan::graphics::shapes::shape_type_t::light: {
      fan::graphics::shapes::light_t::properties_t defaults;
      out["shape"] = "light";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_parallax_factor() != defaults.parallax_factor) {
        out["parallax_factor"] = shape.get_parallax_factor();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_flags() != defaults.flags) {
        out["flags"] = shape.get_flags();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::line: {
      fan::graphics::shapes::line_t::properties_t defaults;
      out["shape"] = "line";
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_src() != defaults.src) {
        out["src"] = shape.get_src();
      }
      if (shape.get_dst() != defaults.dst) {
        out["dst"] = shape.get_dst();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::rectangle: {
      fan::graphics::shapes::rectangle_t::properties_t defaults;
      out["shape"] = "rectangle";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_outline_color() != defaults.outline_color) {
        out["outline_color"] = shape.get_outline_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::graphics::shapes::sprite_t::properties_t defaults;
      out["shape"] = "sprite";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_parallax_factor() != defaults.parallax_factor) {
        out["parallax_factor"] = shape.get_parallax_factor();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      if (shape.get_flags() != defaults.flags) {
        out["flags"] = shape.get_flags();
      }
      if (shape.get_tc_position() != defaults.tc_position) {
        out["tc_position"] = shape.get_tc_position();
      }
      if (shape.get_tc_size() != defaults.tc_size) {
        out["tc_size"] = shape.get_tc_size();
      }
      auto* ri = ((fan::graphics::shapes::sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper));
      if (*fan::graphics::g_shapes->texture_pack) {
        const auto& t = (*fan::graphics::g_shapes->texture_pack)[ri->texture_pack_unique_id];
        if (t.name.size()) {
          out["texture_pack_name"] = t.name;
        }
      }
      if (ri->shape_animations) {
        fan::json animation_array = fan::json::array();
        for (auto& animation_nrs : fan::graphics::shape_animations[ri->shape_animations]) {
          animation_array.push_back(animation_nrs.id);
        }
        if (animation_array.empty() == false) {
          out["animations"] = animation_array;
        }
      }
      fan::json images_array = fan::json::array();

      auto main_image = shape.get_image();
      auto img_json = fan::graphics::image_to_json(main_image);
      if (!img_json.empty()) {
        images_array.push_back(img_json);
      }

      auto images = shape.get_images();
      for (auto& image : images) {
        img_json = fan::graphics::image_to_json(image);
        if (!img_json.empty()) {
          images_array.push_back(img_json);
        }
      }

      if (!images_array.empty()) {
        out["images"] = images_array;
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      fan::graphics::shapes::unlit_sprite_t::properties_t defaults;
      out["shape"] = "unlit_sprite";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_parallax_factor() != defaults.parallax_factor) {
        out["parallax_factor"] = shape.get_parallax_factor();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      if (shape.get_flags() != defaults.flags) {
        out["flags"] = shape.get_flags();
      }
      if (shape.get_tc_position() != defaults.tc_position) {
        out["tc_position"] = shape.get_tc_position();
      }
      if (shape.get_tc_size() != defaults.tc_size) {
        out["tc_size"] = shape.get_tc_size();
      }
      if (*fan::graphics::g_shapes->texture_pack) {
        auto t = (*fan::graphics::g_shapes->texture_pack)[((fan::graphics::shapes::unlit_sprite_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper))->texture_pack_unique_id];
        if (t.name.size()) {
          out["texture_pack_name"] = t.name;
        }
      }

      fan::json images_array = fan::json::array();

      auto main_image = shape.get_image();
      auto img_json = fan::graphics::image_to_json(main_image);
      if (!img_json.empty()) {
        images_array.push_back(img_json);
      }

      auto images = shape.get_images();
      for (auto& image : images) {
        img_json = fan::graphics::image_to_json(image);
        if (!img_json.empty()) {
          images_array.push_back(img_json);
        }
      }

      if (!images_array.empty()) {
        out["images"] = images_array;
      }

      break;
    }
    case fan::graphics::shapes::shape_type_t::text: {
      out["shape"] = "text";
      break;
    }
    case fan::graphics::shapes::shape_type_t::circle: {
      fan::graphics::shapes::circle_t::properties_t defaults;
      out["shape"] = "circle";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_radius() != defaults.radius) {
        out["radius"] = shape.get_radius();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::grid: {
      fan::graphics::shapes::grid_t::properties_t defaults;
      out["shape"] = "grid";
      if (shape.get_position() != defaults.position) {
        out["position"] = shape.get_position();
      }
      if (shape.get_size() != defaults.size) {
        out["size"] = shape.get_size();
      }
      if (shape.get_grid_size() != defaults.grid_size) {
        out["grid_size"] = shape.get_grid_size();
      }
      if (shape.get_rotation_point() != defaults.rotation_point) {
        out["rotation_point"] = shape.get_rotation_point();
      }
      if (shape.get_color() != defaults.color) {
        out["color"] = shape.get_color();
      }
      if (shape.get_angle() != defaults.angle) {
        out["angle"] = shape.get_angle();
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::particles: {
      fan::graphics::shapes::particles_t::properties_t defaults;
      auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper);
      out["shape"] = "particles";
      if (ri.position != defaults.position) {
        out["position"] = ri.position;
      }
      if (ri.size != defaults.size) {
        out["size"] = ri.size;
      }
      if (ri.color != defaults.color) {
        out["color"] = ri.color;
      }
      if (ri.begin_time != defaults.begin_time) {
        out["begin_time"] = ri.begin_time;
      }
      if (ri.alive_time != defaults.alive_time) {
        out["alive_time"] = ri.alive_time;
      }
      if (ri.respawn_time != defaults.respawn_time) {
        out["respawn_time"] = ri.respawn_time;
      }
      if (ri.count != defaults.count) {
        out["count"] = ri.count;
      }
      if (ri.position_velocity != defaults.position_velocity) {
        out["position_velocity"] = ri.position_velocity;
      }
      if (ri.angle_velocity != defaults.angle_velocity) {
        out["angle_velocity"] = ri.angle_velocity;
      }
      if (ri.begin_angle != defaults.begin_angle) {
        out["begin_angle"] = ri.begin_angle;
      }
      if (ri.end_angle != defaults.end_angle) {
        out["end_angle"] = ri.end_angle;
      }
      if (ri.angle != defaults.angle) {
        out["angle"] = ri.angle;
      }
      if (ri.gap_size != defaults.gap_size) {
        out["gap_size"] = ri.gap_size;
      }
      if (ri.max_spread_size != defaults.max_spread_size) {
        out["max_spread_size"] = ri.max_spread_size;
      }
      if (ri.size_velocity != defaults.size_velocity) {
        out["size_velocity"] = ri.size_velocity;
      }
      if (ri.shape != defaults.shape) {
        out["particle_shape"] = ri.shape;
      }
      if (ri.blending != defaults.blending) {
        out["blending"] = ri.blending;
      }
      fan::graphics::image_t image = shape.get_image();
      if (image) {
        out.update(fan::graphics::image_to_json(image), true);
      }
      break;
    }
    default: {
      fan::throw_error("unimplemented shape");
    }
    }
    return false;
  }
  bool json_to_shape(const fan::json& in, fan::graphics::shapes::shape_t* shape, const std::source_location& callers_path) {
    std::string shape_type = in["shape"];
    switch (fan::get_hash(shape_type.c_str())) {
    case fan::get_hash("rectangle"): {
      fan::graphics::shapes::rectangle_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("outline_color")) {
        p.outline_color = in["outline_color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("light"): {
      fan::graphics::shapes::light_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("parallax_factor")) {
        p.parallax_factor = in["parallax_factor"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("flags")) {
        p.flags = in["flags"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("line"): {
      fan::graphics::shapes::line_t::properties_t p;
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("src")) {
        p.src = in["src"];
      }
      if (in.contains("dst")) {
        p.dst = in["dst"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("sprite"): {
      fan::graphics::shapes::sprite_t::properties_t p;
      p.blending = true;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("parallax_factor")) {
        p.parallax_factor = in["parallax_factor"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      if (in.contains("flags")) {
        p.flags = in["flags"];
      }
      if (in.contains("tc_position")) {
        p.tc_position = in["tc_position"];
      }
      if (in.contains("tc_size")) {
        p.tc_size = in["tc_size"];
      }
      bool contains_animations = in.contains("animations");
      // load texture pack only if no sprite sheet animations
      // because sprite sheet animations use image for it
      if (contains_animations == false && in.contains("texture_pack_name") && *fan::graphics::g_shapes->texture_pack) {
        p.texture_pack_unique_id = (*fan::graphics::g_shapes->texture_pack)[in["texture_pack_name"]];
      }
      *shape = p;

      fan::graphics::image_load_properties_t lp;
      if (in.contains("image_visual_output")) {
        lp.visual_output = in["image_visual_output"];
      }
      if (in.contains("image_format")) {
        lp.format = in["image_format"];
      }
      if (in.contains("image_type")) {
        lp.type = in["image_type"];
      }
      if (in.contains("image_min_filter")) {
        lp.min_filter = in["image_min_filter"];
      }
      if (in.contains("image_mag_filter")) {
        lp.mag_filter = in["image_mag_filter"];
      }
      if (in.contains("images") && in["images"].is_array()) {
        for (const auto [i, image_json] : fan::enumerate(in["images"])) {
          fan::graphics::image_t image = fan::graphics::json_to_image(image_json, callers_path);
          if (i == 0) {
            shape->set_image(image);
          }
          else {
            auto images = shape->get_images();
            images[i - 1] = image;
            shape->set_images(images);
          }
        }
      }

      if (contains_animations) {
        for (auto& item : in["animations"]) {
          uint32_t anim_id = item.get<uint32_t>();
          auto existing_animation = fan::graphics::get_sprite_sheet_animation(anim_id);
          shape->add_existing_animation(anim_id);
        }
      }

      break;
    }
    case fan::get_hash("unlit_sprite"): {
      fan::graphics::shapes::unlit_sprite_t::properties_t p;
      p.blending = true;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("parallax_factor")) {
        p.parallax_factor = in["parallax_factor"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      if (in.contains("flags")) {
        p.flags = in["flags"];
      }
      if (in.contains("tc_position")) {
        p.tc_position = in["tc_position"];
      }
      if (in.contains("tc_size")) {
        p.tc_size = in["tc_size"];
      }
      if (in.contains("texture_pack_name") && *fan::graphics::g_shapes->texture_pack) {
        p.texture_pack_unique_id = (*fan::graphics::g_shapes->texture_pack)[in["texture_pack_name"]];
      }
      *shape = p;
      fan::graphics::image_load_properties_t lp;
      if (in.contains("image_visual_output")) {
        lp.visual_output = in["image_visual_output"];
      }
      if (in.contains("image_format")) {
        lp.format = in["image_format"];
      }
      if (in.contains("image_type")) {
        lp.type = in["image_type"];
      }
      if (in.contains("image_min_filter")) {
        lp.min_filter = in["image_min_filter"];
      }
      if (in.contains("image_mag_filter")) {
        lp.mag_filter = in["image_mag_filter"];
      }

      if (in.contains("images") && in["images"].is_array()) {
        for (const auto [i, image_json] : fan::enumerate(in["images"])) {
          if (!image_json.contains("image_path")) continue;

          auto path = image_json["image_path"];
          if (fan::io::file::exists(path)) {
            fan::graphics::image_load_properties_t lp;

            if (image_json.contains("image_visual_output")) lp.visual_output = image_json["image_visual_output"];
            if (image_json.contains("image_format")) lp.format = image_json["image_format"];
            if (image_json.contains("image_type")) lp.type = image_json["image_type"];
            if (image_json.contains("image_min_filter")) lp.min_filter = image_json["image_min_filter"];
            if (image_json.contains("image_mag_filter")) lp.mag_filter = image_json["image_mag_filter"];

            auto image = fan::graphics::g_render_context_handle->image_load_path_props(fan::graphics::g_render_context_handle, path, lp, callers_path);

            if (i == 0) {
              shape->set_image(image);
            }
            else {
              auto images = shape->get_images();
              images[i - 1] = image;
              shape->set_images(images);
            }
            (*fan::graphics::g_render_context_handle.image_list)[image].image_path = path;
          }
        }
      }
      break;
    }
    case fan::get_hash("circle"): {
      fan::graphics::shapes::circle_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("radius")) {
        p.radius = in["radius"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("grid"): {
      fan::graphics::shapes::grid_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("grid_size")) {
        p.grid_size = in["grid_size"];
      }
      if (in.contains("rotation_point")) {
        p.rotation_point = in["rotation_point"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      *shape = p;
      break;
    }
    case fan::get_hash("particles"): {
      fan::graphics::shapes::particles_t::properties_t p;
      if (in.contains("position")) {
        p.position = in["position"];
      }
      if (in.contains("size")) {
        p.size = in["size"];
      }
      if (in.contains("color")) {
        p.color = in["color"];
      }
      if (in.contains("begin_time")) {
        p.begin_time = in["begin_time"];
      }
      if (in.contains("alive_time")) {
        p.alive_time = in["alive_time"];
      }
      if (in.contains("respawn_time")) {
        p.respawn_time = in["respawn_time"];
      }
      if (in.contains("count")) {
        p.count = in["count"];
      }
      if (in.contains("position_velocity")) {
        p.position_velocity = in["position_velocity"];
      }
      if (in.contains("angle_velocity")) {
        p.angle_velocity = in["angle_velocity"];
      }
      if (in.contains("begin_angle")) {
        p.begin_angle = in["begin_angle"];
      }
      if (in.contains("end_angle")) {
        p.end_angle = in["end_angle"];
      }
      if (in.contains("angle")) {
        p.angle = in["angle"];
      }
      if (in.contains("gap_size")) {
        p.gap_size = in["gap_size"];
      }
      if (in.contains("max_spread_size")) {
        p.max_spread_size = in["max_spread_size"];
      }
      if (in.contains("size_velocity")) {
        p.size_velocity = in["size_velocity"];
      }
      if (in.contains("particle_shape")) {
        p.shape = in["particle_shape"];
      }
      if (in.contains("blending")) {
        p.blending = in["blending"];
      }
      p.image = fan::graphics::json_to_image(in, callers_path);
      *shape = p;
      break;
    }
    default: {
      fan::throw_error("unimplemented shape");
    }
    }
    return false;
  }
  bool shape_serialize(fan::graphics::shapes::shape_t& shape, fan::json* out) {
		return shape_to_json(shape, out);
	}
}
#endif

namespace fan::graphics {
  bool shape_to_bin(fan::graphics::shapes::shape_t& shape, std::vector<uint8_t>* data) {
    std::vector<uint8_t>& out = *data;
    fan::write_to_vector(out, shape.get_shape_type());
    fan::write_to_vector(out, shape.gint());
    switch (shape.get_shape_type()) {
    case fan::graphics::shapes::shape_type_t::light: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_parallax_factor());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_flags());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::line: {
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_src());
      fan::write_to_vector(out, shape.get_dst());
      break;
    case fan::graphics::shapes::shape_type_t::rectangle: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_parallax_factor());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      fan::write_to_vector(out, shape.get_flags());
      fan::write_to_vector(out, shape.get_image_data().image_path);
      fan::graphics::image_load_properties_t ilp = fan::graphics::g_render_context_handle->image_get_settings(fan::graphics::g_render_context_handle, shape.get_image());
      fan::write_to_vector(out, ilp.visual_output);
      fan::write_to_vector(out, ilp.format);
      fan::write_to_vector(out, ilp.type);
      fan::write_to_vector(out, ilp.min_filter);
      fan::write_to_vector(out, ilp.mag_filter);
      fan::write_to_vector(out, shape.get_tc_position());
      fan::write_to_vector(out, shape.get_tc_size());
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_parallax_factor());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      fan::write_to_vector(out, shape.get_flags());
      fan::write_to_vector(out, shape.get_image_data().image_path);
      fan::graphics::image_load_properties_t ilp = fan::graphics::g_render_context_handle->image_get_settings(fan::graphics::g_render_context_handle, shape.get_image());
      fan::write_to_vector(out, ilp.visual_output);
      fan::write_to_vector(out, ilp.format);
      fan::write_to_vector(out, ilp.type);
      fan::write_to_vector(out, ilp.min_filter);
      fan::write_to_vector(out, ilp.mag_filter);
      fan::write_to_vector(out, shape.get_tc_position());
      fan::write_to_vector(out, shape.get_tc_size());
      break;
    }
    case fan::graphics::shapes::shape_type_t::circle: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_radius());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::grid: {
      fan::write_to_vector(out, shape.get_position());
      fan::write_to_vector(out, shape.get_size());
      fan::write_to_vector(out, shape.get_grid_size());
      fan::write_to_vector(out, shape.get_rotation_point());
      fan::write_to_vector(out, shape.get_color());
      fan::write_to_vector(out, shape.get_angle());
      break;
    }
    case fan::graphics::shapes::shape_type_t::particles: {
      auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper);
      fan::write_to_vector(out, ri.position);
      fan::write_to_vector(out, ri.size);
      fan::write_to_vector(out, ri.color);
      fan::write_to_vector(out, ri.begin_time);
      fan::write_to_vector(out, ri.alive_time);
      fan::write_to_vector(out, ri.respawn_time);
      fan::write_to_vector(out, ri.count);
      fan::write_to_vector(out, ri.position_velocity);
      fan::write_to_vector(out, ri.angle_velocity);
      fan::write_to_vector(out, ri.begin_angle);
      fan::write_to_vector(out, ri.end_angle);
      fan::write_to_vector(out, ri.angle);
      fan::write_to_vector(out, ri.gap_size);
      fan::write_to_vector(out, ri.max_spread_size);
      fan::write_to_vector(out, ri.size_velocity);
      fan::write_to_vector(out, ri.shape);
      fan::write_to_vector(out, ri.blending);
      break;
    }
    }
    case fan::graphics::shapes::shape_type_t::light_end: {
      break;
    }
    default: {
      fan::throw_error("unimplemented shape");
    }
    }
    return false;
  }
  bool bin_to_shape(const std::vector<uint8_t>& in, fan::graphics::shapes::shape_t* shape, uint64_t& offset, const std::source_location& callers_path) {
    using sti_t = std::remove_reference_t<decltype(fan::graphics::shapes::shape_t().get_shape_type())>;
    using nr_t = std::remove_reference_t<decltype(fan::graphics::shapes::shape_t().gint())>;
    sti_t shape_type = fan::vector_read_data<sti_t>(in, offset);
    nr_t nri = fan::vector_read_data<nr_t>(in, offset);
    switch (shape_type) {
    case fan::graphics::shapes::shape_type_t::rectangle: {
      fan::graphics::shapes::rectangle_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.outline_color = p.color;
      *shape = p;
      return false;
    }
    case fan::graphics::shapes::shape_type_t::light: {
      fan::graphics::shapes::light_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::line: {
      fan::graphics::shapes::line_t::properties_t p;
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.src = fan::vector_read_data<decltype(p.src)>(in, offset);
      p.dst = fan::vector_read_data<decltype(p.dst)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::sprite: {
      fan::graphics::shapes::sprite_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);

      std::string image_path = fan::vector_read_data<std::string>(in, offset);
      fan::graphics::image_load_properties_t ilp;
      ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
      ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
      ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
      ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
      ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
      p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
      p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
      *shape = p;
      if (image_path.size()) {
        shape->get_image_data().image_path = image_path;
        shape->set_image(fan::graphics::g_render_context_handle->image_load_path_props(fan::graphics::g_render_context_handle, image_path, ilp, callers_path));
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::unlit_sprite: {
      fan::graphics::shapes::unlit_sprite_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.parallax_factor = fan::vector_read_data<decltype(p.parallax_factor)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.flags = fan::vector_read_data<decltype(p.flags)>(in, offset);
      std::string image_path = fan::vector_read_data<std::string>(in, offset);
      fan::graphics::image_load_properties_t ilp;
      ilp.visual_output = fan::vector_read_data<decltype(ilp.visual_output)>(in, offset);
      ilp.format = fan::vector_read_data<decltype(ilp.format)>(in, offset);
      ilp.type = fan::vector_read_data<decltype(ilp.type)>(in, offset);
      ilp.min_filter = fan::vector_read_data<decltype(ilp.min_filter)>(in, offset);
      ilp.mag_filter = fan::vector_read_data<decltype(ilp.mag_filter)>(in, offset);
      p.tc_position = fan::vector_read_data<decltype(p.tc_position)>(in, offset);
      p.tc_size = fan::vector_read_data<decltype(p.tc_size)>(in, offset);
      *shape = p;
      if (image_path.size()) {
        shape->get_image_data().image_path = image_path;
        shape->set_image(fan::graphics::g_render_context_handle->image_load_path_props(fan::graphics::g_render_context_handle, image_path, ilp, callers_path));
      }
      break;
    }
    case fan::graphics::shapes::shape_type_t::circle: {
      fan::graphics::shapes::circle_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.radius = fan::vector_read_data<decltype(p.radius)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::grid: {
      fan::graphics::shapes::grid_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.grid_size = fan::vector_read_data<decltype(p.grid_size)>(in, offset);
      p.rotation_point = fan::vector_read_data<decltype(p.rotation_point)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::particles: {
      fan::graphics::shapes::particles_t::properties_t p;
      p.position = fan::vector_read_data<decltype(p.position)>(in, offset);
      p.size = fan::vector_read_data<decltype(p.size)>(in, offset);
      p.color = fan::vector_read_data<decltype(p.color)>(in, offset);
      p.begin_time = fan::vector_read_data<decltype(p.begin_time)>(in, offset);
      p.alive_time = fan::vector_read_data<decltype(p.alive_time)>(in, offset);
      p.respawn_time = fan::vector_read_data<decltype(p.respawn_time)>(in, offset);
      p.count = fan::vector_read_data<decltype(p.count)>(in, offset);
      p.position_velocity = fan::vector_read_data<decltype(p.position_velocity)>(in, offset);
      p.angle_velocity = fan::vector_read_data<decltype(p.angle_velocity)>(in, offset);
      p.begin_angle = fan::vector_read_data<decltype(p.begin_angle)>(in, offset);
      p.end_angle = fan::vector_read_data<decltype(p.end_angle)>(in, offset);
      p.angle = fan::vector_read_data<decltype(p.angle)>(in, offset);
      p.gap_size = fan::vector_read_data<decltype(p.gap_size)>(in, offset);
      p.max_spread_size = fan::vector_read_data<decltype(p.max_spread_size)>(in, offset);
      p.size_velocity = fan::vector_read_data<decltype(p.size_velocity)>(in, offset);
      p.shape = fan::vector_read_data<decltype(p.shape)>(in, offset);
      p.blending = fan::vector_read_data<decltype(p.blending)>(in, offset);
      *shape = p;
      break;
    }
    case fan::graphics::shapes::shape_type_t::light_end: {
      return false;
    }
    default: {
      fan::throw_error("unimplemented");
    }
    }
    if (shape->gint() != nri) {
      fan::throw_error("");
    }
    return false;
  }

  bool shape_serialize(fan::graphics::shapes::shape_t& shape, std::vector<uint8_t>* out) {
		return shape_to_bin(shape, out);
	}
#if defined(fan_physics)
  bool shape_deserialize_t::iterate(const fan::json& json, fan::graphics::shapes::shape_t* shape, const std::source_location& callers_path) {
    if (init == false) {
      data.it = json.cbegin();
      init = true;
    }
    if (data.it == json.cend() || was_object) {
      return 0;
    }
    if (json.type() == fan::json::value_t::object) {
      json_to_shape(json, shape, callers_path);
      was_object = true;
      return 1;
    }
    else {
      json_to_shape(*data.it, shape, callers_path);
      ++data.it;
    }
    return 1;
  }
#endif
  bool shape_deserialize_t::iterate(const std::vector<uint8_t>& bin_data, fan::graphics::shapes::shape_t* shape) {
    if (bin_data.empty()) {
      return 0;
    }
    else if (data.offset >= bin_data.size()) {
      return 0;
    }
    bin_to_shape(bin_data, shape, data.offset);
    return 1;
  }
#if defined(fan_physics)
  fan::graphics::shapes::shape_t extract_single_shape(const fan::json& json_data, const std::source_location& callers_path) {
    fan::graphics::shape_deserialize_t iterator;
    fan::graphics::shapes::shape_t shape;
    iterator.iterate(json_data["shapes"], &shape, callers_path);
    return shape;
  }
  fan::json read_json(const std::string& path, const std::source_location& callers_path) {
    std::string json_bytes;
    fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &json_bytes);
    return fan::json::parse(json_bytes);
  }
#endif
}

#if defined(fan_json)
  fan::graphics::shapes::shape_t::operator fan::json() {
	  fan::json out;
	  fan::graphics::shape_to_json(*this, &out);
	  return out;
  }
  fan::graphics::shapes::shape_t::operator std::string() {
	  fan::json out;
	  fan::graphics::shape_to_json(*this, &out);
	  return out.dump(2);
  }
  fan::graphics::shapes::shape_t::shape_t(const fan::json& json) : fan::graphics::shapes::shape_t() {
	  fan::graphics::json_to_shape(json, this);
  }
  fan::graphics::shapes::shape_t::shape_t(const std::string& json_string) : fan::graphics::shapes::shape_t() {
	  *this = fan::json::parse(json_string);
  }
  fan::graphics::shapes::shape_t& fan::graphics::shapes::shape_t::operator=(const fan::json& json) {
	  fan::graphics::json_to_shape(json, this);
	  return *this;
  }
  fan::graphics::shapes::shape_t& fan::graphics::shapes::shape_t::operator=(const std::string& json_string) {
	  return fan::graphics::shapes::shape_t::operator=(fan::json::parse(json_string));
  }
#endif