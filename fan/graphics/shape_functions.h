static void vi_move(uint16_t shape_type, void* vi) {

}

static void ri_move(uint16_t sti, void* ri) {
	if (sti == fan::graphics::shapes::shape_type_t::sprite) {
		((fan::graphics::shapes::sprite_t::ri_t*)ri)->sprite_sheet_data.frame_update_nr.sic();
	}
}

enum class shape_category_t {
	light,
	common,
	texture,
	unsupported
};

inline static constexpr shape_category_t get_shape_category(uint16_t sti) {
	switch (sti) {
	case fan::graphics::shapes::shape_type_t::light:
		return shape_category_t::light;

	case fan::graphics::shapes::shape_type_t::capsule:
	case fan::graphics::shapes::shape_type_t::gradient:
	case fan::graphics::shapes::shape_type_t::grid:
	case fan::graphics::shapes::shape_type_t::circle:
	case fan::graphics::shapes::shape_type_t::rectangle:
	#if defined(fan_3D)
	case fan::graphics::shapes::shape_type_t::rectangle3d:
	#endif
	case fan::graphics::shapes::shape_type_t::line:
		return shape_category_t::common;

	case fan::graphics::shapes::shape_type_t::particles:
	case fan::graphics::shapes::shape_type_t::universal_image_renderer:
	case fan::graphics::shapes::shape_type_t::unlit_sprite:
	case fan::graphics::shapes::shape_type_t::sprite:
	case fan::graphics::shapes::shape_type_t::shader_shape:
		return shape_category_t::texture;

	default:
		return shape_category_t::unsupported;
	}
	return shape_category_t::unsupported;
}

template <typename modifier_t>
inline static void update_shape(shape_t* shape, modifier_t&& modifier_fn) {
	auto sti = shape->get_shape_type();

	auto key_pack_size_t = g_shapes->shaper.GetKeysSize(*shape);
	std::unique_ptr<uint8_t[]> key_pack(new uint8_t[key_pack_size_t]);
	g_shapes->shaper.WriteKeys(*shape, key_pack.get());

	modifier_fn(sti, key_pack.get());

	auto _vi = shape->GetRenderData(g_shapes->shaper);
	auto vlen_t = g_shapes->shaper.GetRenderDataSize(sti);
	std::unique_ptr<uint8_t[]> vi(new uint8_t[vlen_t]);
	std::memcpy(vi.get(), _vi, vlen_t);

	auto _ri = shape->GetData(g_shapes->shaper);
	auto rlen_t = g_shapes->shaper.GetDataSize(sti);
	std::unique_ptr<uint8_t[]> ri(new uint8_t[rlen_t]);
	std::memcpy(ri.get(), _ri, rlen_t);

	vi_move(sti, _vi);
	ri_move(sti, _ri);

	shape->remove();
	*shape = g_shapes->shaper.add(sti, key_pack.get(), key_pack_size_t, vi.get(), ri.get());

#if defined(debug_shape_t)
	fan::print("+", shape->nri);
#endif
}

template<typename sti_t, typename key_pack_t>
inline static void set_position_impl(sti_t sti, key_pack_t key_pack, const fan::vec3& position) {
	switch (get_shape_category(sti)) {
		case shape_category_t::common:
			shaper_get_key_safe(depth_t, common_t, depth) = position.z;
			break;
		case shape_category_t::texture:
			shaper_get_key_safe(depth_t, texture_t, depth) = position.z;
			break;
		case shape_category_t::light:
			break;
		default:
			fan::print("unimplemented");
	}
}

inline static void set_position(shape_t* shape, const fan::vec3& position) {
	update_shape(shape, [&](auto sti, auto key_pack) {
		set_position_impl(sti, key_pack, position);
	});
}

inline static fan::graphics::camera_t get_camera(const shape_t* shape) {
	auto sti = shape->get_shape_type();
	uint8_t* key_pack = g_shapes->shaper.GetKeys(*shape);

	switch (get_shape_category(sti)) {
		case shape_category_t::light:
			return shaper_get_key_safe(camera_t, light_t, camera);
		case shape_category_t::common:
			return shaper_get_key_safe(camera_t, common_t, camera);
		case shape_category_t::texture:
			return shaper_get_key_safe(camera_t, texture_t, camera);
		default:
			fan::throw_error("unimplemented");
	}
}

template<typename sti_t, typename key_pack_t>
inline static void set_camera_impl(sti_t sti, key_pack_t key_pack, fan::graphics::camera_t camera) {
	auto cat_t = get_shape_category(sti);

	switch (cat_t) {
		case shape_category_t::light:
			shaper_get_key_safe(camera_t, light_t, camera) = camera;
			break;
		case shape_category_t::common:
			shaper_get_key_safe(camera_t, common_t, camera) = camera;
			break;
		case shape_category_t::texture:
			shaper_get_key_safe(camera_t, texture_t, camera) = camera;
			break;
		default:
			fan::throw_error("unimplemented");
	}
}

inline static void set_camera(shape_t* shape, fan::graphics::camera_t camera) {
	update_shape(shape, [&](auto sti, auto key_pack) {
		set_camera_impl(sti, key_pack, camera);
	});
}

inline static fan::graphics::viewport_t get_viewport(const shape_t* shape) {
	auto sti = shape->get_shape_type();
	uint8_t* key_pack = g_shapes->shaper.GetKeys(*shape);

	switch (get_shape_category(sti)) {
		case shape_category_t::light:
			return shaper_get_key_safe(viewport_t, light_t, viewport);
		case shape_category_t::common:
			return shaper_get_key_safe(viewport_t, common_t, viewport);
		case shape_category_t::texture:
			return shaper_get_key_safe(viewport_t, texture_t, viewport);
		default:
			fan::throw_error("unimplemented");
	}
}

template<typename sti_t, typename key_pack_t>
inline static void set_viewport_impl(sti_t sti, key_pack_t key_pack, fan::graphics::viewport_t viewport) {
	auto cat_t = get_shape_category(sti);

	switch (cat_t) {
		case shape_category_t::light:
			shaper_get_key_safe(viewport_t, light_t, viewport) = viewport;
			break;
		case shape_category_t::common:
			shaper_get_key_safe(viewport_t, common_t, viewport) = viewport;
			break;
		case shape_category_t::texture:
			shaper_get_key_safe(viewport_t, texture_t, viewport) = viewport;
			break;
		default:
			fan::throw_error("unimplemented");
	}
}

inline static void set_viewport(shape_t* shape, fan::graphics::viewport_t viewport) {
	update_shape(shape, [&](auto sti, auto key_pack) {
		set_viewport_impl(sti, key_pack, viewport);
	});
}

inline static fan::graphics::image_t get_image(shape_t* shape) {
	auto sti = g_shapes->shaper.ShapeList[*shape].sti;
	uint8_t* key_pack = g_shapes->shaper.GetKeys(*shape);

	if (get_shape_category(sti) == shape_category_t::texture) {
		return shaper_get_key_safe(image_t, texture_t, image);
	}

	fan::throw_error("unimplemented");
}

template<typename sti_t, typename key_pack_t>
inline static void set_image_impl(sti_t sti, key_pack_t key_pack, fan::graphics::image_t image) {
	if (get_shape_category(sti) == shape_category_t::texture) {
		shaper_get_key_safe(image_t, texture_t, image) = image;
	} else {
		fan::throw_error("unimplemented");
	}
}

inline static void set_image(shape_t* shape, fan::graphics::image_t image) {
	if (shape->get_image() == image) return;

	update_shape(shape, [&](auto sti, auto KeyPack) {
		set_image_impl(sti, KeyPack, image);
	});
}

struct shape_functions_t {

#define GEN_SHAPES_SKIP(x)
#define SHAPE_FUNCS(X) GEN_SHAPES(X, GEN_SHAPES_SKIP)
#define GEN_SHAPE_LIST_SKIP(x) x
#define SHAPE_LIST(X) GEN_SHAPES(X, GEN_SHAPE_LIST_SKIP)

	template <typename fn>
	struct base_functions_t {
		fn func;
		base_functions_t(fn f = nullptr) : func(f) {}
		operator fn() const { return func; }
	};

	template <typename>
	struct make_universal;

	template <typename ret_t, typename... args_t>
	static auto universal_callback(args_t&&... args) -> ret_t {
		fan::print_throttled("[universal_callback] called with " + std::to_string(sizeof...(args_t)) + " arguments");

		if constexpr (sizeof...(args_t) > 0) {
			using first_arg_t = std::tuple_element_t<0, std::tuple<args_t...>>;
			if constexpr (std::is_same_v<first_arg_t, shape_t*> ||
				std::is_same_v<first_arg_t, const shape_t*>) {
				auto* s = const_cast<shape_t*>(std::get<0>(std::forward_as_tuple(args...)));
				fan::print_throttled("  for shape " + fan::graphics::shapes::shape_names[s->get_shape_type()]);
			}
		}

		if constexpr (std::is_void_v<ret_t>) {
			return;
		}
		else if constexpr (std::is_reference_v<ret_t>) {
			static std::remove_reference_t<ret_t> dummy{};
			return dummy;
		}
		else {
			return ret_t{};
		}
	}

	template <typename ret_t, typename... args_t>
	struct make_universal<ret_t(*)(args_t...)> {
		static ret_t fn(args_t... args) {
			return universal_callback<ret_t>(std::forward<args_t>(args)...);
		}
	};

#define GENERATE_DEFAULT_PUSH_BACK(shape) \
	static shape_t push_back_##shape(void* properties) { \
		return g_shapes->shape.push_back(*reinterpret_cast<shape##_t::properties_t*>(properties)); \
	}
	SHAPE_FUNCS(GENERATE_DEFAULT_PUSH_BACK)
	#undef GENERATE_DEFAULT_PUSH_BACK

	#define VTABLE_OPS(X, shape) \
	X(shape, push_back, shape_t(*)(void*)) \
	X(shape, get_position, fan::vec3(*)(const shape_t*)) \
	X(shape, set_position2, void(*)(shape_t*, const fan::vec2&)) \
	X(shape, set_position3, void(*)(shape_t*, const fan::vec3&)) \
  X(shape, set_position3_impl, void(*)(shape_t*, const fan::vec3&)) \
	X(shape, get_size, fan::vec2(*)(const shape_t*)) \
	X(shape, get_size3, fan::vec3(*)(const shape_t*)) \
	X(shape, set_size, void(*)(shape_t*, const fan::vec2&)) \
	X(shape, set_size3, void(*)(shape_t*, const fan::vec3&)) \
	X(shape, get_rotation_point, fan::vec2(*)(const shape_t*)) \
	X(shape, set_rotation_point, void(*)(shape_t*, const fan::vec2&)) \
	X(shape, get_color, fan::color(*)(const shape_t*)) \
	X(shape, set_color, void(*)(shape_t*, const fan::color&)) \
	X(shape, get_angle, fan::vec3(*)(const shape_t*)) \
	X(shape, set_angle, void(*)(shape_t*, const fan::vec3&)) \
	X(shape, get_tc_position, fan::vec2(*)(const shape_t*)) \
	X(shape, set_tc_position, void(*)(shape_t*, const fan::vec2&)) \
	X(shape, get_tc_size, fan::vec2(*)(const shape_t*)) \
	X(shape, set_tc_size, void(*)(shape_t*, const fan::vec2&)) \
	X(shape, get_grid_size, fan::vec2(*)(const shape_t*)) \
	X(shape, set_grid_size, void(*)(shape_t*, const fan::vec2&)) \
	X(shape, get_camera, fan::graphics::camera_t(*)(const shape_t*)) \
	X(shape, set_camera, void(*)(shape_t*, fan::graphics::camera_t)) \
	X(shape, get_viewport, fan::graphics::viewport_t(*)(const shape_t*)) \
	X(shape, set_viewport, void(*)(shape_t*, fan::graphics::viewport_t)) \
	X(shape, get_image, fan::graphics::image_t(*)(const shape_t*)) \
	X(shape, set_image, void(*)(shape_t*, fan::graphics::image_t)) \
	X(shape, get_image_data, fan::graphics::image_data_t& (*)(const shape_t*)) \
	X(shape, get_parallax_factor, f32_t(*)(const shape_t*)) \
	X(shape, set_parallax_factor, void(*)(shape_t*, f32_t)) \
	X(shape, get_flags, uint32_t(*)(const shape_t*)) \
	X(shape, set_flags, void(*)(shape_t*, uint32_t)) \
	X(shape, get_radius, f32_t(*)(const shape_t*)) \
	X(shape, get_src, fan::vec3(*)(const shape_t*)) \
	X(shape, get_dst, fan::vec2(*)(const shape_t*)) \
	X(shape, get_outline_size, f32_t(*)(const shape_t*)) \
	X(shape, get_outline_color, fan::color(*)(const shape_t*)) \
	X(shape, set_outline_color, void(*)(shape_t*, const fan::color&))

	struct shape_vtable_t {
	#define MAKE_MEMBER(shape, op, cb_type) base_functions_t<cb_type> op;
		VTABLE_OPS(MAKE_MEMBER, dummy)
		#undef MAKE_MEMBER
	};

	template<typename T>
	struct extract_function_type;

	template<typename func_ptr_t>
	struct extract_function_type<base_functions_t<func_ptr_t>> {
		using type = func_ptr_t;
	};

	template<typename T>
	using extract_function_type_t = typename extract_function_type<T>::type;

	template <typename>
	struct arg_traits;

	template <typename ret_t, typename shape_t2, typename value_t>
	struct arg_traits<ret_t(*)(shape_t2, value_t)> {
		using shape_arg_t = shape_t2;
		using value_arg_t = value_t;
	};

#define GENERATE_GETTER(op, shape) \
	static typename std::invoke_result< \
		extract_function_type_t<decltype(shape_vtable_t::CONCAT(get_, op))>, \
		const shape_t* \
	>::type CONCAT4(get_, op, _, shape)(const shape_t* s) { \
		using func_type = extract_function_type_t<decltype(shape_vtable_t::CONCAT(get_, op))>; \
		using return_t = typename std::invoke_result<func_type, const shape_t*>::type; \
		\
		static auto get_field_value = [](auto* data) -> return_t { \
			auto& field = data->op; \
			using field_t = std::remove_reference_t<decltype(field)>; \
			static constexpr bool is_type_fan = fan::is_vector_type_v<field_t> || fan::is_color_type_v<field_t>; \
			if constexpr (requires { field[0]; field.size(); } && !std::is_same_v<decltype(field), std::string> && !is_type_fan) { \
				return static_cast<return_t>(field[0]); \
			} else { \
				return static_cast<return_t>(field); \
			} \
		}; \
		\
		return []<typename T>(const shape_t* s) -> return_t { \
			if constexpr (requires { typename T::vi_t; } && requires { std::declval<typename T::vi_t>().op; }) { \
				using vi_t = typename T::vi_t; \
				return get_field_value(reinterpret_cast<vi_t*>(s->GetRenderData(g_shapes->shaper))); \
			} else if constexpr (requires { typename T::ri_t; } && requires { std::declval<typename T::ri_t>().op; }) { \
				using ri_t = typename T::ri_t; \
				return get_field_value(reinterpret_cast<ri_t*>(s->GetData(g_shapes->shaper))); \
			} else { \
				return make_universal<func_type>::fn(s); \
			} \
		}.template operator()<shape##_t>(s); \
	}

#define GENERATE_SETTER_ACTUAL(op, shape, actual_op_name) \
	static void CONCAT4(set_, op, _, shape)( \
		shape_t* s, \
		typename arg_traits<extract_function_type_t<decltype(shape_vtable_t::CONCAT(set_, op))>>::value_arg_t v) { \
		\
		using func_type = extract_function_type_t<decltype(shape_vtable_t::CONCAT(set_, op))>; \
		using value_t = typename arg_traits<func_type>::value_arg_t; \
		\
		static auto set_field = [](shape_t* s, auto* data, auto value, auto member_ptr) { \
			auto& field = data->*member_ptr; \
			using field_t = std::remove_reference_t<decltype(field)>; \
			static constexpr bool is_type_fan = fan::is_vector_type_v<field_t> || fan::is_color_type_v<field_t>; \
			if constexpr (requires { field[0]; field.size(); } && !std::is_same_v<field_t, std::string> && !is_type_fan) { \
				field.fill(value); \
			} else { \
				field = value; \
				auto& sldata = g_shapes->shaper.ShapeList[*s]; \
				g_shapes->shaper.ElementIsPartiallyEdited( \
					sldata.sti, sldata.blid, sldata.ElementIndex, fan::member_offset(member_ptr), sizeof(field)); \
			} \
		}; \
		\
		[]<typename T>(shape_t* s, value_t v) { \
			if constexpr (requires { typename T::vi_t; } && requires { std::declval<typename T::vi_t>().actual_op_name; }) { \
				using vi_t = typename T::vi_t; \
				set_field(s, reinterpret_cast<vi_t*>(s->GetRenderData(g_shapes->shaper)), v, &vi_t::actual_op_name); \
			} else if constexpr (requires { typename T::ri_t; } && requires { std::declval<typename T::ri_t>().actual_op_name; }) { \
				using ri_t = typename T::ri_t; \
				set_field(s, reinterpret_cast<ri_t*>(s->GetData(g_shapes->shaper)), v, &ri_t::actual_op_name); \
			} else { \
				make_universal<func_type>::fn(s, v); \
			} \
		}.template operator()<shape##_t>(s, v); \
	}

#define GENERATE_SETTER(op, shape)  GENERATE_SETTER_ACTUAL(op, shape, op)

#define COMMON_OPS(shape, macro) \
	macro(size, shape) \
	macro(size3, shape) \
	macro(rotation_point, shape) \
	macro(color, shape) \
	macro(angle, shape) \
	macro(tc_position, shape) \
	macro(tc_size, shape) \
	macro(grid_size, shape) \
	macro(parallax_factor, shape) \
	macro(flags, shape) \
	macro(outline_color, shape)

#define GETTER_ONLY_OPS(shape, macro) \
	macro(position, shape) \
	macro(image_data, shape) \
	macro(radius, shape) \
	macro(src, shape) \
	macro(dst, shape) \
	macro(outline_size, shape)

#define GENERATE_GETTERS_FOR_SHAPE(shape) \
	COMMON_OPS(shape, GENERATE_GETTER) \
	GETTER_ONLY_OPS(shape, GENERATE_GETTER)

	SHAPE_FUNCS(GENERATE_GETTERS_FOR_SHAPE)

	#define GENERATE_SETTERS_FOR_SHAPE(shape) \
	COMMON_OPS(shape, GENERATE_SETTER)

	SHAPE_FUNCS(GENERATE_SETTERS_FOR_SHAPE)

	#define GENERATE_WRAPPERS(shape) \
	GENERATE_SETTER_ACTUAL(position2, shape, position) \
  GENERATE_SETTER_ACTUAL(position3_impl, shape, position) \
	static void CONCAT2(set_position3_, shape)(shape_t* s, const fan::vec3& pos) { \
		set_position(s, pos); \
		CONCAT2(set_position3_impl_, shape)(s, pos); \
	} \
	static fan::graphics::camera_t CONCAT2(get_camera_, shape)(const shape_t* s) { \
		return get_camera(const_cast<shape_t*>(s)); \
	} \
	static void CONCAT2(set_camera_, shape)(shape_t* s, fan::graphics::camera_t cam) { \
		set_camera(s, cam); \
	} \
	static fan::graphics::viewport_t CONCAT2(get_viewport_, shape)(const shape_t* s) { \
		return get_viewport(const_cast<shape_t*>(s)); \
	} \
	static void CONCAT2(set_viewport_, shape)(shape_t* s, fan::graphics::viewport_t vp) { \
		set_viewport(s, vp); \
	} \
	static fan::graphics::image_t CONCAT2(get_image_, shape)(const shape_t* s) { \
		return get_image(const_cast<shape_t*>(s)); \
	} \
	static void CONCAT2(set_image_, shape)(shape_t* s, fan::graphics::image_t img) { \
		set_image(s, img); \
	}

	SHAPE_FUNCS(GENERATE_WRAPPERS)

	template<typename cb_t, typename func_name_t>
	struct get_function_or_universal {
		static constexpr auto get() -> cb_t {
			if constexpr (requires { func_name_t::value; }) {
				return (cb_t)func_name_t::value;
			}
			else {
				return (cb_t)&make_universal<cb_t>::fn;
			}
		}
	};

#define DEFINE_FUNC_TAG(op, shape) \
	template <typename, typename = void> \
	struct shape##_has_##op : std::false_type {}; \
	template <typename T> \
	struct shape##_has_##op<T, std::void_t<decltype(std::declval<T>().op##_##shape())>> : std::true_type {}; \
	struct tag_##op##_##shape { \
		static constexpr auto value = &op##_##shape; \
	};

#define MAKE_TAG(shape, op, cb_type) DEFINE_FUNC_TAG(op, shape)
#define MAKE_TAGS(shape) VTABLE_OPS(MAKE_TAG, shape)

	SHAPE_FUNCS(MAKE_TAGS)

	#define TRY_DEFAULT(op, cb_type, shape) \
	(get_function_or_universal<cb_type, struct tag_##op##_##shape>::get())

	#define MAKE_ENTRY(shape, op, cb_type) base_functions_t<cb_type>{TRY_DEFAULT(op, cb_type, shape)},
	#define MAKE_VTABLE(shape) {VTABLE_OPS(MAKE_ENTRY, shape)}

	inline static shape_vtable_t vtables[shape_type_t::last] = {
		#define MAKE_ONE(shape) MAKE_VTABLE(shape),
			SHAPE_LIST(MAKE_ONE)
		#undef MAKE_ONE
	};

	struct functions_t {
	#define MAKE_MEMBER(shape, op, cb_type) base_functions_t<cb_type> op;
		VTABLE_OPS(MAKE_MEMBER, dummy)
		#undef MAKE_MEMBER
	};

	#define SHAPE_FUNCTION_OVERRIDE(shape, op, fn) (shape_functions_t::vtables[shape].op = shape_functions_t::base_functions_t{ fn })

	shape_functions_t() {
		{
			static auto f = +[] (const shape_t* shape) -> fan::graphics::image_data_t& {
				return (*fan::graphics::g_render_context_handle.image_list)[shape->get_image()];
			};
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::sprite, get_image_data, f);
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::unlit_sprite, get_image_data, f);
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::universal_image_renderer, get_image_data, f);
		}
		{ // polygon
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::polygon, get_position, +[] (const shape_t* shape) {
				auto ri = (polygon_t::ri_t*)shape->GetData(g_shapes->shaper);
				fan::vec3 position = 0;
				fan::opengl::core::get_glbuffer(
					(*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))),
					&position,
					ri->vbo.m_buffer,
					sizeof(position),
					sizeof(polygon_vertex_t) * 0 + fan::member_offset(&polygon_vertex_t::offset),
					ri->vbo.m_target
				);
				return position;
			});
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::polygon, set_position2, +[](shape_t* shape, const fan::vec2& position) {
				auto ri = (polygon_t::ri_t*)shape->GetData(g_shapes->shaper);
				ri->vao.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
				ri->vbo.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
				uint32_t vertex_count = ri->buffer_size / sizeof(polygon_vertex_t);
				for (uint32_t i = 0; i < vertex_count; ++i) {
					fan::opengl::core::edit_glbuffer(
						(*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))),
						ri->vbo.m_buffer,
						&position,
						sizeof(polygon_vertex_t) * i + fan::member_offset(&polygon_vertex_t::offset),
						sizeof(position),
						ri->vbo.m_target
					);
				}
			});
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::polygon, set_angle, +[] (shape_t* shape, const fan::vec3& angle) {
				auto ri = (polygon_t::ri_t*)shape->GetData(g_shapes->shaper);
				ri->vao.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
				ri->vbo.bind((*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))));
				uint32_t vertex_count = ri->buffer_size / sizeof(polygon_vertex_t);
				for (uint32_t i = 0; i < vertex_count; ++i) {
					fan::opengl::core::edit_glbuffer(
						(*static_cast<fan::opengl::context_t*>(static_cast<void*>(fan::graphics::g_render_context_handle))),
						ri->vbo.m_buffer,
						&angle,
						sizeof(polygon_vertex_t) * i + fan::member_offset(&polygon_vertex_t::angle),
						sizeof(angle),
						ri->vbo.m_target
					);
				}
			});
		}
		{ // light
			SHAPE_FUNCTION_OVERRIDE(shape_type_t::light, get_radius, +[] (const shape_t* shape) {
				return vtables[shape_type_t::light].get_size(shape).x;
			});
		}
	}

	auto& operator[](uint16_t shape){
		return vtables[shape];
	}
};

#undef GENERATE_SETTER_ACTUAL