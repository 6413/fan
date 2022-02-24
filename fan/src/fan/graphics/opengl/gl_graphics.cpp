#include <fan/graphics/renderer.hpp>

#define fan_assert_if_same_path_loaded_multiple_times 0

#if fan_renderer == fan_renderer_opengl

#include <fan/graphics/graphics.hpp>

#include <functional>
#include <numeric>

#include <fan/physics/collision/rectangle.hpp>
#include <fan/physics/collision/circle.hpp>

void fan::depth_test(bool value)
{
  if (value) {
    glEnable(GL_DEPTH_TEST);
  }
  else {
    glDisable(GL_DEPTH_TEST);
  }
}

// webp
fan_2d::graphics::image_t fan_2d::graphics::load_image(const std::string& path)
{
#if fan_assert_if_same_path_loaded_multiple_times

  static std::unordered_map<std::string, bool> existing_images;

  if (existing_images.find(path) != existing_images.end()) {
    fan::throw_error("image already existing " + path);
  }

  existing_images[path] = 0;

#endif

  fan_2d::graphics::image_t info = new fan_2d::graphics::image_T;

  auto image = fan::webp::load_image(path);

  glGenTextures(1, &info->texture);

  glBindTexture(GL_TEXTURE_2D, info->texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

  uintptr_t internal_format = 0, format = 0, type = 0;

  internal_format = GL_RGBA;
  format = GL_RGBA;
  type = GL_UNSIGNED_BYTE;

  info->size = image.size;

  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image.data);

  fan::webp::free_image(image.data);
  
  glGenerateMipmap(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);

  return info;
}

// webp
fan_2d::graphics::image_t fan_2d::graphics::load_image(const fan::webp::image_info_t& image_info)
{
  fan_2d::graphics::image_t info = new fan_2d::graphics::image_T;

  glGenTextures(1, &info->texture);

  glBindTexture(GL_TEXTURE_2D, info->texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

  uintptr_t internal_format = 0, format = 0, type = 0;

  internal_format = GL_RGBA;
  format = GL_RGBA;
  type = GL_UNSIGNED_BYTE;

  info->size = image_info.size;

  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_info.data);

  fan::webp::free_image(image_info.data);

  glGenerateMipmap(GL_TEXTURE_2D);

  glBindTexture(GL_TEXTURE_2D, 0);

  return info;
}

fan_2d::graphics::image_t fan_2d::graphics::load_image(const fan_2d::graphics::image_info_t& image_info)
{
  fan_2d::graphics::image_t info = new fan_2d::graphics::image_T;

  glGenTextures(1, &info->texture);

  glBindTexture(GL_TEXTURE_2D, info->texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

  uintptr_t internal_format = 0, format = 0, type = 0;

  internal_format = GL_RGBA;
  format = GL_RGBA;
  type = GL_UNSIGNED_BYTE;

  info->size = image_info.size;

  glTexImage2D(GL_TEXTURE_2D, 0, internal_format, info->size.x, info->size.y, 0, format, type, image_info.data);

  glGenerateMipmap(GL_TEXTURE_2D);

  glBindTexture(GL_TEXTURE_2D, 0);

  return info;
}

void fan_2d::graphics::rectangle::open(fan::opengl::context_t* context) 
{
  m_shader.open();

  m_shader.set_vertex(
    #include <fan/graphics/glsl/opengl/2D/rectangle.vs>
  );

  m_shader.set_fragment(
    #include <fan/graphics/glsl/opengl/2D/rectangle.fs>
  );

  m_shader.compile();

  m_glsl_buffer.open();
  m_glsl_buffer.init(m_shader.id, sizeof(fan_2d::graphics::rectangle::properties_t));
  m_queue_helper.open();
  m_draw_node_reference = fan::uninitialized;
}

void fan_2d::graphics::rectangle::close(fan::opengl::context_t* context) {

  m_glsl_buffer.close();
  m_queue_helper.close(context);
  m_shader.close();

  if (m_draw_node_reference == fan::uninitialized) {
    return;
  }

  context->disable_draw(m_draw_node_reference);
}

void fan_2d::graphics::rectangle::enable_draw(fan::opengl::context_t* context)
{
  m_draw_node_reference = context->enable_draw(this, [] (fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
}

void fan_2d::graphics::rectangle::disable_draw(fan::opengl::context_t* context)
{
#ifdef fan_debug == fan_debug_soft
  if (m_draw_node_reference == fan::uninitialized) {
    fan::throw_error("trying to disable unenabled draw call");
  }
#endif
 context->disable_draw(m_draw_node_reference);
}

void fan_2d::graphics::rectangle::push_back(fan::opengl::context_t* context, rectangle::properties_t properties)
{
  for (int i = 0; i < rectangle::vertex_count; i++) {
    m_glsl_buffer.push_ram_instance(&properties);
  }
  m_queue_helper.edit(
    context, 
    (this->size(context) - 1) * rectangle::vertex_count * m_glsl_buffer.m_element_size,
    (this->size(context)) * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
    &m_glsl_buffer
  );
}

void fan_2d::graphics::rectangle::insert(fan::opengl::context_t* context, uint32_t i, rectangle::properties_t properties)
{
  for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.insert_ram_instance(i * rectangle::vertex_count + j, &properties);
  }
  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
    (this->size(context)) * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
    & m_glsl_buffer
  );
}

void fan_2d::graphics::rectangle::draw(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
{
  m_glsl_buffer.draw(
    context, 
    m_shader, 
    begin == fan::uninitialized ? 0 : begin * rectangle::vertex_count, 
    end == fan::uninitialized ? m_glsl_buffer.m_buffer.size() / sizeof(f32_t) : end * rectangle::vertex_count
  );

#ifdef fan_debug == fan_debug_hard

  m_glsl_buffer.confirm_buffer();

#endif
}

void fan_2d::graphics::rectangle::erase(fan::opengl::context_t* context, uint32_t i)
{
  m_glsl_buffer.erase_instance(i * rectangle::vertex_count, 1, rectangle::vertex_count);

  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
    (this->size(context)) * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
    & m_glsl_buffer
  );
}

void fan_2d::graphics::rectangle::erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
{
  m_glsl_buffer.erase_instance(begin * rectangle::vertex_count, end - begin, rectangle::vertex_count);

  m_queue_helper.edit(
    context, 
    begin * rectangle::vertex_count * m_glsl_buffer.m_element_size,
    (this->size(context)) * rectangle::vertex_count * m_glsl_buffer.m_element_size,
    &m_glsl_buffer
  );
}

void fan_2d::graphics::rectangle::clear(fan::opengl::context_t* context)
{ 
  m_glsl_buffer.clear_ram();
  m_queue_helper.edit(
    context, 
    0,
    (this->size(context)) * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
    & m_glsl_buffer
  );
}

// 0 top left, 1 top right, 2 bottom left, 3 bottom right
fan_2d::graphics::rectangle_corners_t fan_2d::graphics::rectangle::get_corners(fan::opengl::context_t* context, uint32_t i) const
{
  auto position = this->get_position(context, i);
  auto size = this->get_size(context, i);

  fan::vec2 mid = position;

  auto corners = get_rectangle_corners_no_rotation(position, size);

  f32_t angle = -this->get_angle(context, i);

  fan::vec2 top_left = get_transformed_point(corners[0] - mid, angle) + mid;
  fan::vec2 top_right = get_transformed_point(corners[1] - mid, angle) + mid;
  fan::vec2 bottom_left = get_transformed_point(corners[2] - mid, angle) + mid;
  fan::vec2 bottom_right = get_transformed_point(corners[3] - mid, angle) + mid;

  return { top_left, top_right, bottom_left, bottom_right };
}

const fan::color fan_2d::graphics::rectangle::get_color(fan::opengl::context_t* context, uint32_t i) const
{
	return *(fan::color*)m_glsl_buffer.get_instance(i * rectangle::vertex_count, offset_color);
}

void fan_2d::graphics::rectangle::set_color(fan::opengl::context_t* context, uint32_t i, const fan::color& color)
{
	for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * rectangle::vertex_count + j,
      &color,
      offset_color,
      sizeof(rectangle::properties_t::color)
    );
  }

  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count * m_glsl_buffer.m_element_size + offset_color, 
    (i + 1) * (rectangle::vertex_count) * m_glsl_buffer.m_element_size - offset_color, 
    &m_glsl_buffer
  );
}

fan::vec2 fan_2d::graphics::rectangle::get_position(fan::opengl::context_t* context, uint32_t i) const
{
	return *(fan::vec2*)m_glsl_buffer.get_instance(i * rectangle::vertex_count, offset_position);
}

void fan_2d::graphics::rectangle::set_position(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position)
{
  for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * rectangle::vertex_count + j,
      &position,
      offset_position,
      sizeof(rectangle::properties_t::position)
    );
  }
  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count * m_glsl_buffer.m_element_size + offset_position, 
    (i + 1) * (rectangle::vertex_count) * m_glsl_buffer.m_element_size - offset_position, 
    &m_glsl_buffer
  );
}

fan::vec2 fan_2d::graphics::rectangle::get_size(fan::opengl::context_t* context, uint32_t i) const
{
	return *(fan::vec2*)m_glsl_buffer.get_instance(i * rectangle::vertex_count, offset_size);
}

void fan_2d::graphics::rectangle::set_size(fan::opengl::context_t* context, uint32_t i, const fan::vec2& size)
{
 for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * rectangle::vertex_count + j,
      &size,
      offset_size,
      sizeof(rectangle::properties_t::size)
    );
  }
  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count * m_glsl_buffer.m_element_size + offset_size, 
    (i + 1) * (rectangle::vertex_count) * m_glsl_buffer.m_element_size - offset_size, 
    &m_glsl_buffer
  );
}

f32_t fan_2d::graphics::rectangle::get_angle(fan::opengl::context_t* context, uint32_t i) const
{	
  return *(f32_t*)m_glsl_buffer.get_instance(i * rectangle::vertex_count, offset_angle);
}

// radians
void fan_2d::graphics::rectangle::set_angle(fan::opengl::context_t* context, uint32_t i, f32_t angle)
{
  f32_t a = fmod(angle, fan::math::pi * 2);

  for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * rectangle::vertex_count + j,
      &a,
      offset_angle,
      sizeof(rectangle::properties_t::angle)
    );
  }
  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count * m_glsl_buffer.m_element_size + offset_angle, 
    (i + 1) * (rectangle::vertex_count) * m_glsl_buffer.m_element_size - offset_angle, 
    &m_glsl_buffer
  );
}

fan::vec2 fan_2d::graphics::rectangle::get_rotation_point(fan::opengl::context_t* context, uint32_t i) const {
	return *(fan::vec2*)m_glsl_buffer.get_instance(i * rectangle::vertex_count, offset_rotation_point);
}
void fan_2d::graphics::rectangle::set_rotation_point(fan::opengl::context_t* context, uint32_t i, const fan::vec2& rotation_point) {
  for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * rectangle::vertex_count + j,
      &rotation_point,
      offset_rotation_point,
      sizeof(rectangle::properties_t::rotation_point)
    );
  }
  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count * m_glsl_buffer.m_element_size + offset_rotation_point, 
    (i + 1) * (rectangle::vertex_count) * m_glsl_buffer.m_element_size - offset_rotation_point, 
    &m_glsl_buffer
  );
}

fan::vec3 fan_2d::graphics::rectangle::get_rotation_vector(fan::opengl::context_t* context, uint32_t i) const {
	return *(fan::vec3*)m_glsl_buffer.get_instance(i * rectangle::vertex_count, offset_rotation_vector);
}
void fan_2d::graphics::rectangle::set_rotation_vector(fan::opengl::context_t* context, uint32_t i, const fan::vec3& rotation_vector) {
  for (int j = 0; j < rectangle::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * rectangle::vertex_count + j,
      &rotation_vector,
      offset_rotation_vector,
      sizeof(rectangle::properties_t::rotation_vector)
    );
  }
  m_queue_helper.edit(
    context, 
    i * rectangle::vertex_count * m_glsl_buffer.m_element_size + offset_rotation_point, 
    (i + 1) * (rectangle::vertex_count) * m_glsl_buffer.m_element_size - offset_rotation_point, 
    &m_glsl_buffer
  );
}

uint32_t fan_2d::graphics::rectangle::size(fan::opengl::context_t* context) const
{
	return m_glsl_buffer.m_buffer.size() / m_glsl_buffer.m_element_size / rectangle::vertex_count;
}

bool fan_2d::graphics::rectangle::inside(fan::opengl::context_t* context, uint32_t i, const fan::vec2& position) const {

	auto corners = get_corners(context, i);
	
	return fan_2d::collision::rectangle::point_inside(
    corners[0], 
    corners[1], 
    corners[2], 
    corners[3], 
    position
  );
}

//void fan_2d::graphics::rectangle0::open(fan::opengl::context_t* context, void* user_ptr, std::function<void(void*, uint64_t, uint32_t)> erase_cb)
//{
//	rectangle::open(context);
//	m_erase_cb = erase_cb;
//	m_user_ptr = user_ptr;
//  m_push_back_ids.open();
//}
//
//void fan_2d::graphics::rectangle0::push_back(fan::opengl::context_t* context, const properties_t& properties)
//{
//	m_push_back_ids.push_back(properties.id);
//	fan_2d::graphics::rectangle::push_back(context, properties);
//}
//
//void fan_2d::graphics::rectangle0::erase(fan::opengl::context_t* context, uint32_t i)
//{
//  fan::throw_error("recode");
//	if (i != this->size(context) - 1) {
//
//		//m_glsl_buffer.move_ram_buffer(i * rectangle::vertex_count, (this->size(context) - 1) * rectangle::vertex_count);
//
//		m_erase_cb(m_user_ptr, *(m_push_back_ids.end() - 1), i);
//
//		m_push_back_ids[i] = *(m_push_back_ids.end() - 1);
//		m_push_back_ids.pop_back();
//
//	  m_queue_helper.edit(
//      context, 
//      i * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
//      (this->size(context)) * rectangle::vertex_count *  m_glsl_buffer.m_element_size,
//      & m_glsl_buffer
//    );
//	}
//	else {
//		rectangle::erase(context, i);
//		m_push_back_ids.pop_back();
//	}
//}

std::array<fan_2d::graphics::line::src_dst_t, 4> fan_2d::graphics::line::create_box(fan::opengl::context_t* context, const fan::vec2& position, const fan::vec2& size)
{
	std::array<src_dst_t, 4> box;

	box[0].src = position;
	box[0].dst = position + fan::vec2(size.x, 0);

	box[1].src = position + fan::vec2(size.x, 0);
	box[1].dst = position + fan::vec2(size.x, size.y);

	box[2].src = position + fan::vec2(size.x, size.y);
	box[2].dst = position + fan::vec2(0, size.y);

	box[3].src = position + fan::vec2(0, size.y);
	box[3].dst = position;

	return box;
}

void fan_2d::graphics::line::push_back(fan::opengl::context_t* context, const fan::vec2& src, const fan::vec2& dst, const fan::color& color, f32_t thickness)
{

	line_instance.emplace_back(line_instance_t{
		(src + ((dst - src) / 2)),
		dst,
		thickness
	});

	// - fan::vec2(0, 0.5 * thickness)

	rectangle::properties_t property;
	property.position = src + ((dst - src) / 2);
	property.size = fan::vec2((dst - src).length(), thickness) / 2;
	property.angle = -fan::math::aim_angle(src, dst);
	property.color = color;

	rectangle::push_back(context, property);
}

fan::vec2 fan_2d::graphics::line::get_src(fan::opengl::context_t* context, uint32_t i) const
{
	return line_instance[i].src;
}

fan::vec2 fan_2d::graphics::line::get_dst(fan::opengl::context_t* context, uint32_t i) const
{
	return line_instance[i].dst;
}

void fan_2d::graphics::line::set_line(fan::opengl::context_t* context, uint32_t i, const fan::vec2& src, const fan::vec2& dst)
{
	const auto thickness = this->get_thickness(context, i);

	rectangle::set_position(context, i, src + ((dst - src) / 2));
	rectangle::set_size(context, i, fan::vec2((dst - src).length(), thickness) / 2);
	rectangle::set_angle(context, i, -fan::math::aim_angle(src, dst));
}

f32_t fan_2d::graphics::line::get_thickness(fan::opengl::context_t* context, uint32_t i) const
{
	return line_instance[i].thickness;
}

void fan_2d::graphics::line::set_thickness(fan::opengl::context_t* context, uint32_t i, const f32_t thickness)
{

	const auto src = line_instance[i].src;
	const auto dst = line_instance[i].dst;

	const auto new_src = src;
	const auto new_dst = fan::vec2((dst - src).length(), thickness);

	line_instance[i].thickness = thickness;

	rectangle::set_position(context, i, new_src + ((new_dst - new_src) / 2));
	rectangle::set_size(context, i, new_dst / 2);
}

void fan_2d::graphics::sprite::open(fan::opengl::context_t* context) {

  m_shader.open();
  m_store_sprite.open();

  m_shader.set_vertex(
    #include <fan/graphics/glsl/opengl/2D/sprite.vs>
  );

  m_shader.set_fragment(
    #include <fan/graphics/glsl/opengl/2D/sprite.fs>
  );

  m_shader.compile();

  m_glsl_buffer.open();
  m_glsl_buffer.init(m_shader.id, 
    sizeof(fan_2d::graphics::rectangle::properties_t) +
    sizeof(fan::vec2) // texture_coordinates
  );
  m_queue_helper.open();

  m_draw_node_reference = fan::uninitialized;
}

void fan_2d::graphics::sprite::close(fan::opengl::context_t* context) {
  m_store_sprite.close();

  rectangle::close(context);
}

void fan_2d::graphics::sprite::push_back(fan::opengl::context_t* context, const sprite::properties_t& properties)
{
	sprite::rectangle::properties_t property;
	property.position = properties.position;
	property.size = properties.size;
	property.angle = properties.angle;
	property.rotation_point = properties.rotation_point;
	property.rotation_vector = properties.rotation_vector;
	property.color = properties.color;

	rectangle::push_back(context, property);

	std::array<fan::vec2, 6> texture_coordinates = {
		properties.texture_coordinates[0],
		properties.texture_coordinates[1],
		properties.texture_coordinates[2],

		properties.texture_coordinates[2],
		properties.texture_coordinates[3],
		properties.texture_coordinates[0]
	};

  for (uint32_t j = 0; j < sprite::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      (rectangle::size(context) - 1) * sprite::vertex_count + j,
      &texture_coordinates[j],
      offset_texture_coordinates,
      sizeof(fan::vec2)
    );
  }

  m_store_sprite.resize(m_store_sprite.size() + 1);

	if (m_store_sprite.size() < 2) {
    m_store_sprite[m_store_sprite.size() - 1].m_switch_texture = 0;
	}
	else if (m_store_sprite.size() && m_store_sprite[m_store_sprite.size() - 2].m_texture != properties.image->texture) {
    m_store_sprite[m_store_sprite.size() - 1].m_switch_texture = this->size(context) - 1;
	}
  else {
    m_store_sprite[m_store_sprite.size() - 1].m_switch_texture = fan::uninitialized;
  }

  m_store_sprite[m_store_sprite.size() - 1].m_texture = properties.image->texture;
}

void fan_2d::graphics::sprite::insert(fan::opengl::context_t* context, uint32_t i, const sprite::properties_t& properties)
{
	sprite::rectangle::properties_t property;
	property.position = properties.position;
	property.size = properties.size;
	property.rotation_point = property.position;
	property.rotation_vector = properties.rotation_vector;

	fan_2d::graphics::rectangle::insert(context, i, property);
	 
	std::array<fan::vec2, 6> texture_coordinates = {
		properties.texture_coordinates[0],
		properties.texture_coordinates[1],
		properties.texture_coordinates[2],

		properties.texture_coordinates[2],
		properties.texture_coordinates[3],
		properties.texture_coordinates[0]
	};

	for (uint32_t j = 0; j < sprite::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      i * sprite::vertex_count + j,
      &texture_coordinates[j],
      offset_texture_coordinates,
      sizeof(fan::vec2)
    );
  }

  store_sprite_t sst;
  sst.m_texture = properties.image->texture;

	m_store_sprite.insert(m_store_sprite.begin() + i, sst);

	regenerate_texture_switch();
}

void fan_2d::graphics::sprite::reload_sprite(fan::opengl::context_t* context, uint32_t i, fan_2d::graphics::image_t image)
{
	m_store_sprite[i].m_texture = image->texture;
}

void fan_2d::graphics::sprite::enable_draw(fan::opengl::context_t* context)
{
#ifdef fan_debug == fan_debug_soft
  if (m_draw_node_reference != fan::uninitialized) {
    fan::throw_error("trying to call enable_draw twice");
  }
#endif

	m_draw_node_reference = context->enable_draw(this, [] (fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
}

void fan_2d::graphics::sprite::disable_draw(fan::opengl::context_t* context)
{
  #ifdef fan_debug == fan_debug_soft
    if (m_draw_node_reference == fan::uninitialized) {
      fan::throw_error("trying to disable unenabled draw call");
    }
  #endif
  context->disable_draw(m_draw_node_reference);
}

std::array<fan::vec2, 4> fan_2d::graphics::sprite::get_texture_coordinates(fan::opengl::context_t* context, uint32_t i)
{
  fan::vec2* coordinates = (fan::vec2*)m_glsl_buffer.get_instance(i * sprite::vertex_count, offset_texture_coordinates);

	return std::array<fan::vec2, 4>{
		coordinates[0],
		coordinates[1],
		coordinates[2],
		coordinates[5]
	};
}

void fan_2d::graphics::sprite::set_texture_coordinates(fan::opengl::context_t* context, uint32_t i, const std::array<fan::vec2, 4>& texture_coordinates)
{
	std::array<fan::vec2, 6> tc = {
		texture_coordinates[0],
		texture_coordinates[1],
		texture_coordinates[2],

		texture_coordinates[2],
		texture_coordinates[3],
		texture_coordinates[0]
	};

  for (uint32_t j = 0; j < sprite::vertex_count; j++) {
    m_glsl_buffer.edit_ram_instance(
      (rectangle::size(context) - 1) * sprite::vertex_count + j,
      &tc[j],
      offset_texture_coordinates,
      sizeof(fan::vec2)
    );
  }
  m_queue_helper.edit(
    context, 
    i * sprite::vertex_count * m_glsl_buffer.m_element_size + offset_texture_coordinates, 
    (i + 1) * (sprite::vertex_count) * m_glsl_buffer.m_element_size - offset_texture_coordinates, 
    &m_glsl_buffer
  );
}

void fan_2d::graphics::sprite::draw(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
{
	m_shader.use();

	for (int i = 0; i < m_store_sprite.size(); i++) {

    if (m_store_sprite[i].m_switch_texture == fan::uninitialized) {
      continue;
    }

		m_shader.set_int("texture_sampler0", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite[i].m_switch_texture].m_texture);

		if (i == m_store_sprite.size() - 1) {
			fan_2d::graphics::rectangle::draw(context, m_store_sprite[i].m_switch_texture, end == fan::uninitialized ? this->size(context) : end);
		}
		else {
			fan_2d::graphics::rectangle::draw(context, m_store_sprite[i].m_switch_texture, m_store_sprite[i + 1].m_switch_texture);
		}
	}

}

void fan_2d::graphics::sprite::erase(fan::opengl::context_t* context, uint32_t i)
{
	rectangle::erase(context, i);

	m_store_sprite.erase(i);

	regenerate_texture_switch();
}

void fan_2d::graphics::sprite::erase(fan::opengl::context_t* context, uint32_t begin, uint32_t end)
{
	rectangle::erase(context, begin, end);

  m_store_sprite.erase(begin, end);

	regenerate_texture_switch();
}

void fan_2d::graphics::sprite::clear(fan::opengl::context_t* context)
{
	rectangle::clear(context);

	m_store_sprite.clear();
}

// todo remove
void fan_2d::graphics::sprite::regenerate_texture_switch()
{

	for (int i = 0; i < m_store_sprite.size(); i++) {
		if (i == 0) {
			m_store_sprite[i].m_switch_texture = 0;
		}
		else if (m_store_sprite.size() && m_store_sprite[i].m_texture != m_store_sprite[i - 1].m_texture) {
      m_store_sprite[i].m_switch_texture = i;
		}
    else {
      m_store_sprite[i].m_switch_texture = fan::uninitialized;
    }
	}
}

void fan_2d::graphics::yuv420p_renderer::open(fan::opengl::context_t* context) {
  m_shader.set_vertex(
    #include <fan/graphics/glsl/opengl/2D/yuv420p_renderer.vs>
  );

  m_shader.set_fragment(
    #include <fan/graphics/glsl/opengl/2D/yuv420p_renderer.fs>
  );

	m_shader.compile();

  // from sprite open
	m_glsl_buffer.open();
  m_glsl_buffer.init(m_shader.id, 
    sizeof(fan_2d::graphics::rectangle::properties_t) +
    sizeof(fan::vec2) +// texture_coordinates
    sizeof(uint32_t) + // render op code 0
    sizeof(uint32_t)   // render op code 1
  );
  m_queue_helper.open();
}

void fan_2d::graphics::yuv420p_renderer::push_back(fan::opengl::context_t* context, const yuv420p_renderer::properties_t& properties) {

  m_store_sprite.resize(m_store_sprite.size() + 3);


	glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 3].m_texture);
	glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 2].m_texture);
	glGenTextures(1, &m_store_sprite[m_store_sprite.size() - 1].m_texture);

	glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 3].m_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x, properties.pixel_data.size.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[0]);
	//glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 2].m_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[1]);
//	glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glBindTexture(GL_TEXTURE_2D, m_store_sprite[m_store_sprite.size() - 1].m_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, properties.pixel_data.size.x / 2, properties.pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, properties.pixel_data.pixels[2]);

	//glGenerateMipmap(GL_TEXTURE_2D);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glBindTexture(GL_TEXTURE_2D, 0);

	sprite::sprite::properties_t property;
	property.position = properties.position;
	property.size = properties.size;
	property.angle = properties.angle;
	property.rotation_point = properties.rotation_point;
	property.rotation_vector = properties.rotation_vector;

	fan_2d::graphics::sprite::push_back(context, property);

	image_size.emplace_back(properties.pixel_data.size);
}

void fan_2d::graphics::yuv420p_renderer::reload_pixels(fan::opengl::context_t* context, uint32_t i, const fan_2d::graphics::pixel_data_t& pixel_data) {
	glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3].m_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x, pixel_data.size.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[0]);

//	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 1].m_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[1]);

	//glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 2].m_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, fan_2d::graphics::image_load_properties::visual_output);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fan_2d::graphics::image_load_properties::filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fan_2d::graphics::image_load_properties::filter);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, pixel_data.size.x / 2, pixel_data.size.y / 2, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixel_data.pixels[2]);

	//glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

	image_size[i] = pixel_data.size;
}

fan::vec2ui fan_2d::graphics::yuv420p_renderer::get_image_size(fan::opengl::context_t* context, uint32_t i) const
{
	return this->image_size[i];
}

void fan_2d::graphics::yuv420p_renderer::enable_draw(fan::opengl::context_t* context)
{
  m_draw_node_reference = context->enable_draw(this, [] (fan::opengl::context_t* c, void* d) { ((decltype(this))d)->draw(c); });
}

void fan_2d::graphics::yuv420p_renderer::draw(fan::opengl::context_t* context)
{
	m_shader.use();

	m_shader.set_int("sampler_y", 0);
	m_shader.set_int("sampler_u", 1);
	m_shader.set_int("sampler_v", 2);

	for (int i = 0; i < rectangle::size(context); i++) {
		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3].m_texture);

		glActiveTexture(GL_TEXTURE0 + 1);
		glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 1].m_texture);

		glActiveTexture(GL_TEXTURE0 + 2);
		glBindTexture(GL_TEXTURE_2D, m_store_sprite[i * 3 + 2].m_texture);

		fan_2d::graphics::rectangle::draw(context, i);
	}
}

//void fan_3d::graphics::add_camera_rotation_callback(fan::camera* camera) {
//	camera->m_window->add_mouse_move_callback(std::function<void(fan::window* window, const fan::vec2i& position)>(std::bind(&fan::camera::rotate_camera, camera, 0)));
//}
//
////fan_3d::graphics::rectangle_vector::rectangle_vector(fan::camera* camera, const fan::color& color, uintptr_t block_size)
////	: basic_shape(camera, fan::shader_t(fan_3d::graphics::shader_paths::shape_vector_vs, fan_3d::graphics::shader_paths::shape_vector_fs)),
////	block_size(block_size)
////{
////	glBindVertexArray(m_vao);
////
////	rectangle_vector::basic_shape_color_vector::initialize_buffers(true);
////	rectangle_vector::basic_shape_position::initialize_buffers(true);
////	rectangle_vector::basic_shape_size::initialize_buffers(true);
////
////	glBindBuffer(GL_ARRAY_BUFFER, 0);
////	glBindVertexArray(0);
////	//TODO
////	//generate_textures(path, block_size);
////}
//
////fan_3d::graphics::model_loader::model_loader(const std::string& path, const fan::vec3& size) {
////	load_model(path, size);
////}
//
////void fan_3d::graphics::model_loader::load_model(const std::string& path, const fan::vec3& size) {
////	Assimp::Importer importer;
////	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
////
////	if (scene == nullptr || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || scene->mRootNode == nullptr) {
////		std::cout << "assimp error: " << importer.GetErrorString() << '\n';
////		return;
////	}
////
////	directory = path.substr(0, path.find_last_of('/'));
////
////	process_node(scene->mRootNode, scene, size);
////}
//
////void fan_3d::graphics::model_loader::process_node(aiNode* node, const aiScene* scene, const fan::vec3& size) {
////	for (GLuint i = 0; i < node->mNumMeshes; i++) {
////		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
////
////		meshes.emplace_back(process_mesh(mesh, scene, size));
////	}
////
////	for (GLuint i = 0; i < node->mNumChildren; i++) {
////		process_node(node->mChildren[i], scene, size);
////	}
////}
//
////fan_3d::graphics::model_mesh fan_3d::graphics::model_loader::process_mesh(aiMesh* mesh, const aiScene* scene, const fan::vec3& size) {
////	std::vector<mesh_vertex> vertices;
////	std::vector<GLuint> indices;
////	std::vector<mesh_texture> textures;
////
////	for (GLuint i = 0; i < mesh->mNumVertices; i++)
////	{
////		mesh_vertex vertex;
////		fan::vec3 vector;
////
////		vector.x = mesh->mVertices[i].x / 2 * size.x;
////		vector.y = mesh->mVertices[i].y / 2 * size.y;
////		vector.z = mesh->mVertices[i].z / 2 * size.z;
////		vertex.position = vector;
////		if (mesh->mNormals != nullptr) {
////			vector.x = mesh->mNormals[i].x;
////			vector.y = mesh->mNormals[i].y;
////			vector.z = mesh->mNormals[i].z;
////			vertex.normal = vector;
////		}
////		else {
////			vertex.normal = fan::vec3();
////		}
////
////		if (mesh->mTextureCoords[0]) {
////			fan::vec2 vec;
////			vec.x = mesh->mTextureCoords[0][i].x;
////			vec.y = mesh->mTextureCoords[0][i].y;
////			vertex.texture_coordinates = vec;
////		}
////
////		vertices.emplace_back(vertex);
////	}
////
////	for (GLuint i = 0; i < mesh->mNumFaces; i++) {
////		aiFace face = mesh->mFaces[i];
////		for (uintptr_t j = 0; j < face.mNumIndices; j++) {
////			indices.emplace_back(face.mIndices[j]);
////		}
////	}
////
////	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
////
////	std::vector<mesh_texture> diffuseMaps = this->load_material_textures(material, aiTextureType_DIFFUSE, "texture_diffuse");
////	textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
////
////	std::vector<mesh_texture> specularMaps = this->load_material_textures(material, aiTextureType_SPECULAR, "texture_specular");
////	textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
////
////	if (textures.empty()) {
////		mesh_texture m_texture;
////		unsigned int texture_id;
////		glGenTextures(1, &texture_id);
////
////		aiColor4D color(0.f, 0.f, 0.f, 0.f);
////		aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color);
////		std::vector<unsigned char> pixels;
////		pixels.emplace_back(color.r * 255.f);
////		pixels.emplace_back(color.g * 255.f);
////		pixels.emplace_back(color.b * 255.f);
////		pixels.emplace_back(color.a * 255.f);
////
////		glBindTexture(GL_TEXTURE_2D, texture_id);
////		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
////		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
////		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
////		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
////
////		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
////		glGenerateMipmap(GL_TEXTURE_2D);
////
////		glBindTexture(GL_TEXTURE_2D, 0);
////
////		m_texture.id = texture_id;
////		textures.emplace_back(m_texture);
////		textures_loaded.emplace_back(m_texture);
////	}
////	return model_mesh(vertices, indices, textures);
////}
//
////std::vector<fan_3d::graphics::mesh_texture> fan_3d::graphics::model_loader::load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name) {
////	std::vector<mesh_texture> textures;
////
////	for (uintptr_t i = 0; i < mat->GetTextureCount(type); i++) {
////		aiString a_str;
////		mat->GetTexture(type, i, &a_str);
////		bool skip = false;
////		for (const auto& j : textures_loaded) {
////			if (j.path == a_str) {
////				textures.emplace_back(j);
////				skip = true;
////				break;
////			}
////		}
////
////		if (!skip) {
////			mesh_texture m_texture;
////			m_texture.id = load_texture(a_str.C_Str(), directory, false);
////			m_texture.type = type_name;
////			m_texture.path = a_str;
////			textures.emplace_back(m_texture);
////			textures_loaded.emplace_back(m_texture);
////		}
////	}
////	return textures;
////}
//
////fan_3d::graphics::model::model(fan::camera* camera) : model_loader("", fan::vec3()), m_camera(camera), m_shader(fan_3d::graphics::shader_paths::model_vs, fan_3d::graphics::shader_paths::model_fs) {}
////
////fan_3d::graphics::model::model(fan::camera* camera, const std::string& path, const fan::vec3& position, const fan::vec3& size)
////	: model_loader(path, size / 2.f), m_camera(camera), m_shader(fan_3d::graphics::shader_paths::model_vs, fan_3d::graphics::shader_paths::model_fs),
////	m_position(position), m_size(size)
////{
////	for (uintptr_t i = 0; i < this->meshes.size(); i++) {
////		glBindVertexArray(this->meshes[i].vao);
////	}
////	glBindVertexArray(0);
////}
////
////void fan_3d::graphics::model::draw() {
////
////	fan::mat4 model(1);
////	model = translate(model, get_position());
////	model = scale(model, get_size());
////
////	this->m_shader.use();
////
////	fan::mat4 projection(1);
////	projection = fan::perspective<fan::mat4>(fan::radians(90.f), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 1000.0f);
////
////	fan::mat4 view(m_camera->get_view_matrix());
////
////	this->m_shader.set_int("texture_sampler", 0);
////	this->m_shader.set_mat4("projection", projection);
////	this->m_shader.set_mat4("view", view);
////	this->m_shader.set_vec3("light_position", m_camera->get_position());
////	this->m_shader.set_vec3("view_position",m_camera->get_position());
////	this->m_shader.set_vec3("light_color", fan::vec3(1, 1, 1));
////	this->m_shader.set_int("texture_diffuse", 0);
////	this->m_shader.set_mat4("model", model);
////
////	//_Shader.set_vec3("sky_color", fan::vec3(220.f / 255.f, 219.f / 255.f, 223.f / 255.f));
////	glActiveTexture(GL_TEXTURE0);
////	glBindTexture(GL_TEXTURE_2D, this->textures_loaded[0].id);
////
////	glDepthFunc(GL_LEQUAL);
////	for (uintptr_t i = 0; i < this->meshes.size(); i++) {
////		glBindVertexArray(this->meshes[i].vao);
////		glDrawElementsInstanced(GL_TRIANGLES, (GLsizei)this->meshes[i].indices.size(), GL_UNSIGNED_INT, 0, 1);
////	}
////	glDepthFunc(GL_LESS);
////
////	glBindVertexArray(0);
////}
////
////fan::vec3 fan_3d::graphics::model::get_position()
////{
////	return this->m_position;
////}
////
////void fan_3d::graphics::model::set_position(const fan::vec3& position)
////{
////	this->m_position = position;
////}
////
////fan::vec3 fan_3d::graphics::model::get_size()
////{
////	return this->m_size;
////}
////
////void fan_3d::graphics::model::set_size(const fan::vec3& size)
////{
////	this->m_size = size;
////}
//
////fan::vec3 line_triangle_intersection(const fan::vec3& ray_begin, const fan::vec3& ray_end, const fan::vec3& p0, const fan::vec3& p1, const fan::vec3& p2) {
////
////	const auto lab = (ray_begin + ray_end) - ray_begin;
////
////	const auto p01 = p1 - p0;
////	const auto p02 = p2 - p0;
////
////	const auto normal = fan::math::cross(p01, p02);
////
////	const auto t = fan_3d::math::dot(normal, ray_begin - p0) / fan_3d::math::dot(-lab, normal);
////	const auto u = fan_3d::math::dot(fan::math::cross(p02, -lab), ray_begin - p0) / fan_3d::math::dot(-lab, normal);
////	const auto v = fan_3d::math::dot(fan::math::cross(-lab, p01), ray_begin - p0) / fan_3d::math::dot(-lab, normal);
////
////	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
////		return ray_begin + lab * t;
////	}
////
////	return INFINITY;
////
////}
//
////fan::vec3 fan_3d::graphics::line_triangle_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 3, 3>& triangle) {
////
////	const auto lab = (line[0] + line[1]) - line[0];
////
////	const auto p01 = triangle[1] - triangle[0];
////	const auto p02 = triangle[2] - triangle[0];
////
////	const auto normal = fan::math::cross(p01, p02);
////
////	const auto t = fan_3d::math::dot(normal, line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
////	const auto u = fan_3d::math::dot(fan::math::cross(p02, -lab), line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
////	const auto v = fan_3d::math::dot(fan::math::cross(-lab, p01), line[0] - triangle[0]) / fan_3d::math::dot(-lab, normal);
////
////	if (t >= 0 && t <= 1 && u >= 0 && u <= 1 && v >= 0 && v <= 1 && (u + v) <= 1) {
////		return line[0] + lab * t;
////	}
////
////	return INFINITY;
////}
////
////fan::vec3 fan_3d::graphics::line_plane_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 4, 3>& square) {
////	const fan::da_t<f32_t, 3> plane_normal = fan::math::normalize_no_sqrt(fan::math::cross(square[3] - square[2], square[0] - square[2]));
////	const f32_t nl_dot(math::dot(plane_normal, line[1]));
////
////	if (!nl_dot) {
////		return fan::vec3(INFINITY);
////	}
////
////	const f32_t ray_length = fan_3d::math::dot(square[2] - line[0], plane_normal) / nl_dot;
////	if (ray_length <= 0) {
////		return fan::vec3(INFINITY);
////	}
////	if (fan::math::custom_pythagorean_no_sqrt(fan::vec3(line[0]), fan::vec3(line[0] + line[1])) < ray_length) {
////		return fan::vec3(INFINITY);
////	}
////	const fan::vec3 intersection(line[0] + line[1] * ray_length);
////
////	auto result = fan_3d::math::dot((square[2] - line[0]), plane_normal);
////
////	if (!result) {
////		fan::print("on plane");
////	}
////
////	if (intersection[1] >= square[3][1] && intersection[1] <= square[0][1] &&
////		intersection[2] >= square[3][2] && intersection[2] <= square[0][2])
////	{
////		return intersection;
////	}
////	return fan::vec3(INFINITY);
////}
//
//#endif
//
////fan_3d::graphics::model_t::model_t(fan::camera* camera)
////	: m_camera(camera), m_queue_helper(camera->m_window)
////{
////	m_shader.set_vertex(
////		#include <fan/graphics/glsl/opengl/3D/model.vs>
////	);
////
////	m_shader.set_fragment(
////		#include <fan/graphics/glsl/opengl/3D/model.fs>
////	);
////
////	m_shader.compile();
////
////	fan::bind_vao(vao_t::m_buffer_object, [&] {
////		vertices_t::initialize_buffers(m_shader.id, vertex_layout_location, false, vertices_t::value_type::size());
////		normals_t::initialize_buffers(m_shader.id, normal_layout_location, false, normals_t::value_type::size());
////	
////		glGenBuffers(1, &m_ebo);
////		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
////		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, 0, GL_STATIC_DRAW); 
////	});
////
////}
////
////fan_3d::graphics::model_t::~model_t()
////{
////	m_queue_helper.reset();
////
////	if (m_draw_index != -1) {
////		m_camera->m_window->erase_draw_call(m_draw_index);
////		m_draw_index = -1;
////	}
////}
////
////void fan_3d::graphics::model_t::parse_model(const fan_3d::graphics::model_t::properties_t& properties, uint32_t& current_index, uint32_t& max_index, aiNode *node, const aiScene *scene)
////{
////  // process each mesh located at the current node
////  for(unsigned int i = 0; i < node->mNumMeshes; i++)
////  {
////      aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
////       
////      // walk through each of the mesh's vertices
////      for(unsigned int i = 0; i < mesh->mNumVertices; i++)
////      {
////          fan::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
////          // positions
////          vector.x = mesh->mVertices[i].x;
////          vector.y = mesh->mVertices[i].y;
////          vector.z = mesh->mVertices[i].z;
////
////					vertices_t::push_back(vector);
////					vector = 0;
////          
////          if (mesh->HasNormals())
////          {
////            vector.x = mesh->mNormals[i].x;
////            vector.y = mesh->mNormals[i].y;
////            vector.z = mesh->mNormals[i].z;
////          }
////					normals_t::push_back(vector);
////          //// texture coordinates
////          //if(mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
////          //{
////          //    // tangent
////          //    vector.x = mesh->mTangents[i].x;
////          //    vector.y = mesh->mTangents[i].y;
////          //    vector.z = mesh->mTangents[i].z;
////          //    // bitangent
////          //    vector.x = mesh->mBitangents[i].x;
////          //    vector.y = mesh->mBitangents[i].y;
////          //    vector.z = mesh->mBitangents[i].z;
////          //}
////      }
////
////      for(unsigned int i = 0; i < mesh->mNumFaces; i++)
////      {
////          aiFace face = mesh->mFaces[i];
////					for (unsigned int j = 0; j < face.mNumIndices; j++) {
////              m_indices.push_back(max_index + face.mIndices[j]);        
////							current_index = std::max(current_index, face.mIndices[j]);
////					}
////      }
////  }
////
////  for(unsigned int i = 0; i < node->mNumChildren; i++)
////  {
////			max_index += current_index + !!current_index;
////			current_index = 0;
////      parse_model(properties, current_index, max_index, node->mChildren[i], scene);
////  }
////}
////
////void fan_3d::graphics::model_t::push_back(const properties_t& properties)
////{
////	Assimp::Importer import;
////  const aiScene *scene = import.ReadFile(properties.path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);	
////	
////  if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) 
////  {
////		fan::throw_error("error loading model \"" + properties.path + "\"");
////  }
////
////	uint32_t c=0, m=0;
////
////	auto prev = m_indices.size();
////
////	parse_model(properties, c, m, scene->mRootNode, scene);
////
////	fan::mat4 mat(1);
////
////	mat = mat.translate(properties.position);
////	mat = mat.scale(properties.size);
////	mat = mat.rotate(properties.angle, properties.rotation_vector);
////
////	m_model.push_back(mat);
////
////	m_queue_helper.write([&] {
////
////		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
////		glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(decltype(m_indices)::value_type), m_indices.data(), GL_STATIC_DRAW);
////		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
////
////		vertices_t::write_data();
////		normals_t::write_data();
////
////		m_queue_helper.on_write(m_camera->m_window);
////	});
////}
////
////fan::vec3 fan_3d::graphics::model_t::get_position(uint32_t i) const
////{
////	return 0;
////}
////
////void fan_3d::graphics::model_t::set_position(uint32_t i, const fan::vec3& position)
////{
////
////}
////
////fan::vec3 fan_3d::graphics::model_t::get_size(uint32_t i) const
////{
////	return 0;
////}
////
////void fan_3d::graphics::model_t::set_size(uint32_t i, const fan::vec3& size)
////{
////
////}
////
////f32_t fan_3d::graphics::model_t::get_angle(uint32_t i) const
////{
////	return 0;
////}
////
////void fan_3d::graphics::model_t::set_angle(uint32_t i, f32_t angle)
////{
////	m_model[i] = m_model[i].rotate(angle, fan::vec3(0, 0, 1));
////}
////
////fan::vec3 fan_3d::graphics::model_t::get_rotation_vector(uint32_t i) const
////{
////	return 0;
////}
////
////void fan_3d::graphics::model_t::set_rotation_vector(uint32_t i, const fan::vec3& rotation_vector)
////{
////
////}
////
////void fan_3d::graphics::model_t::enable_draw()
////{
////	if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
////		m_draw_index = m_camera->m_window->push_draw_call(this, [addr = (uint64_t)this]{
////			((model_t*)(addr))->draw();
////		});
////	}
////	else {
////		m_camera->m_window->edit_draw_call(m_draw_index, this, [addr = (uint64_t)this] {
////			((model_t*)(addr))->draw();
////		});
////	}
////}
////
////void fan_3d::graphics::model_t::disable_draw()
////{
////	if (m_draw_index == -1) {
////		return;
////	}
////
////	m_camera->m_window->erase_draw_call(m_draw_index);
////	m_draw_index = -1;
////}
////
////void fan_3d::graphics::model_t::draw()
////{
////	glEnable(GL_DEPTH_TEST);
////
////	m_shader.use();
////
////	fan::mat4 projection(1);
////	projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.0), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 1000.0f);
////
////	fan::mat4 view(m_camera->get_view_matrix());
////
////	m_shader.set_mat4("projection", projection);
////	m_shader.set_mat4("view", view);
////
////	m_shader.set_mat4("models", &m_model[0][0][0], m_model.size());
////
////	fan::bind_vao(vao_t::m_buffer_object, [&] {
////		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
////		glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
////	});
////
////	glDisable(GL_DEPTH_TEST);
////}
//
//static bool fan_3d::graphics::animation::read_skeleton(animation::joint_t& joint, aiNode* node, std::unordered_map<std::string, std::pair<int, fan::mat4>>& boneInfoTable)
//{
//
//	if (boneInfoTable.find(node->mName.C_Str()) != boneInfoTable.end()) { // if node is actually a bone
//		joint.name = node->mName.C_Str();
//		joint.id = boneInfoTable[joint.name].first;
//		joint.offset = boneInfoTable[joint.name].second;
//
//		for (int i = 0; i < node->mNumChildren; i++) {
//			animation::joint_t child;
//			read_skeleton(child, node->mChildren[i], boneInfoTable);
//			joint.children.push_back(child);
//		}
//		return true;
//	}
//	else { // find bones in children
//		for (int i = 0; i < node->mNumChildren; i++) {
//			if (read_skeleton(joint, node->mChildren[i], boneInfoTable)) {
//				return true;
//			}
//
//		}
//	}
//	return false;
//}
//
//static void fan_3d::graphics::animation::load_model(const aiScene* scene, std::vector<animation::vertex_t>& verticesOutput, std::vector<uint32_t>& indicesOutput, animation::joint_t& skeletonOutput, uint32_t& nBoneCount)
//{
//	aiMesh* mesh = scene->mMeshes[0];
//
//	uint32_t max_index = 0;
//	uint32_t current_index = 0;
//
//	for (int j = 0; j < scene->mNumMeshes; j++) {
//		//load position, normal, uv
//		for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
//			//process position 
//			fan_3d::graphics::animation::vertex_t vertex;
//			fan::vec3 vector;
//			vector.x = mesh->mVertices[i].x;
//			vector.y = mesh->mVertices[i].y;
//			vector.z = mesh->mVertices[i].z;
//			vertex.position = vector;
//			//process normal
//			vector.x = mesh->mNormals[i].x;
//			vector.y = mesh->mNormals[i].y;
//			vector.z = mesh->mNormals[i].z;
//			vertex.normal = vector;
//			//process uv
//			fan::vec2 vec;
//			vec.x = mesh->mTextureCoords[0][i].x;
//			vec.y = mesh->mTextureCoords[0][i].y;
//			vertex.uv = vec;
//
//			vertex.bone_ids = fan::vec4i(0);
//			vertex.bone_weights = fan::vec4(0.0f);
//
//			verticesOutput.push_back(vertex);
//		}
//
//			//load indices
//		for (int i = 0; i < mesh->mNumFaces; i++) {
//			aiFace& face = mesh->mFaces[i];
//			for (uint32_t j = 0; j < face.mNumIndices; j++) {
//				indicesOutput.push_back(max_index + face.mIndices[j]);
//				current_index = std::max(current_index, face.mIndices[j]);
//			}
//		}
//
//		max_index += current_index;
//		current_index = 0;
//
//		mesh = scene->mMeshes[j + 1];
//	}
//
//	mesh = scene->mMeshes[0];
//
//	//load boneData to vertices
//	std::unordered_map<std::string, std::pair<int, fan::mat4>> bone_info;
//	std::vector<unsigned int> boneCounts;
//	boneCounts.resize(verticesOutput.size(), 0);
//	nBoneCount = mesh->mNumBones;
//
//		//loop through each bone
//	for (uint32_t i = 0; i < nBoneCount; i++) {
//		aiBone* bone = mesh->mBones[i];
//
//		bone_info[bone->mName.C_Str()] = { i, bone->mOffsetMatrix };
//
//		//loop through each vertex that have that bone
//		for (int j = 0; j < bone->mNumWeights; j++) {
//			unsigned int id = bone->mWeights[j].mVertexId;
//			float weight = bone->mWeights[j].mWeight;
//			boneCounts[id]++;
//			switch (boneCounts[id]) {
//			case 1:
//				verticesOutput[id].bone_ids.x = i;
//				verticesOutput[id].bone_weights.x = weight;
//				break;
//			case 2:
//				verticesOutput[id].bone_ids.y = i;
//				verticesOutput[id].bone_weights.y = weight;
//				break;
//			case 3:
//				verticesOutput[id].bone_ids.z = i;
//				verticesOutput[id].bone_weights.z = weight;
//				break;
//			case 4:
//				verticesOutput[id].bone_ids.w = i;
//				verticesOutput[id].bone_weights.w = weight;
//				break;
//			default:
//				//std::cout << "err: unable to allocate bone to vertex" << std::endl;
//				break;
//
//			}
//		}
//	}
//
//	//normalize weights to make all weights sum 1
//	for (int i = 0; i < verticesOutput.size(); i++) {
//		fan::vec4 & boneWeights = verticesOutput[i].bone_weights;
//		float totalWeight = boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
//		if (totalWeight > 0.0f) {
//			verticesOutput[i].bone_weights = fan::vec4(
//				boneWeights.x / totalWeight,
//				boneWeights.y / totalWeight,
//				boneWeights.z / totalWeight,
//				boneWeights.w / totalWeight
//			);
//		}
//	}
//
//	// create bone hirerchy
//	read_skeleton(skeletonOutput, scene->mRootNode, bone_info);
//	if (skeletonOutput.name.empty()) {
//		skeletonOutput.id = 0;
//		skeletonOutput.offset = fan::mat4(1);
//	}
//}
//
//static void fan_3d::graphics::animation::load_animation(const aiScene* scene, fan_3d::graphics::animation::loaded_animation_t& animation)
//{
//	//loading  first Animation
//	aiAnimation* anim = scene->mAnimations[0];
//
//	if (anim->mTicksPerSecond != 0.0f)
//		animation.ticks_per_second = anim->mTicksPerSecond;
//	else
//		animation.ticks_per_second = 1;
//
//
//	animation.duration = anim->mDuration * anim->mTicksPerSecond;
//	animation.bone_transforms;
//
//	//load positions rotations and scales for each bone
//	// each channel represents each bone
//	for (int i = 0; i < anim->mNumChannels; i++) {
//		aiNodeAnim* channel = anim->mChannels[i];
//		fan_3d::graphics::animation::bone_transform_track_t track;
//		for (int j = 0; j < channel->mNumPositionKeys; j++) {
//			track.position_timestamps.push_back(channel->mPositionKeys[j].mTime);
//			track.positions.push_back(channel->mPositionKeys[j].mValue);
//		}
//		for (int j = 0; j < channel->mNumRotationKeys; j++) {
//			track.rotation_timestamps.push_back(channel->mRotationKeys[j].mTime);
//			track.rotations.push_back(channel->mRotationKeys[j].mValue);
//
//		}
//		for (int j = 0; j < channel->mNumScalingKeys; j++) {
//			track.scale_timestamps.push_back(channel->mScalingKeys[j].mTime);
//			track.scales.push_back(channel->mScalingKeys[j].mValue);
//	
//		}
//		animation.bone_transforms[channel->mNodeName.C_Str()] = track;
//	}
//}
//
//static std::pair<uint32_t, f32_t> fan_3d::graphics::animation::get_time_fraction(std::vector<f32_t>& times, f32_t& dt)
// {
//	uint32_t segment = 0;
//	while (dt > times[segment])
//		segment++;
//	f32_t start = times[segment - 1];
//	f32_t end = times[segment];
//	f32_t frac = (dt - start) / (end - start);
//	return {segment, frac};
//}
//
//void fan_3d::graphics::animation::get_pose(animation::loaded_animation_t* animation, animation::joint_t* skeleton, f32_t dt, std::vector<fan::mat4>* output, fan::mat4 parentTransform, fan::mat4 transform, uint32_t bone_count)
//{
//	std::string key = !bone_count ? animation->bone_transforms.begin()->first : skeleton->name;
//	fan_3d::graphics::animation::bone_transform_track_t btt = animation->bone_transforms[key];
//
//	//if (btt.positions.empty()) {
//	//	return;
//	//}
//
//	dt = fmod(dt, animation->duration);
//	std::pair<unsigned int, float> fp;
//	//calculate interpolated position
//	fp = get_time_fraction(btt.position_timestamps, dt);
//
//	fan::vec3 p1 = btt.positions[fp.first - 1];
//	fan::vec3 p2 = btt.positions[fp.first];
//
//	fan::vec3 position(fan::mix(p1, p2, fp.second));
//
//	//calculate interpolated rotation
//	fp = get_time_fraction(btt.rotation_timestamps, dt);
//	fan::quat rotation1 = btt.rotations[fp.first - 1];
//	fan::quat rotation2 = btt.rotations[fp.first];
//
//	fan::quat rotation = fan::quat::slerp(rotation1, rotation2, fp.second);
//
//	//calculate interpolated scale
//	fp = get_time_fraction(btt.scale_timestamps, dt);
//	fan::vec3 s1 = btt.scales[fp.first - 1];
//	fan::vec3 s2 = btt.scales[fp.first];
//
//	fan::vec3 scale(fan::mix(s1, s2, fp.second));
//
//	fan::mat4 positionMat = fan::mat4(1), scaleMat = fan::mat4(1);
//
//	positionMat = positionMat.translate(position);
//
//	fan::mat4 rotation_matrix = rotation;
//
//	scaleMat = scaleMat.scale(scale);
//	fan::mat4 localTransform = positionMat * rotation_matrix * scaleMat;
//	fan::mat4 global_transform = parentTransform * localTransform;
//
//	(*output)[skeleton->id] = transform * global_transform * skeleton->offset;
//
//	for (int i = 0; i < skeleton->children.size(); i++) {
//		get_pose(animation, &skeleton->children[i], dt, output, global_transform, transform, bone_count);
//	}
//}
//
//fan_3d::graphics::animation::simple_animation_t::simple_animation_t(fan::camera* camera, const properties_t& properties)
//	: m_camera(camera), 
//		m_angle(properties.angle), 
//		m_rotation_vector(properties.rotation_vector),
//		m_keyframe(0),
//		m_model(1)
//{
//
//	m_model = m_model.translate(properties.position);
//	m_model = m_model.scale(properties.size);
//	//m_model = m_model.rotate(properties.angle, properties.rotation_vector);
//
//	m_shader.set_vertex(
//		#include <fan/graphics/glsl/opengl/3D/simple_animation.vs>
//	);
//
//	m_shader.set_fragment(
//		#include <fan/graphics/glsl/opengl/3D/simple_animation.fs>
//	);
//
//	m_shader.compile();
//
//	image_diffuse = properties.model->image_diffuse;
//	m_vertices = properties.model->vertices;
//	m_indices = properties.model->indices;
//	m_animation = properties.model->animation;
//
//	glGenVertexArrays(1, &m_vao);
//	glGenBuffers(1, &m_vbo);
//	glGenBuffers(1, &m_ebo);
//
//	glBindVertexArray(m_vao);
//	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(fan_3d::graphics::animation::vertex_t) * m_vertices.size(), &m_vertices[0], GL_STATIC_DRAW);
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, position));
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, normal));
//	glEnableVertexAttribArray(2);
//	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, uv));
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), &m_indices[0], GL_STATIC_DRAW);
//	glBindVertexArray(0);
//}
//
//fan::vec3 fan_3d::graphics::animation::simple_animation_t::get_position() const
//{
//	return m_model.get_translation();
//}
//
//void fan_3d::graphics::animation::simple_animation_t::set_position(const fan::vec3& position)
//{
//	m_model = m_model.translate(position);
//}
//
//fan::vec3 fan_3d::graphics::animation::simple_animation_t::get_size() const
//{
//	return m_model.get_scale();
//}
//
//void fan_3d::graphics::animation::simple_animation_t::set_size(const fan::vec3& size)
//{
//	m_model = m_model.scale(size);
//}
//
//fan::vec3 fan_3d::graphics::animation::simple_animation_t::get_rotation_vector() const
//{
//	return m_rotation_vector;
//}
//
//void fan_3d::graphics::animation::simple_animation_t::set_rotation_vector(const fan::vec3& vector)
//{
//	m_model = m_model.rotate(m_angle, vector);
//}
//
//f32_t fan_3d::graphics::animation::simple_animation_t::get_angle() const
//{
//	return m_angle;
//}
//
//void fan_3d::graphics::animation::simple_animation_t::set_angle(f32_t angle)
//{
//	m_model = m_model.rotate(angle, m_rotation_vector);
//}
//
//f32_t fan_3d::graphics::animation::simple_animation_t::get_keyframe() const
//{
//	return m_keyframe;
//}
//
//void fan_3d::graphics::animation::simple_animation_t::set_keyframe(uint32_t keyframe)
//{
//	m_keyframe = keyframe;
//}
//
//void fan_3d::graphics::animation::simple_animation_t::enable_draw()
//{
//	if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
//		m_draw_index = m_camera->m_window->push_draw_call(this, [addr = (uint64_t)this]{
//			((simple_animation_t*)(addr))->draw();
//		});
//	}
//	else {
//		m_camera->m_window->edit_draw_call(m_draw_index, this, [addr = (uint64_t)this] {
//			((simple_animation_t*)(addr))->draw();
//		});
//	}
//}
//
//void fan_3d::graphics::animation::simple_animation_t::disable_draw()
//{
//	if (m_draw_index == -1) {
//		return;
//	}
//
//	m_camera->m_window->erase_draw_call(m_draw_index);
//	m_draw_index = -1;
//}
//
//void fan_3d::graphics::animation::simple_animation_t::draw()
//{
//	glEnable(GL_DEPTH_TEST);
//
//	m_shader.use();
//
//	fan::mat4 projection(1);
//	projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.0), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 1000.0f);
//
//	fan::mat4 view(m_camera->get_view_matrix());
//
//	std::vector<fan::mat4> models(m_animation.bone_transforms.size(), fan::mat4(1));
//
//	int i = 0;
//	for (const auto& it : m_animation.bone_transforms) {
//		models[i] = models[i].translate(it.second.positions[m_keyframe % it.second.positions.size()]);
//		models[i] = models[i].scale(it.second.scales[m_keyframe % it.second.scales.size()]);
//		i++;
//	}
//
//	m_shader.set_mat4("projection", projection);
//	m_shader.set_mat4("view", view);
//	m_shader.set_mat4("models", models[0][0].data(), i);
//
//	glBindVertexArray(m_vao);
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, image_diffuse->texture);
//
//	m_shader.set_int("diff_texture", 0);
//
//	glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
//
//	glDisable(GL_DEPTH_TEST);
//}
//
///**/
//
//fan_3d::graphics::animation::animated_model_t_::animated_model_t_(std::string model_path, fan_2d::graphics::image_t image_diffuse_)
// : image_diffuse(image_diffuse_) {
//	scene = importer.ReadFile(model_path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals);
//
//	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
//		fan::throw_error(std::string("animation load error:") + importer.GetErrorString());
//	}
//
//	load_model(scene, vertices, indices, skeleton, bone_count);
//	load_animation(scene, animation);
//}
//
//fan_3d::graphics::animation::animator_t::animator_t(fan::camera* camera, const properties_t& properties)
//	: m_camera(camera), 
//		m_bone_count(0), 
//		m_identity(1), 
//		m_angle(properties.angle), 
//		m_rotation_vector(properties.rotation_vector),
//		m_timestamp(0),
//		m_model(1)
//{
//
//	m_model = m_model.translate(properties.position);
//	m_model = m_model.scale(properties.size);
//	//m_model = m_model.rotate(properties.angle, properties.rotation_vector);
//
//	m_shader.set_vertex(
//		#include <fan/graphics/glsl/opengl/3D/animation.vs>
//	);
//
//	m_shader.set_fragment(
//		#include <fan/graphics/glsl/opengl/3D/animation.fs>
//	);
//
//	m_shader.compile();
//
//	m_transform = properties.model->scene->mRootNode->mTransformation;
//
//	image_diffuse = properties.model->image_diffuse;
//	m_vertices = properties.model->vertices;
//	m_indices = properties.model->indices;
//	m_bone_count = properties.model->bone_count;
//	m_skeleton = properties.model->skeleton;
//	m_animation = properties.model->animation;
//
//	//currentPose is held in this vector and uploaded to gpu as a matrix array uniform
//	m_current_pose.resize(m_bone_count == 0 ? 1 : m_bone_count, m_identity); // use this for no animation
//
//	glGenVertexArrays(1, &m_vao);
//	glGenBuffers(1, &m_vbo);
//	glGenBuffers(1, &m_ebo);
//
//	glBindVertexArray(m_vao);
//	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(fan_3d::graphics::animation::vertex_t) * m_vertices.size(), &m_vertices[0], GL_STATIC_DRAW);
//	glEnableVertexAttribArray(0);
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, position));
//	glEnableVertexAttribArray(1);
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, normal));
//	glEnableVertexAttribArray(2);
//	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, uv));
//	glEnableVertexAttribArray(3);
//	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, bone_ids));
//	glEnableVertexAttribArray(4);
//	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(fan_3d::graphics::animation::vertex_t), (GLvoid*)offsetof(fan_3d::graphics::animation::vertex_t, bone_weights));
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), &m_indices[0], GL_STATIC_DRAW);
//	glBindVertexArray(0);
//}
//
//fan::vec3 fan_3d::graphics::animation::animator_t::get_position() const
//{
//	return m_model.get_translation();
//}
//
//void fan_3d::graphics::animation::animator_t::set_position(const fan::vec3& position)
//{
//	m_model = m_model.translate(position);
//}
//
//fan::vec3 fan_3d::graphics::animation::animator_t::get_size() const
//{
//	return m_model.get_scale();
//}
//
//void fan_3d::graphics::animation::animator_t::set_size(const fan::vec3& size)
//{
//	m_model = m_model.scale(size);
//}
//
//fan::vec3 fan_3d::graphics::animation::animator_t::get_rotation_vector() const
//{
//	return m_rotation_vector;
//}
//
//void fan_3d::graphics::animation::animator_t::set_rotation_vector(const fan::vec3& vector)
//{
//	m_model = m_model.rotate(m_angle, vector);
//}
//
//f32_t fan_3d::graphics::animation::animator_t::get_angle() const
//{
//	return m_angle;
//}
//
//void fan_3d::graphics::animation::animator_t::set_angle(f32_t angle)
//{
//	m_model = m_model.rotate(angle, m_rotation_vector);
//}
//
//f32_t fan_3d::graphics::animation::animator_t::get_timestamp() const
//{
//	return m_timestamp;
//}
//
//void fan_3d::graphics::animation::animator_t::set_timestamp(f32_t timestamp)
//{
//	m_timestamp = timestamp;
//}
//
//void fan_3d::graphics::animation::animator_t::enable_draw()
//{
//	if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
//		m_draw_index = m_camera->m_window->push_draw_call(this, [addr = (uint64_t)this]{
//			((animator_t*)(addr))->draw();
//		});
//	}
//	else {
//		m_camera->m_window->edit_draw_call(m_draw_index, this, [addr = (uint64_t)this] {
//			((animator_t*)(addr))->draw();
//		});
//	}
//}
//
//void fan_3d::graphics::animation::animator_t::disable_draw()
//{
//	if (m_draw_index == -1) {
//		return;
//	}
//
//	m_camera->m_window->erase_draw_call(m_draw_index);
//	m_draw_index = -1;
//}
//
//void fan_3d::graphics::animation::animator_t::draw()
//{
//	glEnable(GL_DEPTH_TEST);
//
//	get_pose(&m_animation, &m_skeleton, m_timestamp + 1, &m_current_pose, m_identity, m_transform, m_bone_count);
//
//	m_shader.use();
//
//	fan::mat4 projection(1);
//	projection = fan::math::perspective<fan::mat4>(fan::math::radians(90.0), (f32_t)m_camera->m_window->get_size().x / (f32_t)m_camera->m_window->get_size().y, 0.1f, 1000.0f);
//
//	fan::mat4 view(m_camera->get_view_matrix());
//	fan::print(m_current_pose[0]);
//	m_shader.set_mat4("projection", projection);
//	m_shader.set_mat4("view", view);
//	m_shader.set_mat4("model_matrix", m_model);
//	m_shader.set_mat4("bone_transforms", &m_current_pose[0][0][0], m_bone_count);
//
//	glBindVertexArray(m_vao);
//	glActiveTexture(GL_TEXTURE0);
//	glBindTexture(GL_TEXTURE_2D, image_diffuse->texture);
//
//	m_shader.set_int("diff_texture", 0);
//
//	glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
//
//	glDisable(GL_DEPTH_TEST);
//}

#endif