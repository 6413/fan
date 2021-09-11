class rounded_rectangle : public fan_2d::graphics::vertice_vector {
public:

	struct properties_t : public vertice_vector::properties_t{

		fan::vec2 size;
		f32_t radius = 0;
	};

	static constexpr int m_segments = 4 * 10;

	rounded_rectangle(fan::camera* camera);

	void push_back(const properties_t& properties);

	fan::vec2 get_position(uintptr_t i) const;
	void set_position(uintptr_t i, const fan::vec2& position);

	fan::vec2 get_size(uintptr_t i) const;
	void set_size(uintptr_t i, const fan::vec2& size);

	f32_t get_radius(uintptr_t i) const;
	void set_radius(uintptr_t i, f32_t radius);

	void draw();

	bool inside(uintptr_t i) const;

	fan::color get_color(uintptr_t i) const;
	void set_color(uintptr_t i, const fan::color& color);

	uint32_t size() const;

	void write_data();

	void edit_data(uint32_t i);

	void edit_data(uint32_t begin, uint32_t end);

private:

	using fan_2d::graphics::vertice_vector::push_back;

	uint32_t total_points = 0;

	std::vector<fan::vec2> m_position;
	std::vector<fan::vec2> m_size;
	std::vector<f32_t> m_radius;
};

class circle : public fan_2d::graphics::vertice_vector {
public:

	circle(fan::camera* camera) : fan_2d::graphics::vertice_vector(camera) {}

	void push_back(const fan::vec2& position, f32_t radius, const fan::color& color) {
		this->m_position.emplace_back(position);
		this->m_radius.emplace_back(radius);

		vertice_vector::properties_t properties;
		properties.color = color;
		properties.rotation_point = position + radius;

		for (int i = 0; i < m_segments; i++) {

			f32_t theta = fan::math::two_pi * f32_t(i) / m_segments;

			properties.position = position + fan::vec2(radius * std::cos(theta), radius * std::sin(theta));

			vertice_vector::push_back(properties);
		}
	}

	fan::vec2 get_position(uintptr_t i) const {
		return this->m_position[i];
	}
	void set_position(uintptr_t i, const fan::vec2& position) {
		this->m_position[i] = position;

		for (int j = 0; j < m_segments; j++) {

			f32_t theta = fan::math::two_pi * f32_t(j) / m_segments;

			vertice_vector::set_position(i * m_segments + j, position + fan::vec2(m_radius[i] * std::cos(theta), m_radius[i] * std::sin(theta)));
		}
	}

	f32_t get_radius(uintptr_t i) const {
		return this->m_radius[i];
	}

	void set_radius(uintptr_t i, f32_t radius) {
		this->m_radius[i] = radius;

		const fan::vec2 position = this->get_position(i);

		for (int j = 0; j < m_segments; j++) {

			f32_t theta = fan::math::two_pi * f32_t(j) / m_segments;

			vertice_vector::set_position(i * m_segments + j, position + fan::vec2(m_radius[i] * std::cos(theta), m_radius[i] * std::sin(theta)));

		}
	}

	void draw() {
		fan_2d::graphics::vertice_vector::draw(fan_2d::graphics::shape::triangle_fan, m_segments, 0, this->size() * m_segments);
	}

	bool inside(uintptr_t i) const {
		const fan::vec2 position = this->get_position(i);
		const fan::vec2 mouse_position = this->m_camera->m_window->get_mouse_position();

		if ((mouse_position.x - position.x) * (mouse_position.x - position.x) +
			(mouse_position.y - position.y) * (mouse_position.y - position.y) <= m_radius[i] * m_radius[i]) {
			return true;
		}

		return false;
	}

	fan::color get_color(uintptr_t i) const {
		return vertice_vector::get_color(i * m_segments);
	}
	void set_color(uintptr_t i, const fan::color& color) {
		for (int j = 0; j < m_segments; j++) {
			vertice_vector::set_color(i * m_segments + j, color);
		}
	}

	uint32_t size() const {
		return this->m_position.size();
	}

	void erase(uintptr_t i)  {
		fan_2d::graphics::vertice_vector::erase(i * m_segments, i * m_segments + m_segments);

		this->m_position.erase(this->m_position.begin() + i);
		this->m_radius.erase(this->m_radius.begin() + i);
	}
	void erase(uintptr_t begin, uintptr_t end) {
		fan_2d::graphics::vertice_vector::erase(begin * m_segments, end * m_segments);

		this->m_position.erase(this->m_position.begin() + begin, this->m_position.begin() + end);
		this->m_radius.erase(this->m_radius.begin() + begin, this->m_radius.begin() + end);
	}

	void write_data() {
		vertice_vector::write_data();
	}

	void edit_data(uint32_t i) {
		vertice_vector::edit_data(i * m_segments, i * m_segments + m_segments);
	}

	void edit_data(uint32_t begin, uint32_t end) {
		vertice_vector::edit_data(begin * m_segments, (end - begin + 1) * m_segments);
	}
protected:

	static constexpr int m_segments = 50;

	std::vector<fan::vec2> m_position;
	std::vector<f32_t> m_radius;

};