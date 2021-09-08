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