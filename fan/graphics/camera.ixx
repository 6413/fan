export module fan.camera;

export import fan.math;
export import fan.types.vector;
export import fan.types.matrix;

export namespace fan {
  struct camera {
    camera();
    fan::mat4 get_view_matrix() const;
    fan::mat4 get_view_matrix(const fan::mat4& m) const;
    fan::vec3 get_front() const;
    void set_front(const fan::vec3 front);
    fan::vec3 get_right() const;
    void set_right(const fan::vec3 right);
    f32_t get_yaw() const;
    void set_yaw(f32_t angle);
    f32_t get_pitch() const;
    void set_pitch(f32_t angle);
    void update_view();
    void rotate_camera(fan::vec2 offset);

    f32_t sensitivity = 0.1f;
    bool first_movement = true;
    static constexpr f32_t max_yaw = 180;
    static constexpr f32_t max_pitch = 89;
    static constexpr f32_t gravity = 500;
    static constexpr f32_t jump_force = 100;
    static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);

    f32_t m_yaw = 0;
    f32_t m_pitch = 0;
    fan::vec3 m_right;
    fan::vec3 m_up;
    fan::vec3 m_front;
    fan::vec3 position = 0;
    fan::vec3 velocity = 0;
  };
}

export namespace fan::math {
  constexpr fan::vec2 convert_position_ndc(const fan::vec2& mouse_position, const fan::vec2i& window_size) {
    return fan::vec2((2.0f * mouse_position.x) / window_size[0] - 1.0f, 1.0f - (2.0f * mouse_position.y) / window_size[1]);
  }
  constexpr fan::ray3_t convert_position_to_ray(const fan::vec2i& mouse_position, const fan::vec2& screen_size, const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
    fan::vec4 ray_ndc((2.0f * mouse_position[0]) / screen_size.x - 1.0f, 1.0f - (2.0f * mouse_position[1]) / screen_size.y, 1.0f, 1.0f);
    fan::mat4 inverted_projection = projection.inverse();
    fan::vec4 ray_clip = inverted_projection * ray_ndc;
    ray_clip.z = -1.0f;
    ray_clip.w = 0.0f;
    fan::mat4 inverted_view = view.inverse();
    fan::vec4 ray_world = inverted_view * ray_clip;
    fan::vec3 ray_dir = fan::vec3(ray_world.x, ray_world.y, ray_world.z).normalized();
    fan::vec3 ray_origin = camera_position;
    return fan::ray3_t(ray_origin, ray_dir);
  }
}