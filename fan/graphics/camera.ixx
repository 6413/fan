module;

export module fan.camera;

export import fan.math;
export import fan.types.vector;
export import fan.types.matrix;

export namespace fan {
  struct camera {
    camera() {
      this->update_view();
    }

    fan::mat4 get_view_matrix() const {
      return fan::math::look_at_left<fan::mat4, fan::vec3>(
        fan::vec3(position),
        position + m_front,
        this->m_up
      );
    }

    fan::mat4 get_view_matrix(const fan::mat4& m) const {
      return m * fan::math::look_at_left<fan::mat4, fan::vec3>(
        fan::vec3(position),
        position + m_front,
        this->world_up
      );
    }

    fan::vec3 get_front() const {
      return this->m_front;
    }

    void set_front(const fan::vec3 front) {
      this->m_front = front;
    }

    fan::vec3 get_right() const {
      return m_right;
    }

    void set_right(const fan::vec3 right) {
      m_right = right;
    }

    f32_t get_yaw() const {
      return this->m_yaw;
    }

    void set_yaw(f32_t angle) {
      this->m_yaw = angle;
      if (m_yaw > max_yaw) {
        m_yaw = -max_yaw;
      }
      if (m_yaw < -max_yaw) {
        m_yaw = max_yaw;
      }
    }

    f32_t get_pitch() const {
      return this->m_pitch;
    }

    void set_pitch(f32_t angle) {
      this->m_pitch = angle;
      if (this->m_pitch > max_pitch) {
        this->m_pitch = max_pitch;
      }
      if (this->m_pitch < -max_pitch) {
        this->m_pitch = -max_pitch;
      }
    }

    void update_view() {
      this->m_front = (fan::math::direction_vector<fan::vec3>(this->m_yaw, this->m_pitch)).normalized();
      this->m_right = (fan::math::cross(this->world_up, this->m_front)).normalized();
      this->m_up = (fan::math::cross(this->m_front, this->m_right)).normalized();
    }

    void rotate_camera(fan::vec2 offset) {
      offset *= sensitivity;

      this->set_yaw(this->get_yaw() + offset.x);
      this->set_pitch(this->get_pitch() - offset.y);

      this->update_view();
    }

    void move(f32_t movement_speed, f32_t friction = 12);

    f32_t sensitivity = 0.1f;

    bool first_movement = true;
    static constexpr f32_t max_yaw = 180;
    static constexpr f32_t max_pitch = 89;

    static constexpr f32_t gravity = 500;
    static constexpr f32_t jump_force = 100;

    static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);
    // protected:

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