#pragma once

namespace fan {
  namespace graphics {
    namespace animation {

      struct frame_transform_t {
        fan::vec3 position = 0;
        fan::vec3 size = 1;
        f32_t angle = 0;
      };

      struct key_frames_t {
        void push_back(const frame_transform_t& t) {
          frame_transforms.push_back(t);
        }
        void erase(uint32_t i) {
          frame_transforms[i] = frame_transforms[frame_transforms.size() - 1];
          frame_transforms.pop_back();
        }

        inline static f32_t get_delta_divide(f32_t current_time) {
          return 1.0 / (fmodf(current_time, 1.0));
        }
        template <typename T>
        inline T get_interpolation(f32_t current_time, T frame_transform_t::*member) {
          uint32_t max_i = frame_transforms.size();
          uint32_t c_time = current_time;
          T value;
          if ((c_time + 1) % max_i < c_time % max_i) {
            //                              or 0
            value = -((frame_transforms[(c_time + 1) % max_i].*member -
              frame_transforms[c_time % max_i].*member) / get_delta_divide(current_time));
            value += frame_transforms[(c_time + 1) % max_i].*member;
          }
          else {
            value = frame_transforms[c_time % max_i].*member + (((frame_transforms[(c_time + 1) % max_i].*member -
              frame_transforms[c_time % max_i].*member)) / get_delta_divide(current_time));
          }
          return value;
        }

        frame_transform_t process(fan::window_t* window) {
          uint32_t max_i = frame_transforms.size();
          uint32_t c_time = m_time;
          frame_transform_t frame;
          frame.position = get_interpolation(m_time, &frame_transform_t::position);
          frame.size = get_interpolation(m_time, &frame_transform_t::size);
          frame.angle = get_interpolation(m_time, &frame_transform_t::angle);
          m_time += window->get_delta_time();
          return frame;
        }

        std::vector<frame_transform_t> frame_transforms;
        f32_t m_time = 0;
      };

      struct strive_t {

        struct properties_t {
          frame_transform_t src;
          frame_transform_t dst;
          uint64_t time_to_destination;
        };

        strive_t(const frame_transform_t& origin) {
          m_origin = origin;
        }

        void set(const properties_t& p) {
          auto p_od = (p.dst.position - m_origin.position);
          auto s_od = (p.dst.size - m_origin.size);
          auto r_od = (p.dst.angle - m_origin.angle);

          auto p_d = (p.dst.position - p.src.position);
          auto s_d = (p.dst.size - p.src.size);
          auto r_d = (p.dst.angle - p.src.angle);

          m_rate_of_change.position = p_od * (f64_t)p.time_to_destination / 1e+9;
          m_rate_of_change.size = s_od * (f64_t)p.time_to_destination / 1e+9;
          m_rate_of_change.angle = r_od * (f64_t)p.time_to_destination / 1e+9;
          m_dst = p.dst;

          m_distance.position = p_d.abs();
          m_distance.size = s_d.abs();
          m_distance.angle = fan::math::abs(r_d);

          started = true;
        }

        frame_transform_t process(fan::window_t* window, const frame_transform_t& current, bool* done) {
          f64_t dt = window->get_delta_time();

          if (!started) {
            *done = true;
            return current;
          }
          
          if (
           
            m_distance.size <= fan::vec3(0) &&
            m_distance.angle <= 0
            ) {
            *done = true;
            return m_dst;
          }


          frame_transform_t ret;
          ret.position = m_rate_of_change.position * dt;
          ret.size = m_rate_of_change.size * dt;
          ret.angle = m_rate_of_change.angle * dt;

          m_distance.position -= ret.position.abs();
          m_distance.size -= ret.size.abs();
          m_distance.angle -= abs(ret.angle);

          ret.position += current.position;
          ret.size += current.size;
          ret.angle += current.angle;

          *done = false;
          return ret;
        }
        
        frame_transform_t m_rate_of_change;
        frame_transform_t m_origin;
        frame_transform_t m_dst;
        frame_transform_t m_distance;
        bool started = false;
      };
    }
  }
}