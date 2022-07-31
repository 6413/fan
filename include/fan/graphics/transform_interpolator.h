#pragma once

namespace fan {
  namespace graphics {
    namespace animation {
      struct key_frames_t {

        typedef void(*move_cb_t)(uint32_t src, uint32_t dst);

        struct frame_transform_t {
          fan::vec3 position;
          fan::vec2 size;
          fan::vec3 rotation;
        };

        void open(move_cb_t move_cb_) {
          frame_transforms.open();
          move_cb = move_cb_;
        }
        void close() {
          frame_transforms.close();
        }

        uint32_t push_back(const frame_transform_t& t) {
          return frame_transforms.push_back(t);
        }
        void erase(uint32_t i) {
          frame_transforms[i] = frame_transforms[frame_transforms.size() - 1];
          frame_transforms.pop_back();
          move_cb(frame_transforms.size(), i);
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

        frame_transform_t process(f32_t current_time) {
          uint32_t max_i = frame_transforms.size();
          uint32_t c_time = current_time;
          frame_transform_t frame;
          frame.position = get_interpolation(current_time, &frame_transform_t::position);
          frame.size = get_interpolation(current_time, &frame_transform_t::size);
          frame.rotation = get_interpolation(current_time, &frame_transform_t::rotation);
          return frame;
        }

        fan::hector_t<frame_transform_t> frame_transforms;
        move_cb_t move_cb;
      };
    }
  }
}