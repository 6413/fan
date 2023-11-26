#pragma once

namespace fan {
  namespace graphics {
    struct animation_t {

      animation_t() = default;
      animation_t(loco_t::texturepack_t* texturepack) {
        tp = texturepack;
      }

      struct key_frame_t {
        f32_t time = 0;
        fan::vec3 position = 0;
        fan::vec2 size = 400;
        f32_t angle = 0;
        fan::vec3 rotation_vector = fan::vec3(0, 0, 1);
      };

      struct controls_t {
        bool playing = false;
        bool loop = true;
        f32_t time = 0;
        f32_t max_time = 0;
      };

      void update() {
        for (auto& obj : objects) {
          if (obj.key_frames.empty()) {
            continue;
          }
          if (obj.frame_index + 1 >= obj.key_frames.size()) {
            continue;
          }
          if (controls.time < obj.key_frames[obj.frame_index].time) {
            continue;
          }
          auto& frame_src = obj.key_frames[obj.frame_index];
          auto& frame_dst = obj.key_frames[obj.frame_index + 1];
          if (controls.time < frame_dst.time) {
            f32_t offset = fan::math::normalize(controls.time, frame_src.time, frame_dst.time);
            obj.current_frame.position = frame_src.position.lerp(frame_dst.position, offset);
            obj.current_frame.size = frame_src.size.lerp(frame_dst.size, offset);
            obj.current_frame.angle = fan::math::lerp(frame_src.angle, frame_dst.angle, offset);
            obj.current_frame.rotation_vector = frame_src.rotation_vector.lerp(frame_dst.rotation_vector, offset);
          }
          else {
            obj.frame_index++;
            obj.current_frame = obj.key_frames[obj.frame_index];
          }
        }
      }
      struct object_t {
        int frame_index = 0;
        std::vector<key_frame_t> key_frames;
        key_frame_t current_frame;
        // can be either image or texturepack image name
        fan::string image_name;
        loco_t::shape_t sprite;
      };

      void play_from_begin() {
        controls.time = 0;
        for (auto& obj : objects) {
          obj.frame_index = 0;
        }
        controls.playing = true;
      }

      static void load_image(loco_t::shape_t& shape, const fan::string& name, loco_t::image_t& image) {
        if (image.load(name)) {
          fan::print_warning("failed to load image:" + name);
        }
        else {
          shape.set_image(&image);
        }
      }

      static void load_image(loco_t::shape_t& shape, const fan::string& name, loco_t::texturepack_t& texturepack) {
        loco_t::texturepack_t::ti_t ti;
        if (texturepack.qti(name, &ti)) {
          fan::print_warning("failed to load texturepack image:" + name);
        }
        else {
          shape.set_tp(&ti);
        }
      }

      bool is_finished() {
        return controls.max_time <= controls.time;
      }

      void play_animation() {
        if (controls.loop && is_finished()) {
          play_from_begin();
        }

        update();

        int index = 0;
        for (auto& obj : objects) {
          key_frame_t kf = obj.current_frame;
          obj.sprite.set_position(origin + kf.position);
          obj.sprite.set_size(kf.size);
          obj.sprite.set_angle(kf.angle);
          obj.sprite.set_rotation_vector(kf.rotation_vector);
          controls.time += gloco->delta_time;
        }
      }

      void push_sprite(uint32_t i, auto&& temp) {
        loco_t::shapes_t::vfi_t::properties_t vfip;
        vfip.shape.rectangle->position = temp.get_position();
        vfip.shape.rectangle->position.z += 1;
        vfip.shape.rectangle->size = temp.get_size();
        objects[i].sprite = std::move(temp);
      }

      void file_load(const fan::string& path) {
        fan::string istr;
        fan::io::file::read(path, &istr);
        uint32_t off = 0;
        fan::read_from_string(istr, off, controls.loop);
        fan::read_from_string(istr, off, controls.max_time);
        uint32_t obj_size = 0;
        fan::read_from_string(istr, off, obj_size);
        objects.resize(obj_size);
        uint32_t iterate_idx = 0;
        for (auto& obj : objects) {
          fan::read_from_string(istr, off, obj.image_name);
          uint32_t keyframe_size = 0;
          fan::read_from_string(istr, off, keyframe_size);
          obj.key_frames.resize(keyframe_size);
          int frame_idx = 0;
          //for (auto& frame : obj.key_frames) {
          //  frame = ((key_frame_t*)&istr[off])[frame_idx++];
          //  //timeline.frames.push_back(frame.time * time_divider);
          //}
          memcpy(obj.key_frames.data(), &istr[off], sizeof(key_frame_t) * obj.key_frames.size());
          off += sizeof(key_frame_t) * obj.key_frames.size();
          if (obj.key_frames.size()) {
            push_sprite(iterate_idx, fan::graphics::sprite_t{ {
              .position = obj.key_frames[0].position,
              .size = obj.key_frames[0].size,
              .angle = obj.key_frames[0].angle,
              .rotation_vector = obj.key_frames[0].rotation_vector
            } });
            load_image(obj.sprite, obj.image_name, *tp);
          }
          iterate_idx += 1;
        }
      }
      void set_origin() {

        for (auto& obj : objects) {
          if (obj.key_frames.empty()) {
            continue;
          }
          fan::vec2 off = -obj.key_frames[0].position;
          for (auto& i : obj.key_frames) {
            i.position += off;
          }
        }
      }
      void set_position(uint32_t i, const fan::vec2& position) {
        origin = position;
        object_t& obj = objects[i];
        fan::vec3 sp = obj.sprite.get_position();
        obj.sprite.set_position(fan::vec3(position, sp.z));
      }

      f32_t time = 0;
      controls_t controls;
      std::vector<object_t> objects;
      fan::vec2 origin = 0;
      loco_t::texturepack_t* tp = nullptr;
    };
  }
}