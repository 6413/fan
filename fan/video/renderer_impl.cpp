module;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

#if defined(FAN_2D)
  extern "C" {
    #include <libavformat/avformat.h>
  }
#endif

#endif

module fan.graphics.video.renderer;

#if defined (FAN_WINDOW) && defined(FAN_VIDEO)

void fan::graphics::video::player_t::update() {
  if (fan::window::is_key_clicked(fan::key_space)) { toggle_pause(); }
  if (fan::window::is_key_clicked(fan::key_left)) { skip(-seek_time_skip); }
  if (fan::window::is_key_clicked(fan::key_right)) { skip(seek_time_skip); }

  if (dropping_seek_frames) {
    for (int i = 0; i < 16; i++) {
      if (read_packet()) {
        auto out = decoder.decode_packet(buf.data(), buf.size(), codec_name, false);
        if (!out.empty()) {
          frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
        }
      }
      while (!frames.empty()) {
        if (static_cast<f32_t>(packet_pts * time_base) >= seek_target_time) {
          dropping_seek_frames = false;
          break;
        }
        frames.erase(frames.begin());
      }
      if (!dropping_seek_frames) break;
    }
    frame_timer.restart();
    return;
  }

  if (frames.size() < 8) {
    if (read_packet()) {
      auto out = decoder.decode_packet(buf.data(), buf.size(), codec_name, false);
      if (!out.empty()) {
        frames.insert(frames.end(), std::make_move_iterator(out.begin()), std::make_move_iterator(out.end()));
      }
    }
  }

  if (paused) return;

  if (frame_timer && !frames.empty()) {
    frame_timer.restart();
    auto& frame = frames.front();
    if (frame.success && !frame.plane_data.empty()) {
      renderer.update(frame.image_size, frame.plane_data, frame.linesize, frame.pixel_format);
      if (packet_pts != std::numeric_limits<std::int64_t>::min()) {
        current_time = static_cast<f32_t>(packet_pts * time_base);
      }
    }
    frames.erase(frames.begin());
  }
}

void fan::graphics::video::player_t::show(bool controls) {
  f32_t padding = 20.f;
  f32_t control_height = 150.f;

  fan::vec2 vp = gui::get_window_size().offset_y(-control_height);
  fan::vec2 fit = fan::vec2(demuxer.get_video_size()).fit(vp);
  renderer.shape.set_size(fit / 2.f);
  renderer.shape.set_position(gui::get_window_pos() + vp / 2.f);
  if (!controls) return;
  fan::vec2 avail = gui::get_window_size();
  fan::vec2 ws_pos = gui::get_window_pos();
  gui::anchor_bottom_left(fan::vec2(padding, -control_height));
  if (auto cw = gui::child_window("##controls", fan::vec2(avail.x, control_height))) {
    gui::set_cursor_pos_y(gui::get_cursor_pos_y() + padding);
    { // times + slider
      f32_t width = gui::get_content_region_avail().x;
      std::string cur = fan::time::format_seconds(current_time);
      std::string dur = fan::time::format_seconds(duration);
      fan::vec2 cur_size = gui::get_text_size(cur);
      fan::vec2 dur_size = gui::get_text_size(dur);
      gui::text(cur);
      gui::same_line();
      gui::spacing();
      gui::same_line();
      gui::set_cursor_pos_y(gui::get_cursor_pos_y() - cur_size.y / 4.f);
      gui::set_next_item_width(width - (cur_size.x + dur_size.x) - padding * 3.f);
      gui::slider("##seek_bar", &display_time, 0.f, duration);
      gui::same_line();
      gui::spacing();
      gui::same_line();
      gui::set_cursor_pos_y(gui::get_cursor_pos_y() - dur_size.y / 4.f);
      gui::text(dur);
    }

    if (gui::is_item_edited()) {
      seek(display_time);
    }
    else if (!gui::is_item_active()) {
      display_time = current_time;
    }
    gui::style_scope_t s;
    s.color(gui::col_button, {0, 0, 0, 0})
     .color(gui::col_button_hovered, {0.25, 0.25, 0.25, 0.25})
     .color(gui::col_button_active, {0.3, 0.3, 0.3, 0.3});
    if (gui::image_button(image_t{ paused ? "icons/wplay.webp" : "icons/wpause.webp", image_presets::pixel_art()}, {28.f, 28.f})) toggle_pause();
    gui::same_line();
    if (gui::button("-" + std::to_string(int(seek_time_skip)) + "s")) { skip(-seek_time_skip); }
    gui::same_line();
    if (gui::button("+" + std::to_string(int(seek_time_skip)) + "s")) { skip(seek_time_skip); }

    //show_debug();

  } // cw controls
}

#endif
