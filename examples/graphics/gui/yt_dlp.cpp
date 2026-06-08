#include <fan/utility.h>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <coroutine>
#include <format>
#include <ranges>
#include <span>

import fan;
using namespace fan::graphics;

static constexpr const char* config_path = "ytdl_config.json";

struct format_t {
  const char* label;
  const char* ext;
  bool audio_only;
};

struct quality_t {
  const char* label;
  const char* filter;
};

static constexpr std::array audio_formats = {
  format_t{"Opus",  "opus", true},
  format_t{"MP3",   "mp3",  true},
  format_t{"AAC",   "aac",  true},
  format_t{"M4A",   "m4a",  true},
  format_t{"FLAC",  "flac", true},
  format_t{"WAV",   "wav",  true},
};

static constexpr std::array video_formats = {
  format_t{"MP4",  "mp4",  false},
  format_t{"WebM", "webm", false},
  format_t{"MKV",  "mkv",  false},
};

static constexpr std::array audio_qualities = {
  quality_t{"Best",     ""},
  quality_t{"~256kbps", "bestaudio[abr<=260]/bestaudio"},
  quality_t{"~128kbps", "bestaudio[abr<=130]/bestaudio"},
  quality_t{"~70kbps",  "bestaudio[abr<=80]/bestaudio"},
  quality_t{"~32kbps",  "bestaudio[abr<=40]/bestaudio"},
};

static constexpr std::array video_qualities = {
  quality_t{"Best",  ""},
  quality_t{"4K",    "bestvideo[height<=2160]+bestaudio/best[height<=2160]"},
  quality_t{"1440p", "bestvideo[height<=1440]+bestaudio/best[height<=1440]"},
  quality_t{"1080p", "bestvideo[height<=1080]+bestaudio/best[height<=1080]"},
  quality_t{"720p",  "bestvideo[height<=720]+bestaudio/best[height<=720]"},
  quality_t{"480p",  "bestvideo[height<=480]+bestaudio/best[height<=480]"},
  quality_t{"360p",  "bestvideo[height<=360]+bestaudio/best[height<=360]"},
};

static constexpr int url_display_len = 60;

static std::vector<std::string> build_args(
  const format_t& fmt,
  const quality_t& qual,
  const std::string& url,
  const std::string& output_dir) {
  std::vector<std::string> args = {"yt-dlp", "--newline", "--no-part", "--embed-metadata", "--verbose",
    "--retries", "3", "--fragment-retries", "3"};
  args.reserve(16);
  if (fmt.audio_only) {
    if (qual.filter[0] != '\0') args.insert(args.end(), {"-f", qual.filter});
    args.insert(args.end(), {"-x", "--audio-format", fmt.ext});
  } else {
    std::string f = qual.filter[0] != '\0'
      ? qual.filter
      : std::format("bestvideo[ext={}]+bestaudio/best", fmt.ext);
    args.insert(args.end(), {"-f", f, "--merge-output-format", fmt.ext});
  }
  args.insert(args.end(), {"-o", fan::path::join(output_dir, "%(title)s.%(ext)s"), url});
  return args;
}

enum class job_state_e { running, succeeded, failed };

struct config_t {
  std::string output_dir;
  int  format     = 0;
  int  quality    = 0;
  bool audio_mode = true;

  void json_read(fan::json& j) {
    j.get_if("output_dir", output_dir);
    j.get_if("format",     format);
    j.get_if("quality",    quality);
    j.get_if("audio_mode", audio_mode);
  }
  void json_write(fan::json& j) const {
    j.set("output_dir", output_dir);
    j.set("format",     format);
    j.set("quality",    quality);
    j.set("audio_mode", audio_mode);
  }
  void load() { fan::json::load_struct(config_path, *this); }
  void save() const { fan::json::save_struct(config_path, *this); }
};

struct download_job_t {
  struct state_info_t { fan::color color; const char* label; };
  static constexpr state_info_t state_info[] = {
    {fan::colors::yellow, "  downloading..."},  // running
    {fan::colors::green,  "  done"},            // succeeded
    {fan::colors::red,    "  failed"},          // failed
  };

  const state_info_t& info() const { return state_info[std::to_underlying(state)]; }
  std::string display_label(int pos) const {
    return std::format("[{}] {:.{}}{}{}", pos, url, url_display_len, url.size() > (size_t)url_display_len ? "..." : "", info().label);
  }

  std::string url;
  job_state_e state = job_state_e::running;
  int format_index = 0;
  int quality_index = 0;
  bool audio_mode = true;
  fan::event::task_t task;
};

static const fan::log_dispatcher_t default_log = fan::graphics::gui::default_logger();

struct ytdl_gui_t {
  ytdl_gui_t() { config.load(); engine.loop([&] { update(); }); }

  fan::event::task_t run_download_coro(int idx) {
    auto url = jobs[idx].url;
    auto args = build_args(
      jobs[idx].audio_mode ? audio_formats[jobs[idx].format_index] : video_formats[jobs[idx].format_index],
      jobs[idx].audio_mode ? audio_qualities[jobs[idx].quality_index] : video_qualities[jobs[idx].quality_index],
      url, config.output_dir);
    fan::printcl("[ytdl] starting:", url);

    auto result = co_await fan::process::run_async(args, default_log);

    jobs[idx].state = result.ok() ? job_state_e::succeeded : job_state_e::failed;

    if (jobs[idx].state == job_state_e::succeeded) fan::printcl("[ytdl] done");
    else                                           fan::printcl_err("[ytdl] failed");
  }

  void start_download() {
    if (url_buf.empty()) return;
    jobs.push_back({url_buf, job_state_e::running, config.format, config.quality, config.audio_mode});
    jobs.back().task = run_download_coro((int)jobs.size() - 1);
    url_buf.clear();
    url_focused = false;
  }

  bool any_running() const {
    return std::ranges::any_of(jobs, [](auto& j){ return j.state == job_state_e::running; });
  }

  void update() {
    auto wnd = gui::fullscreen_window("##ytdl");
    if (!wnd) return;

    gui::text("YouTube Downloader");
    gui::separator();

    gui::text("URL");
    gui::fill_width();
    if (!url_focused) gui::set_keyboard_focus_here();
    {
      std::size_t prev_len = url_buf.size();
      if (gui::input_text("##url", &url_buf)) {
        bool looks_like_paste = url_buf.size() > prev_len + 1;
        if (!any_running() && looks_like_paste && url_buf.contains("http"))
          start_download();
      }
      if (!url_focused && gui::is_item_active()) url_focused = true;
    }

    gui::text("Output directory");
    gui::fill_width_except("Browse");
    if (gui::input_text("##outdir", &config.output_dir))
      config.save();
    gui::same_line();
    if (gui::button("Browse"))
      dl_dialog.open(config.output_dir, [&]{ config.save(); });

    {
      bool changed = false;
      const char* mode_label = config.audio_mode ? "Audio" : "Video";
      const char* widest_qual = config.audio_mode ? "~256kbps" : "1440p";

      std::span<const format_t>  fmts  = config.audio_mode ? std::span<const format_t>{audio_formats}  : std::span<const format_t>{video_formats};
      std::span<const quality_t> quals = config.audio_mode ? std::span<const quality_t>{audio_qualities} : std::span<const quality_t>{video_qualities};

      f32_t btn_w  = gui::calc_button_width(mode_label);
      f32_t qual_w = gui::calc_button_width(widest_qual) + gui::get_style().ItemSpacing.x;
      f32_t avail  = gui::get_content_region_avail().x;
      f32_t fmt_w  = avail - btn_w - qual_w - gui::get_style().ItemSpacing.x * 2;

      if (gui::button(mode_label, {btn_w, 0.f})) {
        config.audio_mode = !config.audio_mode;
        config.format  = 0;
        config.quality = 0;
        changed = true;
      }
      gui::same_line();
      gui::set_next_item_width(fmt_w);
      changed |= gui::combo("##fmt", &config.format, (int)fmts.size(),
        [&](int i){ return fmts[i].label; });
      gui::same_line();
      gui::fill_width();
      changed |= gui::combo("##qual", &config.quality, (int)quals.size(),
        [&](int i){ return quals[i].label; });
      if (changed) config.save();
    }

    gui::spacing(4);
    if (gui::button_fill("Download") && !any_running())
      start_download();

    gui::separator();

    for (int i = 0; i < (int)jobs.size(); ++i)
      gui::text(jobs[i].info().color, jobs[i].display_label(i + 1));

    dl_dialog.is_finished();
  }

  config_t config;
  std::string url_buf;
  bool url_focused = false;
  std::vector<download_job_t> jobs;
  fan::graphics::folder_open_dialog_t dl_dialog;
  engine_t engine;
};

int main() {
  ytdl_gui_t{};
}