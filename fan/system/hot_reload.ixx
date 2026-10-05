export module fan.system.hot_reload;

import std;
import fan.types;
import fan.system.dynamic_library;
import fan.event;
import fan.event.types;
import fan.print;

export namespace fan {
  struct hot_reload_t {
    struct config_t {
      std::string source;
      f64_t debounce_s = 0.35;
      f64_t retry_s = 0.5;
      f64_t poll_fallback_s = 1.0;
      bool use_watcher = true;
    };

    hot_reload_t() = default;
    explicit hot_reload_t(const std::string& source) : hot_reload_t(config_t{.source = source}) {}
    hot_reload_t(const config_t& cfg) : m_cfg(cfg) {
      std::error_code ec;
      m_source = std::filesystem::absolute(cfg.source, ec).string();
      if (ec) {
        m_source = cfg.source;
      }
      m_file = std::filesystem::path(m_source).filename().string();
      if (cfg.use_watcher) {
        std::string dir = std::filesystem::path(m_source).parent_path().string();
        m_watcher = std::make_unique<fan::event::fs_watcher_t>(dir);
        auto res = m_watcher->start([this](const std::string& filename, int events) {
          if (!(events & (fan::fs_change | fan::fs_rename))) {
            return;
          }
          if (std::filesystem::path(filename).filename().string() != m_file) {
            return;
          }
          m_pending = true;
          m_timer = m_cfg.debounce_s;
        });
        if (!res) {
          notify("hot-reload: fs watcher failed: " + res.error());
          m_watcher.reset();
        }
        else {
          notify("hot-reload: watching " + dir + " for " + m_file);
        }
      }
      m_stamp = std::filesystem::last_write_time(m_source, ec);
      m_has_stamp = !ec;
    }
    ~hot_reload_t() {
      if (m_lib.is_loaded()) {
        m_lib.unload();
      }
      std::error_code ec;
      if (!m_cur_stage.empty()) {
        std::filesystem::remove(m_cur_stage, ec);
      }
    }

    hot_reload_t(const hot_reload_t&) = delete;
    hot_reload_t& operator=(const hot_reload_t&) = delete;
    hot_reload_t(hot_reload_t&&) = delete;
    hot_reload_t& operator=(hot_reload_t&&) = delete;

    void update(f64_t dt) {
      if (m_pending) {
        m_timer -= dt;
        if (m_timer <= 0.0) {
          m_pending = false;
          load();
        }
        return;
      }
      if (m_cfg.poll_fallback_s > 0.0) {
        m_poll_acc += dt;
        if (m_poll_acc >= m_cfg.poll_fallback_s) {
          m_poll_acc = 0.0;
          std::error_code ec;
          auto w = std::filesystem::last_write_time(m_source, ec);
          if (!ec && (!m_has_stamp || w != m_stamp)) {
            load();
          }
        }
      }
    }

    // Synchronous (re)load now: boot, R key, etc.
    void reload() {
      load();
    }

    bool is_loaded() const {
      return m_lib.is_loaded();
    }
    void* symbol(const char* name) {
      return m_lib.symbol(name);
    }
    std::uint32_t reload_count() const {
      return m_reloads;
    }
    const std::string& source() const {
      return m_source;
    }

  private:
    void notify(const std::string& m) {
      if (on_message) {
        on_message(m);
      }
    }
    void retry() {
      m_pending = true;
      m_timer = m_cfg.retry_s;
    }
    std::string staging_path(std::uint32_t n) const {
      auto dir = std::filesystem::path(m_source).parent_path();
      std::string name = std::filesystem::path(m_source).stem().string()
        + "_hot_" + std::to_string(n)
        + std::filesystem::path(m_source).extension().string();
      return (dir / name).string();
    }
    void load() {
      std::error_code ec;
      if (std::filesystem::file_size(m_source, ec) == 0 || ec) {
        notify("hot-reload: waiting for " + m_source);
        return;
      }
      m_stamp = std::filesystem::last_write_time(m_source, ec);
      m_has_stamp = !ec;
      std::string stage = staging_path(m_reloads + 1);
      std::filesystem::copy_file(m_source, stage,
        std::filesystem::copy_options::overwrite_existing, ec);
      if (ec) {
        notify("hot-reload: copy failed - " + ec.message());
        return;
      }
      dynamic_library_t next;
      if (!next.load(stage)) {
        notify("hot-reload: dlopen failed: " + dynamic_library_t::last_error());
        std::filesystem::remove(stage, ec);
        retry(); // linker may still be writing
        return;
      }
      bool commit = on_reload ? on_reload(next) : true;
      if (!commit) {
        next.unload();
        std::filesystem::remove(stage, ec);
        retry();
        return;
      }
      m_lib = std::move(next); // closes the old library
      if (!m_cur_stage.empty()) {
        std::filesystem::remove(m_cur_stage, ec); // old handle already closed
      }
      m_cur_stage = stage;
      ++m_reloads;
      notify("hot-reload: library swapped (" + stage + ")");
    }

  public:
    std::function<bool(dynamic_library_t& fresh)> on_reload;
    std::function<void(const std::string&)> on_message = [](const std::string& m) {
      fan::print(m);
    };

  private:
    config_t m_cfg;
    std::string m_source;
    std::string m_file;
    dynamic_library_t m_lib;
    std::unique_ptr<fan::event::fs_watcher_t> m_watcher;
    bool m_pending = false;
    f64_t m_timer = 0.0;
    f64_t m_poll_acc = 0.0;
    std::filesystem::file_time_type m_stamp{};
    bool m_has_stamp = false;
    std::string m_cur_stage;
    std::uint32_t m_reloads = 0;
  };
}
