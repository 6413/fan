module;

#include <fan/utility.h>

#if defined(fan_compiler_gcc)
  #ifndef _GCC_MAX_ALIGN_T
    #define _GCC_MAX_ALIGN_T
  #endif
#endif

#include <coroutine>
#include <fan/nativefiledialog/nfd.h>

module fan.file_dialog;

import fan.event.uv_raw;
import fan.event;

namespace fan::graphics {
  static const char* dialog_default_path(std::string& storage) {
  #if defined(fan_platform_windows)
    return nullptr;
  #else
    storage = std::filesystem::current_path().string();
    return storage.c_str();
  #endif
  }

  struct dialog_t {
    dialog_t& open_file  (std::string_view filter, file_cb_t  on_done, dismissed_cb_t on_dismissed = {});
    dialog_t& open_files (std::string_view filter, files_cb_t on_done, dismissed_cb_t on_dismissed = {});
    dialog_t& save_file  (std::string_view filter, file_cb_t  on_done, dismissed_cb_t on_dismissed = {});
    dialog_t& open_folder(                         file_cb_t  on_done, dismissed_cb_t on_dismissed = {});

    bool        pending()   const;
    void        cancel();
    std::string last_path() const;

    struct state_t;
    std::shared_ptr<state_t> s;
  };

  struct dialog_t::state_t {
    std::shared_ptr<state_t>  self;
    std::mutex                mtx;
    std::string               result;
    std::vector<std::string>  results;
    std::atomic<bool>         pending  {false};
    std::atomic<bool>         cancelled{false};
    file_cb_t                 on_done;
    files_cb_t                on_done_multi;
    dismissed_cb_t            on_dismissed;
    fan::uv::async_t          async_;

    void dispatch() {
      bool got = !result.empty() || !results.empty();
      if (!cancelled && got) {
        if (on_done) {
          on_done(result);
        } else if (on_done_multi) {
          std::vector<std::string_view> views(results.begin(), results.end());
          on_done_multi(views);
        }
      } else if (!cancelled && on_dismissed) {
        on_dismissed();
      }
      on_done = {}; on_done_multi = {}; on_dismissed = {};
      pending.store(false);
      fan::uv::close(
        reinterpret_cast<fan::uv::handle_t*>(&async_),
        [](fan::uv::handle_t* h) {
          static_cast<state_t*>(h->data)->self.reset();
        }
      );
    }
  };

  static void launch(std::shared_ptr<dialog_t::state_t>& s, auto nfd_fn) {
    if (!s) s = std::make_shared<dialog_t::state_t>();
    s->result.clear();
    s->results.clear();
    s->cancelled.store(false);
    s->pending.store(true);
    s->self = s;
    fan::uv::async_init(
      (fan::uv::loop_t*)fan::event::get_loop(),
      &s->async_,
      [](fan::uv::async_t* h) {
        static_cast<dialog_t::state_t*>(h->data)->dispatch();
      }
    );
    s->async_.data = s.get();
  #if defined(_WIN32)
    fan::event::thread_create([s, nfd_fn = std::move(nfd_fn)] {
      if (!s->cancelled) nfd_fn(*s);
      fan::uv::async_send(&s->async_);
    });
  #else
    if (!s->cancelled) nfd_fn(*s);
    fan::uv::async_send(&s->async_);
  #endif
  }

  dialog_t& dialog_t::open_file(std::string_view filter, file_cb_t on_done, dismissed_cb_t on_dismissed) {
    if (!s) s = std::make_shared<state_t>();
    s->on_done = std::move(on_done); s->on_done_multi = {}; s->on_dismissed = std::move(on_dismissed);
    std::string f{filter};
    launch(s, [f = std::move(f)](state_t& st) {
      std::string storage;
      const char* default_path = dialog_default_path(storage);
      nfdchar_t* p = nullptr;
      if (NFD_OpenDialog(f.c_str(), default_path, &p) == NFD_OKAY && p) {
        std::lock_guard lock(st.mtx);
        st.result = p; std::free(p);
      }
    });
    return *this;
  }

  dialog_t& dialog_t::open_files(std::string_view filter, files_cb_t on_done, dismissed_cb_t on_dismissed) {
    if (!s) s = std::make_shared<state_t>();
    s->on_done = {}; s->on_done_multi = std::move(on_done); s->on_dismissed = std::move(on_dismissed);
    std::string f{filter};
    launch(s, [f = std::move(f)](state_t& st) {
      std::string storage;
      const char* default_path = dialog_default_path(storage);
      nfdpathset_t ps;
      if (NFD_OpenDialogMultiple(f.c_str(), default_path, &ps) == NFD_OKAY) {
        std::lock_guard lock(st.mtx);
        std::size_t n = NFD_PathSet_GetCount(&ps);
        st.results.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
          st.results.emplace_back(NFD_PathSet_GetPath(&ps, i));
        NFD_PathSet_Free(&ps);
      }
    });
    return *this;
  }

  dialog_t& dialog_t::save_file(std::string_view filter, file_cb_t on_done, dismissed_cb_t on_dismissed) {
    if (!s) s = std::make_shared<state_t>();
    s->on_done = std::move(on_done); s->on_done_multi = {}; s->on_dismissed = std::move(on_dismissed);
    std::string f{filter};
    launch(s, [f = std::move(f)](state_t& st) {
      std::string storage;
      const char* default_path = dialog_default_path(storage);
      nfdchar_t* p = nullptr;
      if (NFD_SaveDialog(f.c_str(), default_path, &p) == NFD_OKAY && p) {
        std::lock_guard lock(st.mtx);
        st.result = p; std::free(p);
      }
      std::printf("%s\n", NFD_GetError());
    });
    return *this;
  }

  dialog_t& dialog_t::open_folder(file_cb_t on_done, dismissed_cb_t on_dismissed) {
    if (!s) s = std::make_shared<state_t>();
    s->on_done = std::move(on_done); s->on_done_multi = {}; s->on_dismissed = std::move(on_dismissed);
    launch(s, [](state_t& st) {
      std::string storage;
      const char* default_path = dialog_default_path(storage);
      nfdchar_t* p = nullptr;
      if (NFD_PickFolder(default_path, &p) == NFD_OKAY && p) {
        std::lock_guard lock(st.mtx);
        st.result = p; std::free(p);
      }
    });
    return *this;
  }

  bool dialog_t::pending() const   { return s && s->pending.load(); }
  void dialog_t::cancel()          { if (s) s->cancelled.store(true); }
  std::string dialog_t::last_path() const {
    if (!s) return {};
    std::lock_guard lock(s->mtx);
    return s->result;
  }

  dialog_t& open_file(std::string_view filter, file_cb_t on_done, dismissed_cb_t on_dismissed) {
    static dialog_t d; return d.open_file(filter, std::move(on_done), std::move(on_dismissed));
  }
  dialog_t& open_files(std::string_view filter, files_cb_t on_done, dismissed_cb_t on_dismissed) {
    static dialog_t d; return d.open_files(filter, std::move(on_done), std::move(on_dismissed));
  }
  dialog_t& save_file(std::string_view filter, file_cb_t on_done, dismissed_cb_t on_dismissed) {
    static dialog_t d; return d.save_file(filter, std::move(on_done), std::move(on_dismissed));
  }
  dialog_t& open_folder(file_cb_t on_done, dismissed_cb_t on_dismissed) {
    static dialog_t d; return d.open_folder(std::move(on_done), std::move(on_dismissed));
  }

  fan::event::waitv_t<std::optional<std::string>> co_open_file(std::string_view filter) {
    fan::event::signal_awaitable_t<std::optional<std::string>> sig;
    open_file(filter,
      [&](std::string_view p) { sig.signal(std::string{p}); },
      [&]()                   { sig.signal(std::nullopt);   });
    co_return co_await sig;
  }

  fan::event::waitv_t<std::vector<std::string>> co_open_files(std::string_view filter) {
    fan::event::signal_awaitable_t<std::vector<std::string>> sig;
    open_files(filter,
      [&](std::vector<std::string_view> paths) {
        std::vector<std::string> out;
        out.reserve(paths.size());
        for (auto p : paths) out.emplace_back(p);
        sig.signal(std::move(out));
      },
      [&]() { sig.signal({}); });
    co_return co_await sig;
  }

  fan::event::waitv_t<std::optional<std::string>> co_save_file(std::string_view filter) {
    fan::event::signal_awaitable_t<std::optional<std::string>> sig;
    save_file(filter,
      [&](std::string_view p) { sig.signal(std::string{p}); },
      [&]()                   { sig.signal(std::nullopt);   });
    co_return co_await sig;
  }

  fan::event::waitv_t<std::optional<std::string>> co_open_folder() {
    fan::event::signal_awaitable_t<std::optional<std::string>> sig;
    open_folder(
      [&](std::string_view p) { sig.signal(std::string{p}); },
      [&]()                   { sig.signal(std::nullopt);   });
    co_return co_await sig;
  }
}