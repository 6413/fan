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

import fan.event;
import std;

namespace fan::graphics {
  static const char* dialog_filter(const std::string& filter) {
    return filter.empty() ? nullptr : filter.c_str();
  }
  static const char* dialog_default_path(std::string& storage) {
  #if defined(fan_platform_windows)
    return nullptr;
  #else
    storage = std::filesystem::current_path().string();
    return storage.c_str();
  #endif
  }

  struct dialog_t {
    std::atomic<bool> pending_{false};
    std::mutex mtx;
    std::string last_p;

    bool pending() const { return pending_.load(); }
    void cancel() { } 
    std::string last_path() { std::lock_guard lock(mtx); return last_p; }

    template <typename Fn>
    void launch(Fn&& nfd_fn, auto on_done, auto on_dismiss) {
      pending_.store(true);
      fan::event::thread_create([this, nfd_fn = std::forward<Fn>(nfd_fn), on_done = std::move(on_done), on_dismiss = std::move(on_dismiss)]() mutable {
        auto result = nfd_fn();
        fan::event::post_to_main([this, result = std::move(result), on_done = std::move(on_done), on_dismiss = std::move(on_dismiss)]() mutable {
          pending_.store(false);
          if (result) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*result)>, std::string>) {
              { std::lock_guard lock(mtx); last_p = *result; }
              if (on_done) on_done(*result);
            } else {
              if (on_done) {
                std::vector<std::string_view> views(result->begin(), result->end());
                on_done(views);
              }
            }
          } else if (on_dismiss) {
            on_dismiss();
          }
        });
      });
    }

    dialog_t& open_file(std::string_view filter, file_cb_t on_done, dismissed_cb_t on_dismissed = {}) {
      launch([f = std::string(filter)]() -> std::optional<std::string> {
        std::string storage; nfdchar_t* p = nullptr;
        if (NFD_OpenDialog(dialog_filter(f), dialog_default_path(storage), &p) == NFD_OKAY && p) {
          std::string res = p; std::free(p); return res;
        }
        return std::nullopt;
      }, std::move(on_done), std::move(on_dismissed));
      return *this;
    }

    dialog_t& open_files(std::string_view filter, files_cb_t on_done, dismissed_cb_t on_dismissed = {}) {
      launch([f = std::string(filter)]() -> std::optional<std::vector<std::string>> {
        std::string storage; nfdpathset_t ps;
        if (NFD_OpenDialogMultiple(dialog_filter(f), dialog_default_path(storage), &ps) == NFD_OKAY) {
          std::vector<std::string> res;
          std::size_t n = NFD_PathSet_GetCount(&ps);
          res.reserve(n);
          for (std::size_t i = 0; i < n; ++i) res.emplace_back(NFD_PathSet_GetPath(&ps, i));
          NFD_PathSet_Free(&ps);
          return res;
        }
        return std::nullopt;
      }, std::move(on_done), std::move(on_dismissed));
      return *this;
    }

    dialog_t& save_file(std::string_view filter, file_cb_t on_done, dismissed_cb_t on_dismissed = {}) {
      launch([f = std::string(filter)]() -> std::optional<std::string> {
        std::string storage; nfdchar_t* p = nullptr;
        if (NFD_SaveDialog(dialog_filter(f), dialog_default_path(storage), &p) == NFD_OKAY && p) {
          std::string res = p; std::free(p); return res;
        }
        return std::nullopt;
      }, std::move(on_done), std::move(on_dismissed));
      return *this;
    }

    dialog_t& open_folder(file_cb_t on_done, dismissed_cb_t on_dismissed = {}) {
      launch([]() -> std::optional<std::string> {
        std::string storage; nfdchar_t* p = nullptr;
        if (NFD_PickFolder(dialog_default_path(storage), &p) == NFD_OKAY && p) {
          std::string res = p; std::free(p); return res;
        }
        return std::nullopt;
      }, std::move(on_done), std::move(on_dismissed));
      return *this;
    }
  };

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
    open_file(filter, [&](std::string_view p) { sig.signal(std::string{p}); }, [&] { sig.signal(std::nullopt); });
    co_return co_await sig;
  }

  fan::event::waitv_t<std::vector<std::string>> co_open_files(std::string_view filter) {
    fan::event::signal_awaitable_t<std::vector<std::string>> sig;
    open_files(filter, [&](std::vector<std::string_view> paths) {
      std::vector<std::string> out; out.reserve(paths.size());
      for (auto p : paths) out.emplace_back(p);
      sig.signal(std::move(out));
    }, [&] { sig.signal({}); });
    co_return co_await sig;
  }

  fan::event::waitv_t<std::optional<std::string>> co_save_file(std::string_view filter) {
    fan::event::signal_awaitable_t<std::optional<std::string>> sig;
    save_file(filter, [&](std::string_view p) { sig.signal(std::string{p}); }, [&] { sig.signal(std::nullopt); });
    co_return co_await sig;
  }

  fan::event::waitv_t<std::optional<std::string>> co_open_folder() {
    fan::event::signal_awaitable_t<std::optional<std::string>> sig;
    open_folder([&](std::string_view p) { sig.signal(std::string{p}); }, [&] { sig.signal(std::nullopt); });
    co_return co_await sig;
  }
}