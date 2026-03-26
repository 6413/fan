module;

#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <memory>

export module fan.file_dialog;

import fan.event;

export namespace fan::graphics {

  using file_cb_t      = std::function<void(std::string_view)>;
  using files_cb_t     = std::function<void(std::vector<std::string_view>)>;
  using dismissed_cb_t = std::function<void()>;

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

  dialog_t& open_file  (std::string_view filter, file_cb_t  on_done, dismissed_cb_t on_dismissed = {});
  dialog_t& open_files (std::string_view filter, files_cb_t on_done, dismissed_cb_t on_dismissed = {});
  dialog_t& save_file  (std::string_view filter, file_cb_t  on_done, dismissed_cb_t on_dismissed = {});
  dialog_t& open_folder(                         file_cb_t  on_done, dismissed_cb_t on_dismissed = {});

  fan::event::task_value_t<std::optional<std::string>> co_open_file  (std::string_view filter);
  fan::event::task_value_t<std::vector<std::string>>   co_open_files (std::string_view filter);
  fan::event::task_value_t<std::optional<std::string>> co_save_file  (std::string_view filter);
  fan::event::task_value_t<std::optional<std::string>> co_open_folder();
}