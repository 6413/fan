module;

#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <functional>

#include <uv.h>

export module fan.file_dialog;

export namespace fan::graphics {
  struct file_open_dialog_t {
    bool finished = false;
    std::string out_path;
    std::mutex out_path_mutex;
    std::thread worker_thread;
    void load(const std::string& filter_list, std::string* out);
    void load(const std::string& filter_list, std::vector<std::string>* out);
    bool is_finished();
  };
  struct file_save_dialog_t {
    bool finished = false;
    std::string out_path;
    std::mutex out_path_mutex;
    void save(const std::string& filter_list, std::string* out);
    bool is_finished();
  };
  struct folder_open_dialog_t {
    bool finished = false;
    std::string out_path;
    std::mutex out_path_mutex;
    std::thread worker_thread;
    std::function<void()> on_done;
    std::string* target_out = nullptr;
    uv_async_t async_ {};
    void open(std::string& out, std::function<void()> on_done = {});
    bool is_finished();
  };
}