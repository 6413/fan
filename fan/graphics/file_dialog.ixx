export module fan.file_dialog;

import std;

export namespace fan::graphics {
  struct file_open_dialog_t {
    std::atomic<bool> finished{ false };
    std::string out_path;
    std::mutex out_path_mutex;
    std::thread worker_thread;
    void load(const std::string& filter_list, std::string* out);
    void load(const std::string& filter_list, std::vector<std::string>* out);
    bool is_finished();
  };
  struct file_save_dialog_t {
    std::atomic<bool> finished{ false };
    std::string out_path;
    std::mutex out_path_mutex;
    void save(const std::string& filter_list, std::string* out);
    bool is_finished();
  };
}