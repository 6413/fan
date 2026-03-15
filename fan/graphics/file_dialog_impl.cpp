module;

#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <functional>

#include <fan/nativefiledialog/nfd.h>

#include <uv.h>

module fan.file_dialog;

import fan.event;

void fan::graphics::file_open_dialog_t::load(const std::string& filter_list, std::string* out) {
  worker_thread = std::thread([&, out, filter_list] {
    nfdchar_t* nfd_out_path = NULL;
    nfdresult_t result = NFD_OpenDialog(filter_list.c_str(), NULL, &nfd_out_path);
    if (result == NFD_OKAY) {
      std::lock_guard<std::mutex> lock(out_path_mutex);
      out_path = nfd_out_path;
      *out = nfd_out_path;
    }
    free(nfd_out_path);
    finished = true;
  });
  worker_thread.detach();
}

void fan::graphics::file_open_dialog_t::load(const std::string& filter_list, std::vector<std::string>* out) {
  worker_thread = std::thread([&, out, filter_list] {
    nfdpathset_t outPaths;
    nfdresult_t result = NFD_OpenDialogMultiple(filter_list.c_str(), NULL, &outPaths);
    if (result == NFD_OKAY) {
      std::lock_guard<std::mutex> lock(out_path_mutex);
      size_t count = NFD_PathSet_GetCount(&outPaths);
      for (size_t i = 0; i < count; ++i) {
        nfdchar_t* nfd_out_path = NFD_PathSet_GetPath(&outPaths, i);
        out->push_back(nfd_out_path);
      }
      NFD_PathSet_Free(&outPaths);
    }
    finished = true;
  });
  worker_thread.detach();
}

bool fan::graphics::file_open_dialog_t::is_finished() {
  if (!finished) return false;
  finished = false;
  return true;
}

void fan::graphics::file_save_dialog_t::save(const std::string& filter_list, std::string* out) {
  std::thread([&, out, filter_list] {
    nfdchar_t* nfd_out_path = NULL;
    nfdresult_t result = NFD_SaveDialog(filter_list.c_str(), NULL, &nfd_out_path);
    if (result == NFD_OKAY && nfd_out_path != NULL) {
      std::lock_guard<std::mutex> lock(out_path_mutex);
      out_path = nfd_out_path;
      *out = nfd_out_path;
    }
    free(nfd_out_path);
    finished = true;
  }).detach();
}

bool fan::graphics::file_save_dialog_t::is_finished() {
  if (!finished) return false;
  finished = false;
  return true;
}

void fan::graphics::folder_open_dialog_t::open(std::string& out, std::function<void()> cb) {
  target_out = &out;
  on_done = std::move(cb);
  uv_async_init(fan::event::get_loop(), &async_, [](uv_async_t* h) {
    auto* self = static_cast<folder_open_dialog_t*>(h->data);
    {
      std::lock_guard<std::mutex> lock(self->out_path_mutex);
      if (self->target_out) *self->target_out = self->out_path;
    }
    if (self->on_done) { self->on_done(); self->on_done = {}; }
    uv_close(reinterpret_cast<uv_handle_t*>(h), nullptr);
  });
  async_.data = this;
  worker_thread = std::thread([this] {
    nfdchar_t* p = nullptr;
    if (NFD_PickFolder(nullptr, &p) == NFD_OKAY) {
      std::lock_guard<std::mutex> lock(out_path_mutex);
      out_path = p;
    }
    free(p);
    uv_async_send(&async_);
  });
  worker_thread.detach();
}

bool fan::graphics::folder_open_dialog_t::is_finished() {
  if (!finished) return false;
  finished = false;
  {
    std::lock_guard<std::mutex> lock(out_path_mutex);
    if (target_out) *target_out = out_path;
  }
  if (on_done) { on_done(); on_done = {}; }
  return true;
}