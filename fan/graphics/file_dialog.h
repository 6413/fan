#pragma once

#include <atomic>
#include <string>
#include <mutex>
#include <thread>

#include <fan/nativefiledialog/nfd.h>

namespace fan {
  namespace graphics {
    struct file_open_dialog_t {
      std::atomic<bool> finished{ false };
      std::string out_path;
      std::mutex out_path_mutex;
      std::thread worker_thread;

      void load(const std::string& filter_list, std::string* out) {
        worker_thread = std::thread([&, out, filter_list] {
          nfdchar_t* nfd_out_path = NULL;
          nfdresult_t result = NFD_OpenDialog(filter_list.c_str(), NULL, &nfd_out_path);
          if (result == NFD_OKAY) {
            std::lock_guard<std::mutex> lock(out_path_mutex);
            out_path = nfd_out_path;
            *out = nfd_out_path;
          }

          free(nfd_out_path);
          finished.store(true);
          });
          worker_thread.detach();
      }

     void load(const std::string& filter_list, std::vector<std::string>* out) {
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

        finished.store(true);
    });
    worker_thread.detach();
}

      bool is_finished() {
        return finished.load();
      }
    };
    struct file_save_dialog_t {
      std::atomic<bool> finished{ false };
      std::string out_path;
      std::mutex out_path_mutex;

      void save(const std::string& filter_list, std::string* out) {
        std::thread([&, out, filter_list] {
          nfdchar_t* nfd_out_path = NULL;
          nfdresult_t result = NFD_SaveDialog(filter_list.c_str(), NULL, &nfd_out_path);
          if (result == NFD_OKAY && nfd_out_path != NULL) {
            std::lock_guard<std::mutex> lock(out_path_mutex);
            out_path = nfd_out_path;
            *out = nfd_out_path;
          }

          free(nfd_out_path);
          finished.store(true);
        }).detach();
      }

      bool is_finished() {
        return finished.load();
      }
    };
  }
}