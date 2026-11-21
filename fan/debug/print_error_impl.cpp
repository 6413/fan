module;

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif
#include <signal.h>

module fan.print.error;

import std;

// NOTE: Do NOT #include <ctime> here. `import std;` already provides
// std::chrono, std::time_t, std::tm. See gcc_bug.txt.

namespace fan {
  log_t& get_error_log() {
    static log_t log;
    return log;
  }

  void write_error_to_disk(const std::string& msg) {
    auto& log = get_error_log();
    auto now = std::chrono::system_clock::now();
    // NOTE: use std::format chrono (no <ctime> / localtime_r) to stay
    // compatible with `import std;` + GCC -freflection (see gcc_bug.txt).
    std::string ts;
    try {
      ts = std::format("{:%Y-%m-%d %H:%M:%S}",
        std::chrono::floor<std::chrono::seconds>(now));
    }
    catch (...) {
      ts = "unknown-time";
    }
    std::string new_entry = ts + " - " + msg + '\n';

    std::lock_guard<std::mutex> lock(log.mtx);
    std::ifstream in(log.filename, std::ios::binary);
    std::string existing((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    std::string tmp_path = log.filename + ".tmp";
    {
      std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
      out << new_entry << existing;
    }
    std::filesystem::rename(tmp_path, log.filename);
  }

  void push_memory_log(const std::string& tag, const std::string& msg, log_level_e level) {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    log.buffer.push_back({tag, msg, level});
    if (log.buffer.size() > log.max_size) {
      log.buffer.pop_front();
    }
    ++log.total_logs_pushed;
  }
  void push_memory_log(const std::string& msg, log_level_e level) {
    push_memory_log("", msg, level);
  }

  std::vector<log_entry_t> dump_memory_logs() {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    return std::vector<log_entry_t>(log.buffer.begin(), log.buffer.end());
  }

  std::vector<log_entry_t> dump_memory_logs_since(std::uint64_t& cursor) {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    
    std::vector<log_entry_t> result;
    if (cursor >= log.total_logs_pushed) return result;
    
    std::size_t available = log.total_logs_pushed - cursor;
    if (available > log.buffer.size()) {
      available = log.buffer.size();
    }
    
    auto it = log.buffer.end() - available;
    result.assign(it, log.buffer.end());
    
    cursor = log.total_logs_pushed;
    return result;
  }

  void clear_memory_logs() {
    auto& log = get_error_log();
    std::lock_guard<std::mutex> lock(log.mtx);
    log.buffer.clear();
  }

  // ---- crash logging -------------------------------------------------------------------------------------------
  // A crash (access violation, abort(), std::terminate) ends the process before anything else gets to write, so
  // these handlers put what happened into fan_log.txt first, with the stack, and on Windows a fan_crash.dmp next
  // to it (open with the build's .pdb to see the exact line). Best effort: the process is already broken
  namespace {
    std::atomic<bool> crash_logged{false};

    // like write_error_to_disk, but never blocks: the crashing thread may be the one holding the log's lock
    void write_crash_to_disk(const std::string& msg) {
      auto& log = get_error_log();
      bool locked = log.mtx.try_lock();
      try {
        std::string ts;
        try {
          ts = std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));
        }
        catch (...) {
          ts = "unknown-time";
        }
        std::ifstream in(log.filename, std::ios::binary);
        std::string existing((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        std::ofstream out(log.filename, std::ios::binary | std::ios::trunc);
        out << ts << " - " << msg << '\n' << existing;
      }
      catch (...) {}
      if (locked) {
        log.mtx.unlock();
      }
    }

    void log_crash(const std::string& what) {
      // only the first: a crash inside the handler, or abort() after terminate, would only repeat it
      if (crash_logged.exchange(true)) {
        return;
      }
      std::string msg = "CRASH: " + what;
      try {
        msg += std::format("\n{}", std::stacktrace::current(1));
      }
      catch (...) {}
      write_crash_to_disk(msg);
    }

  #if defined(_WIN32)
    std::string module_offset(const void* address) {
      HMODULE module = nullptr;
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)address, &module) && module) {
        char path[MAX_PATH] = {};
        GetModuleFileNameA(module, path, MAX_PATH);
        return std::format("{}+0x{:X}", std::filesystem::path(path).filename().string(),
          (std::uintptr_t)address - (std::uintptr_t)module);
      }
      return std::format("0x{:X}", (std::uintptr_t)address);
    }

    // dbghelp is loaded only here, so a program that never crashes never loads it
    // MINIDUMP_EXCEPTION_INFORMATION, which dbghelp.h declares packed to 4 bytes
  #pragma pack(push, 4)
    struct exception_info_t {
      DWORD thread_id;
      EXCEPTION_POINTERS* pointers;
      BOOL client_pointers;
    };
  #pragma pack(pop)
    static_assert(sizeof(exception_info_t) == 16);

    void write_minidump(EXCEPTION_POINTERS* info) {
      using write_dump_t = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, exception_info_t*, void*, void*);
      HMODULE dbghelp = LoadLibraryA("dbghelp.dll");
      if (!dbghelp) {
        return;
      }
      auto write_dump = (write_dump_t)GetProcAddress(dbghelp, "MiniDumpWriteDump");
      HANDLE file = CreateFileA("fan_crash.dmp", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (write_dump && file != INVALID_HANDLE_VALUE) {
        exception_info_t ex{GetCurrentThreadId(), info, FALSE};
        // MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithThreadInfo: the stacks and what they point at, a few MB
        write_dump(GetCurrentProcess(), GetCurrentProcessId(), file, 0x00000040 | 0x00001000, &ex, nullptr, nullptr);
      }
      if (file != INVALID_HANDLE_VALUE) {
        CloseHandle(file);
      }
    }

    LONG WINAPI unhandled_exception(EXCEPTION_POINTERS* info) {
      if (!crash_logged.load()) {
        const EXCEPTION_RECORD* r = info->ExceptionRecord;
        std::string what = std::format("exception 0x{:08X} at {} (thread {})", (unsigned)r->ExceptionCode,
          module_offset(r->ExceptionAddress), GetCurrentThreadId());
        if (r->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r->NumberParameters >= 2) {
          what += std::format(", {} address 0x{:X}", r->ExceptionInformation[0] == 0 ? "reading" : r->ExceptionInformation[0] == 1 ? "writing" : "executing",
            (std::uintptr_t)r->ExceptionInformation[1]);
        }
        log_crash(what + ", dump in fan_crash.dmp");
        write_minidump(info);
      }
      return EXCEPTION_CONTINUE_SEARCH;
    }
  #else
    void on_signal(int sig) {
      log_crash(std::format("signal {}", sig));
      ::signal(sig, SIG_DFL);
      ::raise(sig);
    }
  #endif

    void on_abort(int) {
      log_crash("abort() was called");
    }

    void install_crash_handlers() {
    #if defined(_WIN32)
      SetUnhandledExceptionFilter(unhandled_exception);
      ::signal(SIGABRT, on_abort);
    #else
      for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL}) {
        ::signal(sig, on_signal);
      }
      ::signal(SIGABRT, on_signal);
    #endif
      static std::terminate_handler previous = std::set_terminate([] {
        std::string what = "std::terminate";
        if (auto e = std::current_exception()) {
          try {
            std::rethrow_exception(e);
          }
          catch (const std::exception& ex) {
            what += std::string(", uncaught exception: ") + ex.what();
          }
          catch (const exception_t& ex) {
            what += std::string(", uncaught fan error: ") + (ex.reason ? ex.reason : "");
          }
          catch (...) {
            what += ", uncaught exception of unknown type";
          }
        }
        log_crash(what);
        if (previous) {
          previous();
        }
        std::abort();
      });
    }

    // any program that links fan's logging gets the handlers before main()
    const bool crash_handlers_installed = (install_crash_handlers(), true);
  }

  void throw_error_impl(const char* reason) {
    std::string res(reason);
    if (res.size()) {
      res += std::format("\n{}", std::stacktrace::current());
      write_error_to_disk(res);
      push_memory_log(res, log_level_e::error);
    }
#if __cpp_exceptions
    throw exception_t{.reason = reason};
#endif
  }
}