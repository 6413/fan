module;
#include <uv.h>
#include <fcntl.h>

module fan.event.types;

#if defined(_WIN32)
  #ifndef O_APPEND
    #define O_APPEND _O_APPEND
    #define O_CREAT  _O_CREAT
    #define O_EXCL   _O_EXCL
    #define O_RDONLY _O_RDONLY
    #define O_RDWR   _O_RDWR
    #define O_TRUNC  _O_TRUNC
    #define O_WRONLY _O_WRONLY
  #endif
#endif

namespace fan {
  const int fs_o_append      = O_APPEND;
  const int fs_o_creat       = O_CREAT;
  const int fs_o_excl        = O_EXCL;
  const int fs_o_rdonly      = O_RDONLY;
  const int fs_o_rdwr        = O_RDWR;
  const int fs_o_trunc       = O_TRUNC;
  const int fs_o_wronly      = O_WRONLY;

#ifdef O_DIRECTORY
  #undef fs_o_directory
  const int fs_o_directory   = O_DIRECTORY;
#else
  const int fs_o_directory   = 0;
#endif
#ifdef O_NONBLOCK
  #undef fs_o_nonblock
  const int fs_o_nonblock    = O_NONBLOCK;
#else
  const int fs_o_nonblock    = 0;
#endif

  const int eof = UV_EOF; 

  const int fs_in        = O_RDONLY;
  const int fs_out       = O_CREAT | O_WRONLY | O_TRUNC;
  const int fs_app       = O_WRONLY | O_APPEND;
  const int fs_trunc     = O_TRUNC;
  const int fs_ate       = O_RDWR;
  const int fs_nocreate  = O_EXCL;
  const int fs_noreplace = O_EXCL;

  const int fs_change = UV_CHANGE;
  const int fs_rename = UV_RENAME;
}