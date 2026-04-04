module;

#include <uv.h>

module fan.event.types;

namespace fan {
  extern const int fs_o_append      = UV_FS_O_APPEND;
  extern const int fs_o_creat       = UV_FS_O_CREAT;
  extern const int fs_o_excl        = UV_FS_O_EXCL;
  extern const int fs_o_filemap     = UV_FS_O_FILEMAP;
  extern const int fs_o_random      = UV_FS_O_RANDOM;
  extern const int fs_o_rdonly      = UV_FS_O_RDONLY;
  extern const int fs_o_rdwr        = UV_FS_O_RDWR;
  extern const int fs_o_sequential  = UV_FS_O_SEQUENTIAL;
  extern const int fs_o_short_lived = UV_FS_O_SHORT_LIVED;
  extern const int fs_o_temporary   = UV_FS_O_TEMPORARY;
  extern const int fs_o_trunc       = UV_FS_O_TRUNC;
  extern const int fs_o_wronly      = UV_FS_O_WRONLY;
  extern const int fs_o_direct      = UV_FS_O_DIRECT;
  extern const int fs_o_directory   = UV_FS_O_DIRECTORY;
  extern const int fs_o_dsync       = UV_FS_O_DSYNC; 
  extern const int fs_o_exlock      = UV_FS_O_EXLOCK; 
  extern const int fs_o_noatime     = UV_FS_O_NOATIME;
  extern const int fs_o_noctty      = UV_FS_O_NOCTTY;
  extern const int fs_o_nofollow    = UV_FS_O_NOFOLLOW;
  extern const int fs_o_nonblock    = UV_FS_O_NONBLOCK;
  extern const int fs_o_symlink     = UV_FS_O_SYMLINK;
  extern const int fs_o_sync        = UV_FS_O_SYNC;
  
  extern const int fs_in        = UV_FS_O_RDONLY;
  extern const int fs_out       = UV_FS_O_WRONLY | UV_FS_O_CREAT | UV_FS_O_TRUNC;
  extern const int fs_app       = UV_FS_O_WRONLY | UV_FS_O_APPEND;
  extern const int fs_trunc     = UV_FS_O_TRUNC;
  extern const int fs_ate       = UV_FS_O_RDWR;
  extern const int fs_nocreate  = UV_FS_O_EXCL;
  extern const int fs_noreplace = UV_FS_O_EXCL;

  extern const int fs_change = UV_CHANGE;
  extern const int fs_rename = UV_RENAME;
  extern const int eof = UV_EOF;
}