module;

module fan.event.types;

namespace fan {
  extern const int fs_o_append      = 1024;        // O_APPEND
  extern const int fs_o_creat       = 64;          // O_CREAT
  extern const int fs_o_excl        = 128;         // O_EXCL
  extern const int fs_o_filemap     = 0;
  extern const int fs_o_random      = 0;
  extern const int fs_o_rdonly      = 0;           // O_RDONLY
  extern const int fs_o_rdwr        = 2;           // O_RDWR
  extern const int fs_o_sequential  = 0;
  extern const int fs_o_short_lived = 0;
  extern const int fs_o_temporary   = 0;
  extern const int fs_o_trunc       = 512;         // O_TRUNC
  extern const int fs_o_wronly      = 1;           // O_WRONLY
  extern const int fs_o_direct      = 0;           // platform-specific
  extern const int fs_o_directory   = 65536;       // O_DIRECTORY
  extern const int fs_o_dsync       = 4096;        // O_DSYNC (platform-dependent)
  extern const int fs_o_exlock      = 0;
  extern const int fs_o_noatime     = 262144;      // O_NOATIME (Linux-specific)
  extern const int fs_o_noctty      = 256;         // O_NOCTTY
  extern const int fs_o_nofollow    = 131072;      // O_NOFOLLOW
  extern const int fs_o_nonblock    = 2048;        // O_NONBLOCK
  extern const int fs_o_symlink     = 0;
  extern const int fs_o_sync        = 1052672;     // O_SYNC (platform-dependent)
  
  extern const int fs_in        = 0;               // O_RDONLY
  extern const int fs_out       = 64 | 1 | 512;    // O_CREAT | O_WRONLY | O_TRUNC
  extern const int fs_app       = 1 | 1024;        // O_WRONLY | O_APPEND
  extern const int fs_trunc     = 512;             // O_TRUNC
  extern const int fs_ate       = 2;               // O_RDWR
  extern const int fs_nocreate  = 128;             // O_EXCL
  extern const int fs_noreplace = 128;             // O_EXCL

  extern const int fs_change = 0;
  extern const int fs_rename = 0;
  extern const int eof = 0;
}