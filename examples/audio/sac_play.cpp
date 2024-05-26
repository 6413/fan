#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef WITCH_INCLUDE_PATH
  #define WITCH_INCLUDE_PATH WITCH
#endif
#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH /usr/local/include
#endif

#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH,WITCH.h)
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#include _WITCH_PATH(IO/IO.h)
#include _WITCH_PATH(FS/FS.h)
#include _WITCH_PATH(A/A.h)
#include _WITCH_PATH(IO/print.h)
#include _WITCH_PATH(TH/TH.h)

#include _FAN_PATH(audio/audio.h)

void WriteOut(const char *format, ...){
	IO_fd_t fd_stdout;
	IO_fd_set(&fd_stdout, FD_OUT);
	va_list argv;
	va_start(argv, format);
	IO_vprint(&fd_stdout, format, argv);
	va_end(argv);
}

int main(int argc, char **argv){
  if(argc != 2){
    return 0;
  }

  fan::system_audio_t system_audio;
  if(system_audio.Open() != 0){
    __abort();
  }

  fan::audio_t audio;
  audio.bind(&system_audio);

  fan::audio_t::piece_t piece;
  sint32_t err = audio.Open(&piece, "audio/friday.sac", 0);
  if(err != 0){
    WriteOut("piece open failed %lx\n", err);
    return 0;
  }

  {
    fan::audio_t::PropertiesSoundPlay_t p;
    p.Flags.Loop = true;
    p.GroupID = 0;
    audio.SoundPlay(&piece, &p);
  }

  IO_fd_t fd_in;
  IO_fd_set(&fd_in, STDIN_FILENO);
  while(1){
    uint8_t buffer[0x200];
    IO_ssize_t r = IO_read(&fd_in, buffer, sizeof(buffer));
    if(r < 0){
      break;
    }
  }

  audio.unbind();

  system_audio.Close();

  return 0;
}
