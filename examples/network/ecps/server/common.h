#include _WITCH_PATH(MEM/MEM.h)
#include _WITCH_PATH(STR/common/common.h)
#include _WITCH_PATH(IO/IO.h)
#include _WITCH_PATH(IO/print.h)
#include _WITCH_PATH(HASH/SHA.h)
#include _WITCH_PATH(RAND/RAND.h)
#include _WITCH_PATH(VEC/VEC.h)
#include _WITCH_PATH(EV/EV.h)
#include _WITCH_PATH(NET/NET.h)
#include _WITCH_PATH(NET/TCP/TCP.h)

#include "../prot.h"

// filler
static fan::event::task_t default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
  __abort();
  return {};
}

void _print(uint32_t fd, const char *format, ...){
  IO_fd_t fd_stdout;
  IO_fd_set(&fd_stdout, fd);
  va_list argv;
  va_start(argv, format);
  IO_vprint(&fd_stdout, format, argv);
  va_end(argv);
}
#define WriteInformation(...) _print(FD_OUT, __VA_ARGS__)
#define WriteError(...) _print(FD_ERR, __VA_ARGS__)

void TCP_write_DynamicPointer(NET_TCP_peer_t *peer, void *Data, uintptr_t Size){
  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = Data;
  Queue.DynamicPointer.size = Size;
  NET_TCP_write_loop(
    peer,
    NET_TCP_GetWriteQueuerReferenceFirst(peer),
    NET_TCP_QueueType_DynamicPointer,
    &Queue);
}
