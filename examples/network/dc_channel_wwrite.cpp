#include <WITCH/WITCH.h>

#include <WITCH/PR/PR.h>

#include <WITCH/IO/IO.h>
#include <WITCH/IO/print.h>

#include <WITCH/VEC/VEC.h>
#include <WITCH/VEC/print.h>

void print(const char* format, ...) {
  IO_fd_t fd_stdout;
  IO_fd_set(&fd_stdout, FD_OUT);
  va_list argv;
  va_start(argv, format);
  IO_vprint(&fd_stdout, format, argv);
  va_end(argv);
}

#include <WITCH/NET/TCP/TCP.h>
#include <WITCH/NET/TCP/TLS/TLS.h>

#define ETC_HTTP_set_prefix HTTP
#include <WITCH/ETC/HTTP/HTTP.h>

#include <WITCH/STR/psu.h>
#include <WITCH/STR/pss.h>

#include <fan/pch.h>

typedef struct {
  NET_TCP_t* tcp;
  NET_TCP_layerid_t LayerStateID;
  NET_TCP_layerid_t LayerReadID;

  EV_timer_t timer;
}tcppile_t;

struct pile_t {
  tcppile_t tcp_pile;
  EV_t listener;

  uint64_t discordUpdateID;

  EV_event_t evevent;

  NET_TCP_peer_t* peer = nullptr;

  struct {
    fan::string message;
  }user_data;
};
pile_t pile;

void Send_discord_GetUpdate() {

  fan::string buf(R"(POST /api/v9/channels/1141704560956670003/messages HTTP/1.1
Host: discord.com
Authorization: MTExMzc4MzkzOTA3MzU3Njk5MQ.GVL9Ji.TmNPC1Y44G2Sk0WUBioMIdCyq_QaSJfexya-Rw
Content-Length: ###
Content-Type: application/json
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8
Accept-Language: en-US,en;q=0.5
DNT: 1
Connection: keep-alive
)");
  buf += "\n";

  //ADD Content-Length: #
  fan::string payload(R"({"content": ")");
  payload += pile.user_data.message;
  //payload += fan::to_string(fan::random::value_i64(0, 100));
  payload += "\"}";

  //// send payload
  buf += payload;

  buf = std::regex_replace(buf, std::regex("\n"), "\r\n");
  buf = std::regex_replace(buf, std::regex("###"), std::to_string(payload.size()));

  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = (void*)buf.c_str();
  Queue.DynamicPointer.size = buf.size();
  NET_TCP_write_loop(
    pile.peer,
    NET_TCP_GetWriteQueuerReferenceFirst(pile.peer),
    NET_TCP_QueueType_DynamicPointer_e,
    &Queue);
  //fan::print("\n\n\nPAYLOAD", payload, "\n\n");

}

void ConnectServer() {
  uint32_t ips[] = {
    0xa29f87e8
  };
  static uint32_t ip_index = 0;

  NET_addr_t addr;
  addr.ip = ips[ip_index];
  ip_index = (ip_index + 1) % (sizeof(ips) / sizeof(ips[0]));
  addr.port = 443;

  NET_TCP_sockopt_t sockopt;
  sockopt.level = IPPROTO_TCP;
  sockopt.optname = TCP_NODELAY;
  sockopt.value = 1;

  NET_TCP_peer_t* FillerPeer;
  sint32_t err = NET_TCP_connect(pile.tcp_pile.tcp, &FillerPeer, &addr, &sockopt, 1);
  if (err != 0) {
    PR_abort();
  }
}

uint32_t
cb_connstate(
  NET_TCP_peer_t* peer,
  uint8_t* sd,
  auto* pd,
  uint32_t flag
) {
  if (flag & NET_TCP_state_succ_e) {
    pile.peer = peer;
    NET_TCP_StartReadLayer(peer, pile.tcp_pile.LayerReadID);
    fan::print_no_endline("message:");
  }
  else do {
    pile.peer = nullptr;
    ConnectServer();

    if (!(flag & NET_TCP_state_init_e)) {
      break;
    }
  } while (0);

  //fan::print(flag);
  /*
  if (flag & NET_TCP_state_succ_e) {
    InitHTTPVariables(peer, pd);

    discordPacketTypeList_Open(&pd->PacketTypeList);

    NET_TCP_StartReadLayer(peer, pile.discord.LayerReadID);

    Send_discord_GetUpdate(peer, pd);
  }
  else do {
    discordPacketTypeList_Close(&pd->PacketTypeList);

    ConnectServer();

    if (!(flag & NET_TCP_state_init_e)) {
      break;
    }
  } while (0);*/

  return 0;
}
NET_TCP_layerflag_t
cb_read(
  NET_TCP_peer_t* peer,
  uint8_t* sd,
  auto* pd,
  NET_TCP_QueuerReference_t QueuerReference,
  uint32_t* type,
  NET_TCP_Queue_t* Queue
) {

  uint8_t* ReadData;
  uintptr_t ReadSize = 0;
  uint8_t _EventReadBuffer[0x1000];
  switch (*type) {
    case NET_TCP_QueueType_DynamicPointer_e: {
      ReadData = (uint8_t*)Queue->DynamicPointer.ptr;
      ReadSize = Queue->DynamicPointer.size;

      break;
    }
    case NET_TCP_QueueType_CloseHard_e: {
      return 0;
    }
    default: {
      fan::throw_error("type not handled");
    }
  }

  fan::string text(ReadData, ReadData + ReadSize);
  /*uint64_t offset = 0;
  while (1) {
    auto found = text.find("content\":", offset);
    if (found == std::string::npos) {
      break;
    }
    auto old_p = found;
    found = text.find(",", old_p);
    if (found == std::string::npos) {
      break;
    }
    fan::print(text.substr(old_p, found - old_p));
    offset += found;
  }*/
  fan::print(text);

  //fan::print("a");
  return 0;
}

void cb_initial_discord_timer(EV_t* listener, EV_timer_t* evt, uint32_t flag) {
  ConnectServer();

  EV_timer_stop(listener, evt);
}

int main() {
  fan::print("loading");
  EV_open(&pile.listener);

  pile.tcp_pile.tcp = NET_TCP_alloc(&pile.listener);

  {
    TLS_ctx_t tctx;
    bool r = TLS_ctx_generate(&tctx);
    if (r != 0) {
      PR_abort();
    }
    NET_TCP_TLS_add(pile.tcp_pile.tcp, &tctx);
  }

  static constexpr auto peer_data_size = 32;

  NET_TCP_extid_t extid = NET_TCP_EXT_new(pile.tcp_pile.tcp, 0, peer_data_size);
  pile.tcp_pile.LayerStateID = NET_TCP_layer_state_open(
    pile.tcp_pile.tcp,
    extid,
    (NET_TCP_cb_state_t)cb_connstate);
  pile.tcp_pile.LayerReadID = NET_TCP_layer_read_open(
    pile.tcp_pile.tcp,
    extid,
    (NET_TCP_cb_read_t)cb_read,
    0,
    0,
    0
  );

  NET_TCP_open(pile.tcp_pile.tcp);
  //
  EV_timer_init(&pile.tcp_pile.timer, 1, (EV_timer_cb_t)cb_initial_discord_timer);
  EV_timer_start(&pile.listener, &pile.tcp_pile.timer);

  IO_fd_t fd;
  IO_fd_set(&fd, STDIN_FILENO);

  EV_event_init_fd(&pile.evevent, &fd, [](EV_t*, EV_event_t*, uint32_t) {
    fan::string buffer;
    buffer.resize(0xfff);
    IO_fd_t fd;
    IO_fd_set(&fd, STDIN_FILENO);
    auto size = IO_read(&fd, buffer.data(), buffer.size());
    if (size < 0) {
      fan::throw_error("io_read failed");
    }
    else if (size > 0) {
      if (buffer[0] == '\r') {
        if (pile.peer != nullptr) {
          Send_discord_GetUpdate();
        }
        pile.user_data.message.clear();
        fan::print_no_endline("\nmessage:");
      }
      else {
        pile.user_data.message += buffer.substr(0, size);
      }
    }
    else {
      return;
    }
    }, EV_READ);

  EV_event_start(&pile.listener, &pile.evevent);

  EV_start(&pile.listener);

  return 0;
}
