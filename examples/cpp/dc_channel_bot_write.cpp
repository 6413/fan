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

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#define FAN_INCLUDE_PATH C:/libs/fan/include
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

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

void send_data(const fan::string& data) {
  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = (void*)data.c_str();
  Queue.DynamicPointer.size = data.size();
  NET_TCP_write_loop(
    pile.peer,
    NET_TCP_GetWriteQueuerReferenceFirst(pile.peer),
    NET_TCP_QueueType_DynamicPointer_e,
    &Queue);
}

enum ws_opcode {
  ws_opcode_continuation = 0x0,
  ws_opcode_text = 0x1,
  ws_opcode_binary = 0x2,
  ws_opcode_rsv3 = 0x3,
  ws_opcode_rsv4 = 0x4,
  ws_opcode_rsv5 = 0x5,
  ws_opcode_rsv6 = 0x6,
  ws_opcode_rsv7 = 0x7,
  ws_opcode_close = 0x8,
  ws_opcode_ping = 0x9,
  ws_opcode_pong = 0xA,
  ws_opcode_control_rsvb = 0xB,
  ws_opcode_control_rsvc = 0xC,
  ws_opcode_control_rsvd = 0xD,
  ws_opcode_control_rsve = 0xE,
  ws_opcode_control_rsvf = 0xF
};

const unsigned char WS_MASKBIT = (1u << 7u);
const unsigned char WS_FINBIT = (1u << 7u);
const unsigned char WS_PAYLOAD_LENGTH_MAGIC_LARGE = 126;
const unsigned char WS_PAYLOAD_LENGTH_MAGIC_HUGE = 127;
const size_t WS_MAX_PAYLOAD_LENGTH_SMALL = 125;
const size_t WS_MAX_PAYLOAD_LENGTH_LARGE = 65535;
const size_t MAXHEADERSIZE = sizeof(uint64_t) + 2;

size_t create_websock_header(unsigned char* outbuf, size_t sendlength, uint8_t opcode = ws_opcode::ws_opcode_binary) {
  size_t pos = 0;
  outbuf[pos++] = WS_FINBIT | opcode;

  if (sendlength <= WS_MAX_PAYLOAD_LENGTH_SMALL)
  {
    outbuf[pos++] = (unsigned int)sendlength;
  }
  else if (sendlength <= WS_MAX_PAYLOAD_LENGTH_LARGE)
  {
    outbuf[pos++] = WS_PAYLOAD_LENGTH_MAGIC_LARGE;
    outbuf[pos++] = (sendlength >> 8) & 0xff;
    outbuf[pos++] = sendlength & 0xff;
  }
  else
  {
    outbuf[pos++] = WS_PAYLOAD_LENGTH_MAGIC_HUGE;
    const uint64_t len = sendlength;
    for (int i = sizeof(uint64_t) - 1; i >= 0; i--)
      outbuf[pos++] = ((len >> i * 8) & 0xff);
  }

  outbuf[1] |= WS_MASKBIT;
  outbuf[pos++] = 0;
  outbuf[pos++] = 0;
  outbuf[pos++] = 0;
  outbuf[pos++] = 0;

  return pos;
}

void construct_websock_frame(const std::string& data) {
  unsigned char out[MAXHEADERSIZE];
  size_t s = create_websock_header(out, data.length(), ws_opcode::ws_opcode_binary);

  send_data(std::string((const char*)out, s));
  send_data(data);
}

void f() {
  construct_websock_frame(R"({
  "op": 2,
  "d": {
    "token": "MTE0MDMyOTMzOTE0NjIxOTY3MQ.GciwqM.W3r6PIn4nJdG9M1Xhz99Izp8LlTZdqBMfT0wNM",
    "intents": 512,
    "properties": {
      "os": "something",
      "browser": "wfb",
      "device": "durum"
    }
  }
}
)");
}

void Send_discord_GetUpdate() {
  {
    fan::string buf(R"(GET /?v=6&encoding=json HTTP/1.1
Host: gateway.discord.gg
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13)");

    buf += "\n\n";

    buf = std::regex_replace(buf, std::regex("\n"), "\r\n");
    send_data(buf);
  }
  {
    construct_websock_frame(R"({
  "op": 2,
  "d": {
    "token": "MTE0MDMyOTMzOTE0NjIxOTY3MQ.GciwqM.W3r6PIn4nJdG9M1Xhz99Izp8LlTZdqBMfT0wNM",
    "intents": 512,
    "properties": {
      "os": "something",
      "browser": "wfb",
      "device": "durum"
    }
  }
}
)");
  }
}

void ConnectServer() {
  uint32_t ips[] = {
    0xa29f85ea
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
    Send_discord_GetUpdate();
    //fan::print_no_endline("message:");
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

 /* if (text.find("MESSAGE_CREATE") != std::string::npos) {
    uint64_t offset = 0;
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
    }
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
  EV_open(&pile.listener);

  pile.tcp_pile.tcp = NET_TCP_alloc(&pile.listener);

  TLS_ctx_t tctx;
  {
    bool r = TLS_ctx_generate(&tctx);
    if (r != 0) {
      PR_abort();
    }
    NET_TCP_TLS_add(pile.tcp_pile.tcp, &tctx);
  }

  static constexpr auto peer_data_size = 136;

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
  //Send_discord_GetUpdate();
  //
  EV_timer_init(&pile.tcp_pile.timer, 1, (EV_timer_cb_t)cb_initial_discord_timer);
  EV_timer_start(&pile.listener, &pile.tcp_pile.timer);

 /* IO_fd_t fd;
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

  EV_event_start(&pile.listener, &pile.evevent);*/

  EV_start(&pile.listener);

  return 0;
}
