#include fan_pch

#include <WITCH/WITCH.h>

#include <WITCH/PR/PR.h>

#include <WITCH/IO/IO.h>
#include <WITCH/IO/print.h>

#include <WITCH/VEC/VEC.h>
#include <WITCH/VEC/print.h>

#include <WITCH/NET/TCP/TCP.h>
#include <WITCH/NET/TCP/TLS/TLS.h>

#define ETC_HTTP_set_prefix HTTP
#include <WITCH/ETC/HTTP/HTTP.h>

#include <WITCH/STR/psu.h>
#include <WITCH/STR/pss.h>

#define ETC_HTTP_pall_set_prefix http_pall 
#include <WITCH/ETC/HTTP/pall.h>

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

  http_pall_t http_pall;

  uint32_t connect_ip = 0;
  std::string host_data;
  std::string url_rest;
  bool html_response_processed = false;
};
pile_t pile;

void send_data(const fan::string& data) {
  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = (void*)data.c_str();
  Queue.DynamicPointer.size = data.size();
  NET_TCP_write_loop(
    pile.peer,
    NET_TCP_GetWriteQueuerReferenceFirst(pile.peer),
    NET_TCP_QueueType_DynamicPointer,
    &Queue);
}

constexpr uint32_t ip_to_hex(const char* ip) {
  uint32_t res = 0;
  for (int i = 0; i < 4; ++i) {
    uint32_t octet = 0;
    while (*ip != '.' && *ip != '\0') {
      octet = octet * 10 + (*ip - '0');
      ++ip;
    }
    res = (res << 8) | octet;
    if (*ip != '\0') {
      ++ip;
    }
  }
  return res;
}

void ConnectServer() {
  uint32_t ips[] = {
    pile.connect_ip
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

void Send_discord_GetUpdate() {
  {
    fan::string buf(R"(GET )");
    buf += pile.url_rest;
    buf += " HTTP/1.1";
    buf += "\n";
    buf += pile.host_data;
    buf += "\n\n";

    buf = std::regex_replace(buf, std::regex("\n"), "\r\n");
    send_data(buf);
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

  uint8_t* ReadData = 0;
  uintptr_t ReadSize = 0;
  uint8_t _EventReadBuffer[0x1000];
  switch (*type) {
    case NET_TCP_QueueType_DynamicPointer: {
      ReadData = (uint8_t*)Queue->DynamicPointer.ptr;
      ReadSize = Queue->DynamicPointer.size;

      break;
    }
    case NET_TCP_QueueType_CloseHard: {
      return 0;
    }
    default: {
      fan::throw_error("type not handled:", *type);
    }
  }

  uintptr_t data_index = 0;
  uintptr_t prev_index = 0;
  http_pall_ParsedData parsed_data;
  while (data_index < ReadSize) {
    http_pall_ParseType parse_response = http_pall_Parse(&pile.http_pall, ReadData, ReadSize, &data_index, &parsed_data);
    switch (parse_response) {
      case http_pall_ParseType::http_pall_ParseType_Payload: {
        fan::string text(ReadData + prev_index, ReadData + data_index);
        /*static int x = 0;
        auto found = text.find("temperature_2m");
        if (found != std::string::npos) {
          if (x) {
            fan::string temp_data = text.substr(found);
            auto begin = temp_data.find("[") + 1;
            auto end = temp_data.find(",");
            fan::print(temp_data.substr(begin, end - begin));
          }
          x = 1;
        }*/
        fan::print(text);
        prev_index = data_index;
        break;
      }
    }
  }


  return 0;
}

void cb_initial_discord_timer(EV_t* listener, EV_timer_t* evt, uint32_t flag) {
  ConnectServer();

  EV_timer_stop(listener, evt);
}


uint32_t get_ip_unstripped(const std::string& hostname) {
  struct hostent* he = gethostbyname(hostname.c_str());
  if (he == nullptr) {
    return -1;
  }
  struct in_addr** addr_list = (struct in_addr**)he->h_addr_list;
  if (addr_list[0] != nullptr) {
    return ip_to_hex(inet_ntoa(*addr_list[0]));
  }
  return -1;
}

bool get_ip(const std::string& hostname, std::string& url_rest, std::string& host_data, uint32_t& ip) {
  auto http = hostname.find("http://");
  std::string temp = hostname;
  if (http != std::string::npos) {
    temp = hostname.substr(strlen("http://"));
  }
  auto https = hostname.find("https://");
  if (https != std::string::npos) {
    temp = hostname.substr(strlen("https://"));
  }
  auto first_dot = temp.find_first_of(".");
  if (first_dot == std::string::npos) {
    return 1;
  }
  auto found = temp.find_first_of("/", first_dot);
  if (found != std::string::npos) {
    url_rest = temp.substr(found);
    temp = temp.substr(0, found);
  }

  host_data = +"Host: " + temp;
  uint32_t received_ip = get_ip_unstripped(temp);
  if (received_ip == (uint32_t)-1) {
    url_rest = "";
    host_data = "";
    return 1;
  }
  if (url_rest.empty()) {
    url_rest = "/";
  }
  ip = received_ip;
  return 0;
}

int main() {
  if (get_ip("https://api.open-meteo.com/v1/forecast?latitude=60.736737225&longitude=24.775321449&hourly=temperature_2m", pile.url_rest, pile.host_data, pile.connect_ip)) {
    return 1;
  }

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

  http_pall_Open(&pile.http_pall);

  NET_TCP_open(pile.tcp_pile.tcp);
  //Send_discord_GetUpdate();
  //
  EV_timer_init(&pile.tcp_pile.timer, 1, (EV_timer_cb_t)cb_initial_discord_timer);
  EV_timer_start(&pile.listener, &pile.tcp_pile.timer);

  EV_start(&pile.listener);

  return 0;
}
