#include <fan/types/types.h>

#include <exception>
#include <coroutine>
#include <string>
#include <format>
#include <array>
import fan.fmt;
import fan.network;

#pragma pack(push, 1)

struct Protocol_AccountID_t {
  uint32_t i;
};
struct Protocol_SessionID_t {
  uint32_t i;
};

namespace tcp {
  struct ProtocolBasePacket_t {
    uint32_t ID;
    uint16_t Command;
  };
}

namespace udp {
  struct BasePacket_t {
    Protocol_SessionID_t SessionID;
    uint32_t ID;
    uint64_t IdentifySecret;
    uint16_t Command;
  };
}

// clien to server
namespace c2s {
  enum {
    KeepAlive,
    Request_Login,
    CreateChannel,
    JoinChannel,
    QuitChannel,
    Response_UDPIdentifySecret,
    Channel_ScreenShare_Share_InformationToViewSetFlag,
    Channel_ScreenShare_Share_InformationToViewMouseCoordinate,
    Channel_ScreenShare_View_ApplyToHostMouseCoordinate,
    Channel_ScreenShare_View_ApplyToHostMouseMotion,
    Channel_ScreenShare_View_ApplyToHostMouseButton,
    Channel_ScreenShare_View_ApplyToHostKeyboard
  };
};

struct ecps_backend_t;

namespace s2c {
  enum {
    KeepAlive,
    InformInvalidIdentify,
    Response_Login,
    CreateChannel_OK,
    CreateChannel_Error,
    JoinChannel_OK,
    JoinChannel_Error,
    KickedFromChannel,
    Request_UDPIdentifySecret,
    UseThisUDPIdentifySecret,
    Channel_ScreenShare_View_InformationToViewSetFlag,
    Channel_ScreenShare_View_InformationToViewMouseCoordinate,
    Channel_ScreenShare_Share_ApplyToHostMouseCoordinate,
    Channel_ScreenShare_Share_ApplyToHostMouseMotion,
    Channel_ScreenShare_Share_ApplyToHostMouseButton,
    Channel_ScreenShare_Share_ApplyToHostKeyboard,
    last
  };
  using cb_t = fan::event::task_t(*)(ecps_backend_t&, const tcp::ProtocolBasePacket_t&);
  std::array<cb_t, s2c::last> callbacks;
}

struct ecps_backend_t {

  ecps_backend_t() {
#define make_cb(code) [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t{ \
    code \
    co_return; \
  }
    s2c::callbacks = {
      make_cb({
        fan::print("tcp keep alive came");
        backend.tcp_keep_alive.reset();
      }),
      make_cb({

      }),
      make_cb({
        auto msg = co_await backend.tcp_client.read<fan_temporary_struct_maker(
          Protocol_AccountID_t AccountID;
          Protocol_SessionID_t SessionID;
        )>();
          fan::print_format(R"({{
  [SERVER] Response_login
  SessionID: {}
  AccountID: {}
}})", msg->SessionID.i, msg->AccountID.i);
      }),
      make_cb({ // join channel ok
        auto msg = co_await backend.tcp_client.read<fan_temporary_struct_maker(
          uint8_t Type;
          uint16_t ChannelID;
          uint32_t ChannelSessionID;
        )>();
          fan::print_format(R"({{
  [SERVER] JoinChannel_OK
  ID: {}
  ChannelID: {}
}})", base.ID, msg->ChannelID);
      }),
    };
#undef make_cb
  }

  fan::event::task_t connect(const std::string& ip, uint16_t port) {
    this->ip = ip;
    this->port = port;
    co_await tcp_client.connect(ip, port);
    udp_keep_alive.set_server(
      fan::network::buffer_t{
        (char*)&keep_alive_payload,
        (char*)&keep_alive_payload + sizeof(keep_alive_payload)
      },
      { ip, port }
    );
    task_udp_listen = udp_client.listen(
      fan::network::listen_address_t{ ip, port },
      [this, ip, port](const fan::network::udp_t& udp, const fan::network::udp_datagram_t& datagram) -> fan::event::task_t {
        fan::print("udp keep alive came");
        udp::BasePacket_t bp = *(udp::BasePacket_t*)datagram.data.data();
        udp_keep_alive.reset();
        co_return;
      }
    );
    tcp_keep_alive.reset();
    udp_keep_alive.reset();
    task_tcp_read = tcp_read();
  }
  
  fan::event::task_t tcp_write(int command) {
    static int id = 0;
    tcp::ProtocolBasePacket_t bp;
    bp.ID = id++;
    bp.Command = command;
    co_await tcp_client.write_raw(&bp, sizeof(bp));
    uint8_t random = 0;
    co_await tcp_client.write_raw(&random, sizeof(random));
  }

  fan::event::task_t login() {
    co_await tcp_write(c2s::Request_Login);
  }
  fan::event::task_t channel_create() {
    co_await tcp_write(c2s::CreateChannel);
  }
  fan::event::task_t channel_join(int channel_id) {
    co_await tcp_write(c2s::CreateChannel);
  }

  fan::event::task_t tcp_read() {
    while (1) {
      auto msg = co_await tcp_client.read<tcp::ProtocolBasePacket_t>();
      fan::print_format(R"({{
  ID: {}
  Command: {}
}})", msg->ID, msg->Command);
      s2c::callbacks[msg->Command](*this, msg.data);
    }
  }
  fan::event::task_t task_tcp_read;

  std::string ip;
  uint16_t port;
  fan::network::tcp_t tcp_client;
  fan::network::udp_t udp_client;
  fan::event::task_t task_udp_listen;

  tcp::ProtocolBasePacket_t keep_alive_payload{ .ID = 0, .Command = c2s::KeepAlive };
  fan::network::tcp_keep_alive_t tcp_keep_alive{ 
    tcp_client, 
    fan::network::buffer_t{
      (char*)&keep_alive_payload,
      (char*)&keep_alive_payload + sizeof(keep_alive_payload)
    }
  };
  fan::network::udp_keep_alive_t udp_keep_alive{ udp_client };

}ecps_backend;

#pragma pack(pop)

fan::event::task_t ecps_client() {
  try {
    co_await ecps_backend.connect("127.0.0.1", 43255);
    co_await ecps_backend.login();
    co_await ecps_backend.channel_create();
  }
  catch (std::exception& e) {
    fan::print_warning(std::string("client error:") + e.what());
  }
}

int main() {
  auto client = ecps_client();
  fan::event::loop();
}