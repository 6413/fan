struct ecps_backend_t {

#include "prot.h"

  ecps_backend_t() {
    __dme_get(Protocol_S2C, KeepAlive) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      fan::print("tcp keep alive came");
      backend.tcp_keep_alive.reset();
      co_return;
    };
    __dme_get(Protocol_S2C, InformInvalidIdentify) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::InformInvalidIdentify_t>();
      if (msg->ClientIdentify != identify_secret) {
        co_return;
      }
      identify_secret = msg->ServerIdentify;
      fan::print("inform invalid identify came");
      co_return;
    };
    
    __dme_get(Protocol_S2C, Response_Login) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::Response_Login_t>();
      fan::print_format(R"({{
  [SERVER] Response_login
  SessionID: {}
  AccountID: {}
}})", msg->SessionID.i, msg->AccountID.i);
      backend.session_id = msg->SessionID;
    };
    __dme_get(Protocol_S2C, CreateChannel_OK) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::CreateChannel_OK_t>();
      fan::print_format(R"({{
[SERVER] CreateChannel_OK
ID: {}
ChannelID: {}
}})", base.ID, msg->ChannelID.i);

      auto it = backend.pending_requests.find(base.ID);
      if (it != backend.pending_requests.end()) {
        it->second.channel_id = msg->ChannelID;
        it->second.completed = true;
        if (it->second.continuation) {
          it->second.continuation.resume();
        }
      }
    };
    __dme_get(Protocol_S2C, JoinChannel_OK) = [this](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_OK_t>();
      fan::print_format(R"({{
  [SERVER] JoinChannel_OK
  ID: {}
  ChannelID: {}
}})", base.ID, msg->ChannelID.i);
      channel_info.front().session_id = msg->ChannelSessionID;
    };
    __dme_get(Protocol_S2C, JoinChannel_Error) = [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      auto msg = co_await backend.tcp_client.read<Protocol_S2C_t::JoinChannel_Error_t>();
      fan::print_format(R"({{
  [SERVER] JoinChannel_Error
  ID: {}
  ChannelID: {}
}})", base.ID, Protocol::JoinChannel_Error_Reason_String[(uint8_t)msg->Reason]);
    };
  }
  struct channel_create_awaiter {
    ecps_backend_t& backend;
    uint32_t request_id;

    channel_create_awaiter(ecps_backend_t& b, uint32_t id) : backend(b), request_id(id) {}

    bool await_ready() const noexcept {
      auto it = backend.pending_requests.find(request_id);
      return it != backend.pending_requests.end() && it->second.completed;
    }
    void await_suspend(std::coroutine_handle<> h) noexcept {
      auto it = backend.pending_requests.find(request_id);
      if (it != backend.pending_requests.end()) {
        it->second.continuation = h;
      }
    }
    uint16_t await_resume() {
      auto it = backend.pending_requests.find(request_id);
      if (it != backend.pending_requests.end()) {
        uint16_t channel_id = it->second.channel_id;
        backend.pending_requests.erase(it);
        return channel_id;
      }
      throw std::runtime_error("Channel creation request not found");
    }
  };
  fan::event::task_t tcp_read() {
    uint8_t data[] = {
    0x00, 0x00, 0x00, 0x00,
    0x00, 0x00
};








    tcp::ProtocolBasePacket_t* bp = (tcp::ProtocolBasePacket_t*)data;
    while (1) {
      auto msg = co_await tcp_client.read<tcp::ProtocolBasePacket_t>();
    /*  fan::print_format(R"({{
  ID: {}
  Command: {}
}})", msg->ID, msg->Command));*/
      if (msg->Command >= Protocol_S2C.size()) {
        fan::print("invalid command, ignoring...");
      }
      else {
        co_await(*Protocol_S2C.NA(msg->Command))(*this, msg.data);
      }
    }
  }
  fan::event::task_value_resume_t<uint32_t> tcp_write(int command, void* data = 0, uint32_t len = 0) {
    static int id = 0;
    tcp::ProtocolBasePacket_t bp;
    bp.ID = id++;
    bp.Command = command;
    co_await tcp_client.write_raw(&bp, sizeof(bp));
    if (data == nullptr) {
      uint8_t random = 0;
      co_await tcp_client.write_raw(&random, sizeof(random));
    }
    else {
      co_await tcp_client.write_raw(data, len);
    }
    co_return bp.ID;
  }
  template <typename T>
  fan::event::task_t udp_write(
    uint32_t ID,
    const T& Command,
    T CommandData,
    const void* data,
    uintptr_t size
  ) {
    uint8_t buffer[2048];
    auto BasePacket = (ProtocolUDP::BasePacket_t*)buffer;
    BasePacket->SessionID = session_id;
    BasePacket->ID = ID;
    BasePacket->IdentifySecret = identify_secret;
    BasePacket->Command = Command;
    auto CommandDataPacket = (T*)&BasePacket[1];
    *CommandDataPacket = CommandData;
    auto RestPacket = (uint8_t*)CommandDataPacket + T::dss;
    __builtin_memcpy(RestPacket, data, size);
    uint16_t TotalSize = sizeof(ProtocolUDP::BasePacket_t) + T::dss + size;
    co_await udp_client.send(fan::network::buffer_t((uint8_t*)buffer, (uint8_t*)buffer + TotalSize), ip, port);
    //IO_ssize_t r = NET_sendto(&g_pile->UDP.udp, buffer, TotalSize, &g_pile->UDP.Address);
  }

  struct view_t {
    uint64_t frame_index = 0;

    uint16_t m_Sequence;
    uint16_t m_Possible;

    uint16_t m_ModuloSize;

    std::vector<uint8_t> m_DataCheck = std::vector<uint8_t>(0x201);
    std::vector<uint8_t> m_data = std::vector<uint8_t>(0x400400);

    struct {
      uint64_t Frame_Total;
      uint64_t Frame_Drop;
      uint64_t Packet_Total;
      uint64_t Packet_HeadDrop;
      uint64_t Packet_BodyDrop;
    }m_stats;

    void SetNewSequence(uint16_t Sequence) {
      m_Sequence = Sequence;
      m_Possible = (uint16_t)-1;

      __builtin_memset(m_DataCheck.data(), 0, m_DataCheck.size());
    }

    bool IsSequencePast(uint16_t PacketSequence) {
      if (this->m_Sequence > PacketSequence) {
        if (this->m_Sequence - PacketSequence < 0x800) {
          return 1;
        }
        return 0;
      }
      else {
        if (PacketSequence - this->m_Sequence < 0x800) {
          return 0;
        }
        return 1;
      }
    }

    void SetDataCheck(uint16_t Index) {
      uint8_t* byte = &this->m_DataCheck[Index / 8];
      *byte |= 1 << Index % 8;
    }
    bool GetDataCheck(uint16_t Index) {
      uint8_t* byte = &this->m_DataCheck[Index / 8];
      return (*byte & (1 << Index % 8)) >> Index % 8;
    }

    uint16_t FindLastDataCheck(uint16_t start) {
      uint8_t* DataCheck = &this->m_DataCheck[start / 8];
      uint8_t* DataCheck_End = this->m_DataCheck.data() - 1;
      do {
        if (*DataCheck) {
          uint16_t r = (uintptr_t)DataCheck - (uintptr_t)this->m_DataCheck.data();
          for (r = r * 8 + 7;; r--) {
            if (GetDataCheck(r)) {
              return r;
            }
          }
        }
        DataCheck--;
      } while (DataCheck != DataCheck_End);
      return (uint16_t)-1;
    }

    void FixFramePacket() {
      this->m_stats.Frame_Total++;

      uint16_t LastDataCheck;
      if (this->m_Possible == (uint16_t)-1) {
        LastDataCheck = FindLastDataCheck(0x1000);
      }
      else {
        LastDataCheck = FindLastDataCheck(this->m_Possible);
      }

      if (LastDataCheck == (uint16_t)-1) {
        /* we cant fix anything in this packet */
        this->m_stats.Frame_Drop++;
        this->m_Possible = (uint16_t)-1;
        return;
      }

      this->m_stats.Packet_Total++;
      if (this->m_Possible == (uint16_t)-1) {
        this->m_stats.Packet_HeadDrop++;
        this->m_Possible = LastDataCheck + 1;
      }

      this->m_stats.Packet_Total += this->m_Possible;
      for (uint16_t i = 0; i < this->m_Possible; i++) {
        if (!GetDataCheck(i)) {
          this->m_stats.Packet_BodyDrop++;
          __builtin_memset(&this->m_data[i * 0x400], 0, 0x400);
        }
      }
    }

    void WriteFramePacket();

    struct FramePacketNodeData_t {
      std::vector<uint8_t> data;
    };
    std::deque<FramePacketNodeData_t> frame_packets;

  }view;
  struct share_t {
    uint64_t frame_index = 0;

    struct m_NetworkFlow_t {
      struct FrameListNodeData_t {
#if set_VerboseProtocol_HoldStreamTimes == 1
        ScreenShare_StreamHeader_Head_t::_VerboseTime_t _VerboseTime;
#endif
        std::vector<uint8_t> vec;
        uintptr_t SentOffset;
      };
      uint64_t WantedInterval = 5000000;
      uint64_t Bucket; /* in bit */
      uint64_t BucketSize; /* in bit */
      uint64_t TimerLastCallAt;
      uint64_t TimerCallCount;
    }m_NetworkFlow;
  }share;
  fan::event::task_t write_stream(
    uint16_t Current,
    uint16_t Possible,
    uint8_t Flag,
    void* Data,
    uintptr_t DataSize
  ) {
    uint8_t buffer[sizeof(ScreenShare_StreamHeader_Head_t) + 0x400];

    auto Body = (ScreenShare_StreamHeader_Body_t*)buffer;
    Body->SetSequence(share.frame_index);
    Body->SetCurrent(Current);

    void* DataWillBeAt;
    if (Current == 0) {
      auto Head = (ScreenShare_StreamHeader_Head_t*)buffer;
      Head->SetPossible(Possible);
      Head->SetFlag(Flag);
      DataWillBeAt = (void*)&Head[1];
    }
    else {
      DataWillBeAt = (void*)&Body[1];
    }

    uintptr_t BufferSize = (uintptr_t)DataWillBeAt - (uintptr_t)buffer + DataSize;
    //if(share.m_NetworkFlow.Bucket < BufferSize * 8){
    //  return 1;
    //}
    //share.m_NetworkFlow.Bucket -= BufferSize * 8;

    __builtin_memcpy(DataWillBeAt, Data, DataSize);

    ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData_t rest;
    rest.ChannelID = channel_info.front().channel_id;
    rest.ChannelSessionID = channel_info.front().session_id;
    co_await udp_write(0, ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData, rest, buffer, BufferSize);
  }

  fan::event::task_t connect(const std::string& ip, uint16_t port) {
    this->ip = ip;
    this->port = port;
    try {
      co_await tcp_client.connect(ip, port);
    }
    catch (...) { co_return; }
    udp_keep_alive.set_server(
      fan::network::socket_address_t{ ip, port },
      [this](fan::network::udp_t& udp) -> fan::event::task_t {
        co_await udp_write(0, ProtocolUDP::C2S_t::KeepAlive, {}, 0, 0);
      }
    );
    // udp read
    task_udp_listen = udp_client.listen(
      fan::network::listen_address_t{ this->ip, this->port },
      [this, ip, port](const fan::network::udp_t& udp, const fan::network::udp_datagram_t& datagram) -> fan::event::task_t {
        
        auto size = datagram.data.size();

        udp::BasePacket_t bp = *(udp::BasePacket_t*)datagram.data.data();
        if (bp.SessionID != session_id) {
          co_return;
        }
        uintptr_t RelativeSize = size - sizeof(bp);
        if (bp.Command == ProtocolUDP::S2C_t::KeepAlive) {
          if (sizeof(bp) != datagram.data.size()) {
            fan::print("size is not same as sizeof expected arrival");
          }
          fan::print("udp keep alive came");
          udp_keep_alive.reset();
        }
        else if (bp.Command == ProtocolUDP::S2C_t::Channel_ScreenShare_View_StreamData) {
          auto CommandData = (ProtocolUDP::S2C_t::Channel_ScreenShare_View_StreamData_t*)&(((udp::BasePacket_t*)datagram.data.data())[1]);
          if (RelativeSize < sizeof(*CommandData)) {
            co_return;
          }
          RelativeSize -= sizeof(*CommandData);

          if (RelativeSize < sizeof(ScreenShare_StreamHeader_Body_t)) {
            co_return;
          }
          auto StreamData = (uint8_t*)&CommandData[1];

          auto Body = (ScreenShare_StreamHeader_Body_t*)StreamData;
          uint16_t Sequence = Body->GetSequence();
          if (Sequence != view.m_Sequence) {
            if (view.IsSequencePast(Sequence)) {
              /* this packet came from past */
              co_return;
            }
            view.FixFramePacket();
            if (view.m_Possible != (uint16_t)-1) {
              view.WriteFramePacket();
            }
            view.SetNewSequence(Sequence);
          }
          uint8_t* PacketData;
          uint16_t Current = Body->GetCurrent();
          if (Current == 0) {
            if ((uintptr_t)size < sizeof(ScreenShare_StreamHeader_Head_t)) {
              co_return;
            }
            if (view.m_Possible != (uint16_t)-1) {
              /* we already have head somehow */
              co_return;
            }
            auto Head = (ScreenShare_StreamHeader_Head_t*)StreamData;
            view.m_Possible = Head->GetPossible();
            /*
            View->m_Flag = Head->GetFlag();
            */
            PacketData = &StreamData[sizeof(ScreenShare_StreamHeader_Head_t)];
          }
          else {
            PacketData = &StreamData[sizeof(ScreenShare_StreamHeader_Body_t)];
          }

          uint16_t PacketSize = RelativeSize - ((uintptr_t)PacketData - (uintptr_t)StreamData);

          if (PacketSize != 0x400) {
            view.m_ModuloSize = PacketSize;
          }
          __builtin_memcpy(&view.m_data[Current * 0x400], PacketData, PacketSize);
          view.SetDataCheck(Current);
          if (
            (view.m_Possible != (uint16_t)-1 && Current == view.m_Possible) ||
            PacketSize != 0x400
            ) {
            view.FixFramePacket();
            if (view.m_Possible != (uint16_t)-1) {
              view.WriteFramePacket();
            }
            view.SetNewSequence((view.m_Sequence + 1) % 0x1000);
          }
          udp_keep_alive.reset();
        }
        else {
          fan::print("unprocessed data came");
        }
        co_return;
      }
    );
    tcp_keep_alive.reset();
    udp_keep_alive.reset();
    task_tcp_read = tcp_read();
  }
  fan::event::task_t login() {
    co_await tcp_write(Protocol_C2S_t::Request_Login);
  }
  fan::event::task_value_resume_t<Protocol_ChannelID_t> channel_create() {
    uint32_t request_id = co_await tcp_write(Protocol_C2S_t::CreateChannel);
    pending_requests[request_id] = pending_request_t{
      .request_id = request_id,
      .continuation = {},
      .channel_id = 0,
      .completed = false
    };
    co_return co_await channel_create_awaiter(*this, request_id);
  }
  // add check if channel join is ok
  fan::event::task_t channel_join(Protocol_ChannelID_t channel_id) {
    co_await tcp_write(Protocol_C2S_t::JoinChannel, &channel_id, sizeof(channel_id));
    channel_info_t ci;
    ci.channel_id = channel_id;
    channel_info.emplace_back(ci);
  }

  fan::event::task_t task_tcp_read;

  std::string ip;
  uint16_t port;
  fan::network::tcp_t tcp_client;
  fan::network::udp_t udp_client;
  fan::event::task_t task_udp_listen;

  fan::network::tcp_keep_alive_t tcp_keep_alive{
    tcp_client,
    [this] (fan::network::tcp_t& tcp) -> fan::event::task_t{
      co_await tcp_write(Protocol_C2S_t::KeepAlive);
    }
  };
  fan::network::udp_keep_alive_t udp_keep_alive{ udp_client };

  struct pending_request_t {
    uint32_t request_id;
    std::coroutine_handle<> continuation;
    Protocol_ChannelID_t channel_id = 0;
    bool completed = false;
  };
  std::unordered_map<uint32_t, pending_request_t> pending_requests;

  uint64_t identify_secret = 0;

  Protocol_SessionID_t session_id;

  struct channel_info_t {
    Protocol_ChannelID_t channel_id = 0;
    Protocol_ChannelSessionID_t session_id = 0;
  };
  std::vector<channel_info_t> channel_info;
}ecps_backend;