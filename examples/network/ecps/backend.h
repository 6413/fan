struct ecps_backend_t {

#define __ecps_client
#include "prot.h"

  ecps_backend_t();
  ~ecps_backend_t() {
    should_stop_tcp_read.store(true);
    udp_keep_alive.stop();
    tcp_keep_alive.stop();

    auto start_time = std::chrono::steady_clock::now();
    while ((task_tcp_read.handle || task_udp_listen.handle) &&
      std::chrono::steady_clock::now() - start_time < std::chrono::milliseconds(100)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (task_tcp_read.handle) {
      task_tcp_read = {};
    }
    if (task_udp_listen.handle) {
      task_udp_listen = {};
    }
    MD_Mice_Close(&mice);
    MD_Keyboard_close(&keyboard);
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
  std::atomic<bool> should_stop_tcp_read{ false };

  fan::event::task_t tcp_read() {
    while (!should_stop_tcp_read.load()) {
      fan::network::typed_message_t<tcp::ProtocolBasePacket_t> msg;
      try {
        if (should_stop_tcp_read.load()) {
          co_return;
        }

        msg = co_await tcp_client.read<tcp::ProtocolBasePacket_t>();

        if (should_stop_tcp_read.load()) {
          co_return;
        }

        if (msg.status != 0) {
          continue;
        }
      }
      catch (const std::exception& e) {
        if (should_stop_tcp_read.load()) {
          co_return;
        }
        fan::print("TCP read error: " + std::string(e.what()));
        continue;
      }
      catch (...) {
        co_return;
      }

      if (msg.status != 0 || msg->Command >= Protocol_S2C.size()) {
        fan::print("invalid command, ignoring...");
        continue;
      }

      try {
        if (should_stop_tcp_read.load()) {
          co_return;
        }

        co_await(*Protocol_S2C.NA(msg->Command))(*this, msg.data);
      }
      catch (const std::exception& e) {
        fan::print("Message processing error: " + std::string(e.what()));
        continue;
      }
      catch (...) {
        continue;
      }
    }
  }
  fan::event::task_value_resume_t<uint32_t> tcp_write(int command, void* data, uint32_t len) {
    static int id = 0;
    tcp::ProtocolBasePacket_t bp;
    bp.ID = id++;
    bp.Command = command;
    co_await tcp_client.write_raw(&bp, sizeof(bp));
    co_await tcp_client.write_raw(data, len);
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

  struct recovery_request_t {
    uint16_t sequence;
    uint16_t missing_count;
  };

  struct view_t {
    uint64_t frame_index = 0;
    uint16_t m_Sequence;
    uint16_t m_Possible;
    uint16_t m_ModuloSize;
    uint64_t frame_start_time = 0;

    std::vector<uint8_t> m_DataCheck = std::vector<uint8_t>(0x201);
    std::vector<uint8_t> m_data = std::vector<uint8_t>(0x400400);
    std::vector<uint16_t> m_MissingPackets;
    uint32_t m_PacketCounter = 0;
    bool m_AckSent = false;

    static constexpr uint32_t recovery_ack_interval = 1;
    std::function<void(std::vector<uint8_t>)> recovery_callback;

    struct {
      uint64_t Frame_Total;
      uint64_t Frame_Drop;
      uint64_t Packet_Total;
      uint64_t Packet_HeadDrop;
      uint64_t Packet_BodyDrop;
      uint64_t Frame_Corrected;
    } m_stats;

    uint64_t get_adaptive_timeout() {
      f32_t ping_ms = 0;
      auto* rt = render_thread_ptr.load(std::memory_order_acquire);
      if (rt) {
        ping_ms = OFFSETLESS(this, ecps_backend_t, view)->ping_ms;
      }

      // Base timeout + (RTT * safety multiplier) + recovery time
      uint64_t base_timeout = 50000000ULL; // 50ms base
      uint64_t rtt_timeout = (uint64_t)(ping_ms * 2.0f * 1000000.0f); // 2x RTT in nanoseconds
      uint64_t recovery_time = 30000000ULL; // 30ms for recovery processing

      uint64_t total_timeout = base_timeout + rtt_timeout + recovery_time;

      uint64_t min_timeout = 80000000ULL; 
      uint64_t max_timeout = 300000000ULL;

      return std::clamp(total_timeout, min_timeout, max_timeout);
    }

    void CheckAndSendRecoveryRequest() {
      if (frame_start_time > 0) {
        uint64_t current_time = fan::event::now();
        uint64_t frame_age = current_time - frame_start_time;
        uint64_t adaptive_timeout = get_adaptive_timeout();
        
        if (frame_age > adaptive_timeout) {
            SetNewSequence((m_Sequence + 1) % 0x1000);
            return;
        }
    }

    m_PacketCounter++;
    if (m_PacketCounter % 1 != 0) {
        return;
    }
    
    UpdateMissingPackets();
    if (m_MissingPackets.empty()) {
        m_AckSent = false;
        return;
    }
    
    static int retry_count = 0;
    if (!m_AckSent || retry_count < 3) {
        SendRecoveryRequest();
        retry_count++;
    }
}

    void FixFrameOnComplete();

    void ApplyImprovedErrorConcealment() {
      for (uint16_t missing : m_MissingPackets) {
        if (missing > 0 && missing < m_Possible - 1) {
          uint8_t* dst = &m_data[missing * 0x400];
          const uint8_t* prev = &m_data[(missing - 1) * 0x400];
          const uint8_t* next = &m_data[(missing + 1) * 0x400];

          for (int i = 0; i < 0x400; i += 4) {
            dst[i] = static_cast<uint8_t>((prev[i] + next[i]) / 2);
            dst[i + 1] = static_cast<uint8_t>((prev[i + 1] + next[i + 1]) / 2);
            dst[i + 2] = static_cast<uint8_t>((prev[i + 2] + next[i + 2]) / 2);
            dst[i + 3] = static_cast<uint8_t>((prev[i + 3] + next[i + 3]) / 2);
          }
        }
        else if (missing > 0) {
          uint8_t* dst = &m_data[missing * 0x400];
          const uint8_t* src = &m_data[(missing - 1) * 0x400];
          for (int i = 0; i < 0x400; i += 4) {
            dst[i] = static_cast<uint8_t>(src[i] * 0.8f);
            dst[i + 1] = static_cast<uint8_t>(src[i + 1] * 0.8f);
            dst[i + 2] = static_cast<uint8_t>(src[i + 2] * 0.8f);
            dst[i + 3] = static_cast<uint8_t>(src[i + 3] * 0.8f);
          }
        }
        else {
          memset(&m_data[missing * 0x400], 0, 0x400);
        }
      }
    }

    void RequestKeyframe();

    void UpdateMissingPackets() {
      m_MissingPackets.clear();
      if (m_Possible == (uint16_t)-1) {
        return;
      }

      for (uint16_t i = 0; i < m_Possible; i++) {
        if (!GetDataCheck(i)) {
          m_MissingPackets.push_back(i);
        }
      }
    }

    void SendRecoveryRequest() {
      if (m_MissingPackets.empty()) {
        return;
      }

      struct recovery_request_t {
        uint16_t sequence;
        uint16_t missing_count;
      };

      size_t request_size = sizeof(recovery_request_t) + m_MissingPackets.size() * sizeof(uint16_t);
      std::vector<uint8_t> request_data(request_size);

      recovery_request_t* req = reinterpret_cast<recovery_request_t*>(request_data.data());
      req->sequence = m_Sequence;
      req->missing_count = static_cast<uint16_t>(m_MissingPackets.size());

      uint16_t* missing_indices = reinterpret_cast<uint16_t*>(req + 1);
      for (size_t i = 0; i < m_MissingPackets.size(); ++i) {
        missing_indices[i] = m_MissingPackets[i];
      }

      if (recovery_callback) {
        recovery_callback(std::move(request_data));
      }

      m_AckSent = true;
    }

    void SetNewSequence(uint16_t Sequence) {
    m_Sequence = Sequence;
    m_Possible = (uint16_t)-1;
    m_AckSent = false;
    m_MissingPackets.clear();
    __builtin_memset(m_DataCheck.data(), 0, m_DataCheck.size());
    frame_start_time = fan::event::now(); // ADD THIS LINE
}

    bool IsSequencePast(uint16_t PacketSequence) {
      int16_t diff = (int16_t)(PacketSequence - this->m_Sequence);
      return diff < 0;
    }

    void SetDataCheck(uint16_t Index) {
      uint8_t* byte = &this->m_DataCheck[Index / 8];
      *byte |= 1 << Index % 8;
    }

    bool GetDataCheck(uint16_t Index) {
      uint8_t* byte = &this->m_DataCheck[Index / 8];
      return (*byte & (1 << Index % 8)) >> Index % 8;
    }

    void WriteFramePacket();
  }view;

  struct share_t {
    uint64_t frame_index = 0;

    std::unordered_map<uint16_t, std::vector<std::vector<uint8_t>>> m_SentPackets;
    std::function<void(const std::vector<uint8_t>&)> resend_packet_callback;

    void StorePacketForRecovery(uint16_t sequence, uint16_t current,
      const void* data, size_t size) {
      if (m_SentPackets[sequence].size() <= current) {
        m_SentPackets[sequence].resize(current + 1);
      }

      m_SentPackets[sequence][current].assign(
        static_cast<const uint8_t*>(data),
        static_cast<const uint8_t*>(data) + size
      );

      if (m_SentPackets.size() > 50) {
        auto it = m_SentPackets.begin();
        for (int i = 0; i < 10 && it != m_SentPackets.end(); ++i) {
          it = m_SentPackets.erase(it);
        }
      }
    }

    void ProcessRecoveryRequest(uint16_t sequence, const uint16_t* missing_indices,
      uint16_t missing_count) {
      auto it = m_SentPackets.find(sequence);
      if (it == m_SentPackets.end()) {
        return;
      }

      auto& packets = it->second;

      for (uint16_t i = 0; i < missing_count; ++i) {
        uint16_t packet_index = missing_indices[i];
        if (packet_index < packets.size() && !packets[packet_index].empty()) {

          if (resend_packet_callback) {
            resend_packet_callback(packets[packet_index]);
          }
        }
      }
    }


    struct m_NetworkFlow_t {
      struct FrameListNodeData_t {
#if set_VerboseProtocol_HoldStreamTimes == 1
        ScreenShare_StreamHeader_Head_t::_VerboseTime_t _VerboseTime;
#endif
        std::vector<uint8_t> vec;
        uintptr_t SentOffset;
      };
#include <fan/fan_bll_preset.h>
#define BLL_set_prefix FrameList
#define BLL_set_Language 1
#define BLL_set_Usage 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeDataType FrameListNodeData_t
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>
      FrameList_t FrameList;
      uint64_t WantedInterval = 5000000;
      uint64_t Bucket; /* in bit */
      uint64_t BucketSize; /* in bit */
      uint64_t TimerLastCallAt;
      uint64_t TimerCallCount;
    }m_NetworkFlow;
    std::timed_mutex frame_list_mutex;
    void CalculateNetworkFlowBucket();
  }share;

  fan::event::task_value_resume_t<bool> write_stream(
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
    share.StorePacketForRecovery(share.frame_index, Current, buffer, BufferSize);
    if (share.m_NetworkFlow.Bucket < BufferSize * 8) {
      int64_t deficit_threshold = static_cast<int64_t>(share.m_NetworkFlow.BucketSize) * 0.1;
      if (static_cast<int64_t>(share.m_NetworkFlow.Bucket) < -deficit_threshold) {
        co_return 1;
      }
    }
    share.m_NetworkFlow.Bucket -= BufferSize * 8;

    __builtin_memcpy(DataWillBeAt, Data, DataSize);

    ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData_t rest;
    Protocol_ChannelID_t streaming_channel_id;
    Protocol_ChannelSessionID_t streaming_session_id;
    bool found_streaming = false;

    for (const auto& channel : channel_info) {
      if (channel.is_streaming) {
        streaming_channel_id = channel.channel_id;
        streaming_session_id = channel.session_id;
        found_streaming = true;
        break;
      }
    }

    if (!found_streaming) {
      co_return 1;
    }

    rest.ChannelID = streaming_channel_id;
    rest.ChannelSessionID = streaming_session_id;

    co_await udp_write(0, ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData, rest, buffer, BufferSize);
#if ecps_debug_prints >= 4
    static std::atomic<uint64_t> total_udp_packets_sent{ 0 };
    printf("UDP_SEND: seq=%u, pkt=%u/%u, size=%zu, total_sent=%llu\n",
      share.frame_index, Current, Possible, BufferSize,
      total_udp_packets_sent.fetch_add(1));
#endif
    co_return 0;
  }

  fan::event::task_value_resume_t<bool> connect(const std::string& ip, uint16_t port) {
    this->ip = ip;
    this->port = port;

    should_stop_tcp_read.store(true);
    udp_keep_alive.stop();
    tcp_keep_alive.stop();

    if (task_tcp_read.handle) {
      for (int i = 0; i < 50 && task_tcp_read.handle; ++i) {
        co_await fan::co_sleep(10);
      }

      if (task_tcp_read.handle) {
        task_tcp_read = {};
      }
    }

    if (task_udp_listen.handle) {
      task_udp_listen = {};
    }

    pending_requests.clear();
    channel_info.clear();
    available_channels.clear();
    channel_sessions.clear();
    channel_list_received = false;
    session_id = {};
    identify_secret = 0;

    try {
      co_await tcp_client.connect(ip, port);
    }
    catch (...) { co_return false; }

    should_stop_tcp_read.store(false);

    udp_keep_alive.set_server(
      fan::network::socket_address_t{ ip, port },
      [this](fan::network::udp_t& udp) -> fan::event::task_t {
        keepalive_sent_time = fan::event::now();
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
        if (bp.Command == ProtocolUDP::S2C_t::KeepAlive) {
          if (sizeof(bp) != datagram.data.size()) {
            fan::print("size is not same as sizeof expected arrival");
          }
#if ecps_debug_prints >= 2
          fan::print("udp keep alive came");
#endif
          if (keepalive_sent_time > 0) {
            uint64_t now = fan::event::now();
            ping_ms = (now - keepalive_sent_time) / 1000000.0;
            keepalive_sent_time = 0;
          }
          udp_keep_alive.reset();
        }
        else if (bp.Command == ProtocolUDP::S2C_t::Channel_ScreenShare_View_StreamData) {
          uintptr_t RelativeSize = size - sizeof(bp);
          auto CommandData = (ProtocolUDP::S2C_t::Channel_ScreenShare_View_StreamData_t*)&(((udp::BasePacket_t*)datagram.data.data())[1]);
          set_channel_viewing(CommandData->ChannelID, true);
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
              co_return;
            }
            if (view.m_Possible != (uint16_t)-1) {
              view.FixFrameOnComplete();
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
              co_return;
            }
            auto Head = (ScreenShare_StreamHeader_Head_t*)StreamData;
            view.m_Possible = Head->GetPossible();
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
          view.CheckAndSendRecoveryRequest();
          if (
            (view.m_Possible != (uint16_t)-1 && Current == view.m_Possible) ||
            PacketSize != 0x400
            ) {
            view.FixFrameOnComplete();
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
    co_return true;
  }
  fan::event::task_t login() {
    // while (1) {
    try {
      Protocol_C2S_t::Request_Login_t rest;
      rest.Type = Protocol::LoginType_t::Anonymous;
      co_await tcp_write(Protocol_C2S_t::Request_Login, &rest, sizeof(rest));
      co_return;
    }
    catch (fan::exception_t e) {
      login_fail_cb(e);
    }
    //co_await fan::co_sleep(1000);
    //co_await connect(ip, port);
 // }
  }
  fan::event::task_value_resume_t<Protocol_ChannelID_t> channel_create() {
    Protocol_C2S_t::CreateChannel_t rest;
    rest.Type = Protocol::ChannelType_ScreenShare_e;
    uint32_t request_id = co_await tcp_write(Protocol_C2S_t::CreateChannel, &rest, sizeof(rest));
    pending_requests[request_id] = pending_request_t{
      .request_id = request_id,
      .continuation = {},
      .channel_id = 0,
      .completed = false
    };
    co_return co_await channel_create_awaiter(*this, request_id);
  }
  // add check if channel join is ok
  fan::event::task_t channel_join(Protocol_ChannelID_t channel_id, bool is_own_channel = false) {
    co_await tcp_write(Protocol_C2S_t::JoinChannel, &channel_id, sizeof(channel_id));
    channel_info_t ci;
    ci.channel_id = channel_id;
    ci.joined_at.start();
    channel_info.emplace_back(ci);
    udp_keep_alive.reset();
    {
      ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
      rest.ChannelID = channel_info.back().channel_id;
      rest.Flag = ecps_backend_t::ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
      co_await tcp_write(
        ecps_backend_t::Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
        &rest,
        sizeof(rest)
      );
    }
    did_just_join = !is_own_channel;
  }

  bool is_channel_id_valid(Protocol_ChannelID_t channel_id) const {
    return channel_id.i != (uint16_t)-1 && channel_id.i != 0;
  }
  bool is_channel_available(Protocol_ChannelID_t channel_id) const {
    if (!is_channel_id_valid(channel_id)) {
      return false;
    }
    return std::any_of(available_channels.begin(), available_channels.end(),
      [channel_id](const channel_list_info_t& channel) {
        return channel.channel_id.i == channel_id.i;
      });
  }
  bool is_channel_joined(Protocol_ChannelID_t channel_id) const {
    if (!is_channel_id_valid(channel_id)) {
      return false;
    }

    return std::any_of(channel_info.begin(), channel_info.end(),
      [channel_id](const channel_info_t& joined_channel) {
        return joined_channel.channel_id.i == channel_id.i;
      });
  }

  bool cleanup_disconnected_channels() {
    size_t original_size = channel_info.size();
    channel_info.erase(
      std::remove_if(channel_info.begin(), channel_info.end(),
        [this](const auto& joined) {
          if (joined.joined_at.elapsed() < 5e+9) {
            return false;
          }

          return std::none_of(available_channels.begin(), available_channels.end(),
            [&](const auto& available) {
              return available.channel_id.i == joined.channel_id.i;
            });
        }),
      channel_info.end()
    );
    return channel_info.size() < original_size;
  }

  bool is_current_user_host_of_channel(Protocol_ChannelID_t channel) const {
    auto session_it = channel_sessions.find(channel);
    if (session_it == channel_sessions.end()) {
      return false;
    }

    const std::string current_username = get_current_username();
    for (const auto& session : session_it->second) {
      if (session.username == current_username && session.is_host) {
        return true;
      }
    }
    return false;
  }

  fan::event::task_t task_tcp_read;

  std::string ip;
  uint16_t port;
  fan::network::tcp_t tcp_client;
  fan::network::udp_t udp_client;
  fan::event::task_t task_udp_listen;
  uint64_t keepalive_sent_time = 0;
  f32_t ping_ms = 0;

  std::function<void(fan::exception_t)> login_fail_cb;

  fan::network::tcp_keep_alive_t tcp_keep_alive{
    tcp_client,
    [this](fan::network::tcp_t& tcp) -> fan::event::task_t {
      co_await tcp_write(Protocol_C2S_t::KeepAlive, 0, 0);
    }
  };

  fan::event::task_t request_channel_list() {
    uint8_t a;
    co_await tcp_write(Protocol_C2S_t::RequestChannelList, &a, 1);
  }

  fan::event::task_t request_channel_session_list(Protocol_ChannelID_t channel_id) {
    Protocol_C2S_t::RequestChannelSessionList_t request;
    request.ChannelID = channel_id;
    co_await tcp_write(Protocol_C2S_t::RequestChannelSessionList, &request, sizeof(request));
  }


  fan::event::task_t handle_channel_list(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
    auto data = reinterpret_cast<const uint8_t*>(&base) + sizeof(base);
    auto channel_list = reinterpret_cast<const Protocol_S2C_t::ChannelList_t*>(data);

    backend.available_channels.clear();
    backend.available_channels.reserve(channel_list->ChannelCount);

    auto channel_info_ptr = reinterpret_cast<const Protocol_S2C_t::ChannelInfo_t*>(data + sizeof(Protocol_S2C_t::ChannelList_t));

    for (uint16_t i = 0; i < channel_list->ChannelCount; ++i) {
      ecps_backend_t::channel_list_info_t info;
      info.channel_id = channel_info_ptr[i].ChannelID;
      info.type = channel_info_ptr[i].Type;
      info.user_count = channel_info_ptr[i].UserCount;
      info.name = std::string(channel_info_ptr[i].Name, strnlen(channel_info_ptr[i].Name, 63));
      info.is_password_protected = channel_info_ptr[i].IsPasswordProtected != 0;
      info.host_session_id = channel_info_ptr[i].HostSessionID;
      backend.available_channels.push_back(info);
    }

    backend.channel_list_received = true;
#if ecps_debug_prints >= 3
    fan::print("Received channel list with {} channels", channel_list->ChannelCount);
#endif

    co_return;
  }

  fan::event::task_t handle_channel_session_list(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) {
    auto data = reinterpret_cast<const uint8_t*>(&base) + sizeof(base);
    auto session_list = reinterpret_cast<const Protocol_S2C_t::ChannelSessionList_t*>(data);

    Protocol_ChannelID_t channel_id = session_list->ChannelID;

    backend.channel_sessions[channel_id].clear();
    backend.channel_sessions[channel_id].reserve(session_list->SessionCount);

    auto session_info_ptr = reinterpret_cast<const Protocol_S2C_t::SessionInfo_t*>(data + sizeof(Protocol_S2C_t::ChannelSessionList_t));

    for (uint16_t i = 0; i < session_list->SessionCount; ++i) {
      ecps_backend_t::session_info_t info;
      info.session_id = session_info_ptr[i].SessionID;
      info.channel_session_id = session_info_ptr[i].ChannelSessionID;
      info.account_id = session_info_ptr[i].AccountID;
      info.username = std::string((const char*)session_info_ptr[i].Username, strnlen((const char*)session_info_ptr[i].Username, 31));
      info.is_host = session_info_ptr[i].IsHost != 0;
      info.joined_at = session_info_ptr[i].JoinedAt;
      backend.channel_sessions[channel_id].push_back(info);
    }
#if ecps_debug_prints >= 3
    fan::print("Received session list for channel {} with {} sessions", (int)channel_id, session_list->SessionCount);
#endif

    co_return;
  }


  bool is_connected() {
    return false;
  }

  std::string get_current_username() const {
    for (const auto& [channel_id, sessions] : channel_sessions) {
      for (const auto& session : sessions) {
        if (session.session_id.i == this->session_id.i) {
          return session.username;
        }
      }
    }

    if (session_id.i != 0) {
      return "Anonymous_" + std::to_string(session_id.i);
    }

    return "Not Connected";
  }

  void set_channel_streaming(Protocol_ChannelID_t channel_id, bool streaming) {
    if (channel_id.i == (uint16_t)-1) {
      return;
    }
    for (auto& channel : channel_info) {
      if (channel.channel_id.i == channel_id.i) {
        channel.is_streaming = streaming;
        channel.stream_start_timer.start();
        return;
      }
    }
  }
  void set_channel_viewing(Protocol_ChannelID_t channel_id, bool viewing) {
    if (channel_id.i == (uint16_t)-1) {
      return;
    }
    for (auto& channel : channel_info) {
      if (channel.channel_id.i == channel_id.i) {
        channel.is_viewing = viewing;
        return;
      }
    }
  }
  bool is_channel_streaming(Protocol_ChannelID_t channel_id) const {
    if (channel_id.i == (uint16_t)-1) {
      return false;
    }
    for (const auto& channel : channel_info) {
      if (channel.channel_id.i == channel_id.i) {
        return channel.is_streaming;
      }
    }
    return false;
  }
  uint64_t get_channel_stream_time(Protocol_ChannelID_t channel_id) {
    if (channel_id.i == (uint16_t)-1) {
      return 0;
    }
    for (const auto& channel : channel_info) {
      if (channel.channel_id.i == channel_id.i) {
        return channel.stream_start_timer.elapsed();
      }
    }
    return 0;
  }
  bool is_channel_viewing(Protocol_ChannelID_t channel_id) const {
    for (const auto& channel : channel_info) {
      if (channel.channel_id.i == channel_id.i) {
        return channel.is_viewing;
      }
    }
    return false;
  }
  bool is_streaming_to_any_channel() const {
    return std::any_of(channel_info.begin(), channel_info.end(),
      [](const channel_info_t& ch) { return ch.is_streaming; });
  }
  bool is_viewing_any_channel() const {
    return std::any_of(channel_info.begin(), channel_info.end(),
      [](const channel_info_t& ch) { return ch.is_viewing; });
  }
  fan::event::task_t request_idr_reset() {
    try {
      for (const auto& channel : channel_info) {
        if (channel.is_viewing) {
          Protocol_C2S_t::Channel_ScreenShare_ViewToShare_t rest;
          rest.ChannelID = channel.channel_id;
          rest.Flag = ProtocolChannel::ScreenShare::ChannelFlag::ResetIDR;
          co_await tcp_write(
            Protocol_C2S_t::Channel_ScreenShare_ViewToShare,
            &rest,
            sizeof(rest)
          );
        }
      }
    }
    catch (const std::exception& e) {
      fan::print("Failed to send IDR reset: " + std::string(e.what()));
    }
    catch (fan::exception_t e) {
      fan::print("Failed to send IDR reset: " + std::string(e.reason));
    }
    catch (...) {
      fan::print("Failed to send IDR reset: unknown error");
    }
    co_return;
  }

  void update_host_mouse_coordinate(Protocol_ChannelID_t channel_id, const fan::vec2ui& pos);

  bool channel_list_received = false;
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

    bool is_streaming = false;
    fan::time::timer stream_start_timer;
    bool is_viewing = false;
    fan::time::timer joined_at;
    int flag = 0;
  };
  std::vector<channel_info_t> channel_info;


  std::vector<channel_list_info_t> available_channels;
  std::unordered_map<Protocol_ChannelID_t::Type, std::vector<session_info_t>> channel_sessions;
  bool did_just_join = false; // need manual reset from gui

  MD_Mice_t mice;
  MD_Keyboard_t keyboard;

}ecps_backend;