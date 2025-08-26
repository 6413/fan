#include <functional>
#include <set>
#include <algorithm>

import fan.event;
import fan.print;

#include "Base.h"

template <typename T>
void
UDP_send(
  Protocol_SessionID_t SessionID,
  uint32_t ID,
  const T& Command,
  T CommandData,
  const void* data,
  IO_size_t size
) {
  auto Session = &g_pile->SessionList[SessionID];
  uint8_t buffer[2048];
  auto BasePacket = (ProtocolUDP::BasePacket_t*)buffer;
  BasePacket->SessionID = SessionID;
  BasePacket->ID = ID;
  BasePacket->IdentifySecret = 0;
  BasePacket->Command = Command;
  auto CommandDataPacket = (T*)&BasePacket[1];
  *CommandDataPacket = CommandData;
  auto RestPacket = (uint8_t*)CommandDataPacket + T::dss;
  __builtin_memcpy(RestPacket, data, size);
  uint16_t TotalSize = sizeof(ProtocolUDP::BasePacket_t) + T::dss + size;
  IO_ssize_t r = NET_sendto(&g_pile->udp, buffer, TotalSize, &Session->UDP.Address);
  if (r != TotalSize) {
    if (r == 0) {
      fan::print_throttled("UDP send buffer full, backing off\r\n");
    }
    else {
      fan::print_throttled("[INTERNAL ERROR] NET_sendto failed. wanted " + std::to_string(TotalSize) + " got " + std::to_string(r));
    }
  }
}

#include "channel.h"

#include "Session.h"
#include "TCPPeer.h"

#include "ScreenShare.h"

uint32_t TCPMain_state_cb(NET_TCP_peer_t* peer, TCPMain_SockData_t* SockData, TCPMain_PeerData_t* PeerData, uint32_t flag) {
  if (flag & NET_TCP_state_succ_e) {
    TCPPeer::Open(peer);
  }
  else do {
    if (!(flag & NET_TCP_state_init_e)) {
      break;
    }
    TCPPeer::Close(peer);
  } while (0);

  return 0;
}

uint32_t TCPMain_read_cb(
  NET_TCP_peer_t* peer,
  uint8_t* SockData,
  TCPMain_PeerData_t* PeerData,
  NET_TCP_QueuerReference_t QueuerReference,
  uint32_t* type,
  NET_TCP_Queue_t* Queue
) {
  uint8_t* ReadData;
  uintptr_t ReadSize;
  uint8_t _EventReadBuffer[4096];
  switch (*type) {
  case NET_TCP_QueueType_DynamicPointer: {
    ReadData = (uint8_t*)Queue->DynamicPointer.ptr;
    ReadSize = Queue->DynamicPointer.size;
    break;
  }
  case NET_TCP_QueueType_PeerEvent: {
    IO_fd_t peer_fd;
    EV_event_get_fd(&peer->event, &peer_fd);
    IO_ssize_t len = IO_read(&peer_fd, _EventReadBuffer, sizeof(_EventReadBuffer));
    if (len < 0) {
      NET_TCP_CloseHard(peer);
      return NET_TCP_EXT_PeerIsClosed_e;
    }
    ReadData = _EventReadBuffer;
    ReadSize = len;
    break;
  }
  case NET_TCP_QueueType_CloseHard: {
    return 0;
  }
  default: {
    WriteError("*type %lx\r\n", *type);
    __abort();
  }
  }
  NET_TCP_UpdatePeerTimer(peer->parent->listener, peer->parent, peer);
  for (uintptr_t iSize = 0; iSize < ReadSize;) {
    switch (PeerData->state) {
    case PeerState_Idle_e: {
      PeerData->iBuffer = 0;
      PeerData->Buffer = A_resize(0, sizeof(ProtocolBasePacket_t));
      PeerData->state = PeerState_Waitting_BasePacket_e;
      break;
    }
    case PeerState_Waitting_BasePacket_e: {
      uintptr_t size = ReadSize - iSize;
      uintptr_t needed_size = sizeof(ProtocolBasePacket_t) - PeerData->iBuffer;
      if (size > needed_size) {
        size = needed_size;
      }
      __builtin_memcpy(&PeerData->Buffer[PeerData->iBuffer], &ReadData[iSize], size);
      iSize += size;
      PeerData->iBuffer += size;
      if (PeerData->iBuffer == sizeof(ProtocolBasePacket_t)) {

        auto BasePacket = (ProtocolBasePacket_t*)PeerData->Buffer;
        //WriteInformation("Received packet: ID=%u, Command=%u, Max=%u\r\n", 
        //BasePacket->ID, BasePacket->Command, Protocol_C2S_t::GetMemberAmount());
        if (BasePacket->Command >= Protocol_C2S_t::GetMemberAmount()) {
          TWriteErrorClient("BasePacket->Command >= Protocol_C2S_t::GetMemberAmount()\r\n");
          TWriteErrorClient("Command: %u, Max: %u\r\n", BasePacket->Command, Protocol_C2S_t::GetMemberAmount());
          A_resize(PeerData->Buffer, 0);
          NET_TCP_CloseHard(peer);
          return NET_TCP_EXT_PeerIsClosed_e;
        }
        PeerData->Buffer = A_resize(PeerData->Buffer, sizeof(ProtocolBasePacket_t) + Protocol_C2S.NA(BasePacket->Command)->m_DSS);
        PeerData->state = PeerState_Waitting_Data_e;
      }
      break;
    }
    case PeerState_Waitting_Data_e: {
      auto BasePacket = (ProtocolBasePacket_t*)PeerData->Buffer;
      uintptr_t TotalSize = sizeof(ProtocolBasePacket_t) + Protocol_C2S.NA(BasePacket->Command)->m_DSS;
      uintptr_t size = ReadSize - iSize;
      uintptr_t needed_size = TotalSize - PeerData->iBuffer;
      if (size > needed_size) {
        size = needed_size;
      }
      __builtin_memcpy(&PeerData->Buffer[PeerData->iBuffer], &ReadData[iSize], size);
      iSize += size;
      PeerData->iBuffer += size;
      if (PeerData->iBuffer == TotalSize) {
        auto RestPacket = (uint8_t*)BasePacket + sizeof(ProtocolBasePacket_t);

        // lets preget things
        auto SessionID = PeerData->SessionID;
        auto Session = &g_pile->SessionList[SessionID];

        switch (BasePacket->Command) {
#include "ReadDone.h"
        default: {
          __unreachable();
        }
        }
      }
      break;
    }
    default: {
      __unreachable();
    }
    }
    continue;
  StateDone_gt:
    A_resize(PeerData->Buffer, 0);
    PeerData->state = PeerState_Idle_e;
  }
  return 0;
}

bool CompareCommand(const uint8_t* Command, uintptr_t* iCommand, uintptr_t CommandSize, const void* Str) {
  for (; *iCommand < CommandSize; (*iCommand)++) {
    if (!STR_ischar_blank(Command[*iCommand])) {
      break;
    }
  }
  uintptr_t StrSize = MEM_cstreu(Str);
  if (StrSize != CommandSize) {
    return 0;
  }
  if (STR_ncasecmp(&Command[*iCommand], Str, StrSize)) {
    return 0;
  }
  return 1;
}

bool GetNextArgument(const uint8_t* Command, uintptr_t* iCommand, uintptr_t CommandSize) {
  for (; *iCommand < CommandSize; (*iCommand)++) {
    if (!STR_ischar_blank(Command[*iCommand])) {
      break;
    }
  }
  for (; *iCommand < CommandSize; (*iCommand)++) {
    if (STR_ischar_blank(Command[*iCommand])) {
      break;
    }
  }
  for (; *iCommand < CommandSize; (*iCommand)++) {
    if (!STR_ischar_blank(Command[*iCommand])) {
      return 1;
    }
  }
  return 0;
}

uintptr_t GetSizeOfArgument(const uint8_t* Command, uintptr_t* iCommand, uintptr_t CommandSize) {
  uintptr_t iCommandLocal = *iCommand;
  for (; iCommandLocal < CommandSize; iCommandLocal++) {
    if (STR_ischar_blank(Command[iCommandLocal])) {
      return iCommandLocal - *iCommand;
    }
  }
  return iCommandLocal - *iCommand;
}

void ProcessInput(uint8_t* Input, uintptr_t InputSize) {
  uintptr_t iCommand = 0;
  uintptr_t CommandSize = GetSizeOfArgument(Input, &iCommand, InputSize);
  if (CompareCommand(Input, &iCommand, CommandSize, "help")) {
    WriteInformation("commands:\r\n"
      "help - shows this\r\n"
    );
  }
  else {
    WriteInformation("failed to find command type help\r\n");
  }
}

bool IsInputLineDone(uint8_t** buffer, uintptr_t* size) {
  for (uintptr_t i = 0; i < *size; i++) {
    if ((*buffer)[i] == 0x7f || (*buffer)[i] == 0x08) {
      if (g_pile->InputSize) {
        g_pile->InputSize--;
      }
      continue;
    }
    if ((*buffer)[i] == '\n' || (*buffer)[i] == '\r') {
      if ((*buffer)[i] == '\r') {
        WriteInformation("\r\n");
      }
      *buffer += i + 1;
      *size -= i + 1;
      return 1;
    }
    g_pile->Input[g_pile->InputSize++] = (*buffer)[i];
  }
  return 0;
}

void evio_stdin_cb(EV_t* listener, EV_event_t* evio_stdin, uint32_t flag) {
  IO_fd_t event_fd;
  EV_event_get_fd(evio_stdin, &event_fd);

  uint8_t _buffer[0x1000];
  uint8_t* buffer = _buffer;
  IO_ssize_t size;
  size = IO_read(&event_fd, buffer, sizeof(_buffer));
  if (size < 0) {
    __abort();
  }
  while (1) {
    if (!IsInputLineDone(&buffer, (uintptr_t*)&size)) {
      return;
    }
    ProcessInput(g_pile->Input, g_pile->InputSize);
    g_pile->InputSize = 0;
  }
}

struct server_frame_tracker_t {
  std::set<uint16_t> received_packets;
  uint16_t expected_count = 0;
  uint64_t first_packet_time = 0;
  Protocol_SessionID_t host_session_id;
};

std::unordered_map<uint16_t, server_frame_tracker_t> g_server_frames;
uint64_t g_last_server_cleanup = 0;

void cleanup_server_frames() {
  uint64_t now = EV_nowi(&g_pile->listener);
  if (now - g_last_server_cleanup < 5000000000ULL) return;

  auto it = g_server_frames.begin();
  while (it != g_server_frames.end()) {
    if (now - it->second.first_packet_time > 10000000000ULL) {
      it = g_server_frames.erase(it);
    }
    else {
      ++it;
    }
  }
  g_last_server_cleanup = now;
}

void request_missing_from_host(Protocol_SessionID_t host_session_id,
  Protocol_ChannelID_t channel_id,
  uint16_t sequence,
  const std::vector<uint16_t>& missing_packets) {
  struct recovery_request_t {
    uint16_t sequence;
    uint16_t missing_count;
  };

  size_t request_size = sizeof(recovery_request_t) + missing_packets.size() * sizeof(uint16_t);
  uint8_t request_buffer[2048];

  auto req = (recovery_request_t*)request_buffer;
  req->sequence = sequence;
  req->missing_count = missing_packets.size();

  auto missing_data = (uint16_t*)(req + 1);
  for (size_t i = 0; i < missing_packets.size(); i++) {
    missing_data[i] = missing_packets[i];
  }

  ProtocolUDP::C2S_t::Channel_ScreenShare_RecoveryRequest_t recovery_packet;
  recovery_packet.ChannelID = channel_id;
  recovery_packet.ChannelSessionID = Protocol_ChannelSessionID_t((Protocol_ChannelSessionID_t::Type)-1);

  UDP_send(host_session_id, 0, ProtocolUDP::C2S_t::Channel_ScreenShare_RecoveryRequest,
    recovery_packet, request_buffer, request_size);
}

void drop_old_incomplete_frames() {
  uint64_t now = EV_nowi(&g_pile->listener);
  auto it = g_server_frames.begin();
  while (it != g_server_frames.end()) {
    uint64_t estimated_rtt = 50000000ULL; // 50ms default if unknown
    uint64_t adaptive_timeout = 50000000ULL + estimated_rtt * 2 + 30000000ULL;
    adaptive_timeout = std::clamp(adaptive_timeout, 80000000ULL, 300000000ULL);

    if (now - it->second.first_packet_time > adaptive_timeout) {
      it = g_server_frames.erase(it);
    }
    else {
      ++it;
    }
  }
}

void track_server_packet(Protocol_SessionID_t host_session_id,
  Protocol_ChannelID_t channel_id,
  uint16_t sequence,
  uint16_t current,
  uint16_t possible) {
  auto& frame = g_server_frames[sequence];

  if (current == 0) {
    frame.expected_count = possible;
    frame.first_packet_time = EV_nowi(&g_pile->listener);
    frame.host_session_id = host_session_id;
  }

  frame.received_packets.insert(current);

  if (frame.expected_count > 0 && frame.received_packets.size() < frame.expected_count) {
    uint64_t now = EV_nowi(&g_pile->listener);
    if (now - frame.first_packet_time > 30000000ULL) {
      std::vector<uint16_t> missing;
      for (uint16_t i = 0; i < frame.expected_count; i++) {
        if (frame.received_packets.find(i) == frame.received_packets.end()) {
          missing.push_back(i);
        }
      }
      if (!missing.empty()) {
        request_missing_from_host(host_session_id, channel_id, sequence, missing);
        g_server_frames.erase(sequence);
      }
    }
  }
  else if (frame.received_packets.size() >= frame.expected_count) {
    g_server_frames.erase(sequence);
  }

  cleanup_server_frames();
  drop_old_incomplete_frames();
}

struct recovery_request_t {
  uint16_t sequence;
  uint16_t missing_count;
};
void evio_udp_cb(EV_t* listener, EV_event_t* evio_udp, uint32_t flag) {
  uint8_t buffer[0x800];
  NET_addr_t dstaddr;
  IO_ssize_t size = NET_recvfrom(&g_pile->udp, buffer, sizeof(buffer), &dstaddr);
  if (size < 0) {
    WriteInformation("%lx\r\n", size);
    __abort();
  }
  if (size == sizeof(buffer)) {
    WriteInformation("where is mtu for this packet??\r\n");
    __abort();
  }

  if ((uintptr_t)size < sizeof(ProtocolUDP::BasePacket_t)) {
    WriteInformation("invalid packet came\n");
    return;
  }

  auto BasePacket = (ProtocolUDP::BasePacket_t*)buffer;
  auto SessionID = BasePacket->SessionID;
  if (Session::IsInvalid(SessionID) == true) {
    return;
  }
  auto Session = &g_pile->SessionList[SessionID];
  if (Session::IsIdentifyInvalid(SessionID, BasePacket->IdentifySecret) == true) {
    uint64_t CurrentTime = EV_nowi(&g_pile->listener);
    if (CurrentTime < Session->UDP.LastInvalidIdentifyAt + Protocol::InformInvalidIdentifyAt) {
      return;
    }
    Session->UDP.LastInvalidIdentifyAt = CurrentTime;
    Protocol_S2C_t::InformInvalidIdentify_t rest;
    rest.ClientIdentify = BasePacket->IdentifySecret;
    rest.ServerIdentify = Session->UDP.IdentifySecret;
    Session::WriteCommand(
      SessionID,
      0,
      Protocol_S2C_t::InformInvalidIdentify,
      rest);
    return;
  }

  switch (BasePacket->Command) {
  case ProtocolUDP::C2S_t::KeepAlive: {

    g_pile->SessionList[SessionID].UDP.Address = dstaddr;

    UDP_send(BasePacket->SessionID, 0, ProtocolUDP::S2C_t::KeepAlive, {}, 0, 0);

    break;
  }
  case ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData: {
    auto RestPacket = (ProtocolUDP::C2S_t::Channel_ScreenShare_Host_StreamData_t*)&BasePacket[1];
    IO_size_t RestPacketSize = size - sizeof(ProtocolUDP::BasePacket_t);
    if (RestPacketSize < sizeof(*RestPacket)) {
      break;
    }
    if (IsAnythingInvalid(SessionID, RestPacket->ChannelID, RestPacket->ChannelSessionID) == true) {
      return;
    }
    if (ScreenShare_IsSessionHost(SessionID, RestPacket->ChannelID) == false) {
      return;
    }

    auto StreamData = (uint8_t*)&RestPacket[1];
    IO_size_t StreamSize = size - (StreamData - buffer);

    auto Body = (ScreenShare_StreamHeader_Body_t*)StreamData;
    uint16_t current = Body->GetCurrent();
    uint16_t sequence = Body->GetSequence();

    uint16_t possible = 0;
    if (current == 0) {
      auto Head = (ScreenShare_StreamHeader_Head_t*)StreamData;
      possible = Head->GetPossible();
    }

    track_server_packet(SessionID, RestPacket->ChannelID, sequence, current, possible);

    ScreenShare_StreamPacket(RestPacket->ChannelID, StreamData, StreamSize);
    break;
  }
  case ProtocolUDP::C2S_t::Channel_ScreenShare_RecoveryRequest: {
    auto RestPacket = (ProtocolUDP::C2S_t::Channel_ScreenShare_RecoveryRequest_t*)&BasePacket[1];
    IO_size_t RestPacketSize = size - sizeof(ProtocolUDP::BasePacket_t);
    if (RestPacketSize < sizeof(*RestPacket)) {
      break;
    }

    if (IsAnythingInvalid(SessionID, RestPacket->ChannelID, RestPacket->ChannelSessionID) == true) {
      break;
    }

    auto recovery_data = (uint8_t*)&RestPacket[1];
    IO_size_t recovery_size = size - (recovery_data - buffer);

    if (recovery_size < sizeof(recovery_request_t)) {
      break;
    }

    auto req = (recovery_request_t*)recovery_data;

    if (recovery_size < sizeof(recovery_request_t) + req->missing_count * sizeof(uint16_t)) {
      break;
    }

    auto missing_indices = (uint16_t*)(req + 1);

    ProcessRecoveryRequest(RestPacket->ChannelID, req->sequence, missing_indices, req->missing_count);

    break;
  }
  }
}

int main(int argc, char** argv) {
  uint16_t port = 43254;
  if (argc != 2) {
    fan::print("usage:a.exe <port number>");
    fan::print("defaulting to port:", port);
  }
  else {
    port = std::stoul(argv[1]);
  }

  g_pile = new pile_t;

  sint32_t err;

  RAND_hard_ram(g_pile->ServerRandom, sizeof(g_pile->ServerRandom));

  EV_open(&g_pile->listener);

  EV_event_t evio_stdin;
  IO_fd_t fd_stdin;
  IO_fd_set(&fd_stdin, FD_IN);
  EV_event_init_fd(&evio_stdin, &fd_stdin, evio_stdin_cb, EV_READ);
  EV_event_start(&g_pile->listener, &evio_stdin);

  if (NET_socket2(NET_AF_INET, NET_SOCK_DGRAM | NET_SOCK_NONBLOCK, NET_IPPROTO_UDP, &g_pile->udp) < 0) {
    WriteInformation("%lx\r\n", g_pile->udp);
    __abort();
  }

  if (NET_setsockopt(&g_pile->udp, SOL_SOCKET, SO_SNDBUF, 8388608) < 0) {
    WriteInformation("Failed to set SO_SNDBUF\r\n");
  }

  if (NET_setsockopt(&g_pile->udp, SOL_SOCKET, SO_RCVBUF, 8388608) < 0) {
    WriteInformation("Failed to set SO_RCVBUF\r\n");
  }

  NET_addr_t udpaddr;
  udpaddr.ip = NET_INADDR_ANY;
  udpaddr.port = port;
  err = NET_bind(&g_pile->udp, &udpaddr);
  if (err) {
    WriteInformation("%lx\r\n");
    __abort();
  }

  EV_event_t evio_udp;
  EV_event_init_socket(&evio_udp, &g_pile->udp, evio_udp_cb, EV_READ);
  EV_event_start(&g_pile->listener, &evio_udp);

  g_pile->TCP = NET_TCP_alloc(&g_pile->listener);
  g_pile->TCP->ssrcaddr.port = port;

  g_pile->extid = NET_TCP_EXT_new(g_pile->TCP, sizeof(TCPMain_SockData_t), sizeof(TCPMain_PeerData_t));
  TCPMain_SockData_t* SockData = (TCPMain_SockData_t*)NET_TCP_GetSockData(g_pile->TCP, g_pile->extid);

  NET_TCP_layer_state_open(g_pile->TCP, g_pile->extid, (NET_TCP_cb_state_t)TCPMain_state_cb);
  SockData->ReadLayerID = NET_TCP_layer_read_open(g_pile->TCP, g_pile->extid, (NET_TCP_cb_read_t)TCPMain_read_cb, A_resize, 0, 0);

  NET_TCP_open(g_pile->TCP);
  err = NET_TCP_listen(g_pile->TCP);
  if (err) {
    WriteInformation("listen error %ld\r\n", err);
    __abort();
  }
  EV_event_start(&g_pile->listener, &g_pile->TCP->ev);

  g_pile->InputSize = 0;

  EV_start(&g_pile->listener);
  return 0;
}
