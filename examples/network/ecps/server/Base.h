#include <WITCH/WITCH.h>
#include <WITCH/STR/psh.h>

#include <fan/types/vector_simple.h>

#include "common.h"

#ifndef set_Verbose
  #define set_Verbose 0
#endif

#define TWriteErrorClient(...) _print(FD_ERR, "[ErrorClient] " __VA_ARGS__)

void TCP_WriteCommand(NET_TCP_peer_t *peer, uint32_t ID, Protocol_CI_t Command, auto&&...args){
  ProtocolBasePacket_t BasePacket;
  BasePacket.ID = ID;
  BasePacket.Command = Command;
  TCP_write_DynamicPointer(peer, &BasePacket, sizeof(BasePacket));
  (
    TCP_write_DynamicPointer(peer, &args, sizeof(args)),
  ...);
}

#define BLL_set_Language 1
#define BLL_set_type_node Protocol_ChannelSessionID_t::Type
#define BLL_set_prefix ChannelSessionList
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#define BLL_set_IsNodeRecycled 1
#define BLL_set_NodeReference_Overload_Declare \
  ChannelSessionList_NodeReference_t(Protocol_ChannelSessionID_t ID){ \
    NRI = ID; \
  }
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_type_node Protocol_SessionChannelID_t::Type
#define BLL_set_prefix SessionChannelList
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#define BLL_set_IsNodeRecycled 1
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_type_node Protocol_AccountID_t::Type
#define BLL_set_prefix AccountList
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#define BLL_set_IsNodeRecycled 1
#define BLL_set_NodeReference_Overload_Declare \
  AccountList_NodeReference_t(Protocol_AccountID_t ID){ \
    NRI = ID; \
  }
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_type_node Protocol_SessionID_t::Type
#define BLL_set_prefix SessionList
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#define BLL_set_IsNodeRecycled 1
#define BLL_set_NodeReference_Overload_Declare \
  SessionList_NodeReference_t(Protocol_SessionID_t ID){ \
    NRI = ID; \
  }
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_type_node Protocol_ChannelID_t::Type
#define BLL_set_prefix ChannelList
#define BLL_set_declare_NodeReference 1
#define BLL_set_declare_rest 0
#define BLL_set_IsNodeRecycled 1
#define BLL_set_NodeReference_Overload_Declare \
  ChannelList_NodeReference_t(Protocol_ChannelID_t ID){ \
    NRI = ID; \
  }
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_type_node Protocol_ChannelSessionID_t::Type
#define BLL_set_prefix ChannelSessionList
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#define BLL_set_NodeData \
  Protocol_SessionID_t SessionID; \
  SessionChannelList_NodeReference_t SessionChannelID;
#define BLL_set_IsNodeRecycled 1
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_type_node Protocol_SessionChannelID_t::Type
#define BLL_set_prefix SessionChannelList
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#define BLL_set_NodeData \
  ChannelList_NodeReference_t ChannelID; \
  ChannelSessionList_NodeReference_t ChannelSessionID;
#define BLL_set_IsNodeRecycled 1
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_type_node Protocol_AccountID_t::Type
#define BLL_set_prefix AccountList
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#define BLL_set_NodeData \
  uint8_t nothing_yet;
#define BLL_set_IsNodeRecycled 1
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_type_node Protocol_SessionID_t::Type
#define BLL_set_prefix SessionList
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#define BLL_set_NodeData \
  Protocol_AccountID_t AccountID; \
  SessionChannelList_t ChannelList; \
  struct{ \
    NET_TCP_peer_t *peer; \
  }TCP; \
  struct{ \
    uint64_t LastInvalidIdentifyAt; \
    NET_addr_t Address; \
    uint64_t IdentifySecret; \
  }UDP;
#define BLL_set_IsNodeRecycled 1
#include <BLL/BLL.h>

#define BLL_set_Language 1
#define BLL_set_CPP_ConstructDestruct 1
#define BLL_set_type_node Protocol_ChannelID_t::Type
#define BLL_set_prefix ChannelList
#define BLL_set_declare_NodeReference 0
#define BLL_set_declare_rest 1
#define BLL_set_NodeData \
  uint8_t Type; \
  ChannelSessionList_t SessionList; \
  uint8_t *Buffer;
#define BLL_set_IsNodeRecycled 1
#include <BLL/BLL.h>

struct Channel_ScreenShare_Data_t{
  uint8_t Flag;
  Protocol_SessionID_t HostSessionID;
  std::unordered_map<uint16_t, std::vector<std::vector<uint8_t>>> SentPackets;
};

struct pile_t{
  EV_t listener;

  NET_TCP_t *TCP;

  NET_TCP_extid_t extid;

  /* need to be unique to every server */
  /* user will hash secure things with it */
  uint8_t ServerRandom[HASH_SHA512_size];

  SessionList_t SessionList;
  AccountList_t AccountList;
  ChannelList_t ChannelList;

  NET_socket_t udp;

  uint8_t Input[0x1000];
  uintptr_t InputSize;
};

pile_t *g_pile;

Protocol_SessionID_t::Protocol_SessionID_t(auto p){
  static_assert(__is_type_same<decltype(p), SessionList_NodeReference_t>);
  *this = *(decltype(this))&p;
}
Protocol_SessionID_t::Protocol_SessionID_t(
  auto ChannelID,
  auto ChannelSessionID
){
  static_assert(__is_type_same<decltype(ChannelID), Protocol_ChannelID_t>);
  static_assert(__is_type_same<decltype(ChannelSessionID), Protocol_ChannelSessionID_t>);
  *this = *(decltype(this))&g_pile->ChannelList[ChannelID].SessionList[ChannelSessionID].SessionID;
}
Protocol_ChannelID_t::Protocol_ChannelID_t(auto p){
  static_assert(__is_type_same<decltype(p), ChannelList_NodeReference_t>);
  *this = *(decltype(this))&p;
}
Protocol_ChannelSessionID_t::Protocol_ChannelSessionID_t(auto p){
  static_assert(__is_type_same<decltype(p), ChannelSessionList_NodeReference_t>);
  *this = *(decltype(this))&p;
}

/* peer state enums */
enum{
  PeerState_Idle_e,
  PeerState_Waitting_BasePacket_e,
  PeerState_Waitting_Data_e
};

struct TCPMain_SockData_t{
  NET_TCP_layerid_t ReadLayerID;
};
struct TCPMain_PeerData_t{
  Protocol_SessionID_t SessionID;
  uint8_t state;
  uint32_t iBuffer;
  uint8_t *Buffer;
};
