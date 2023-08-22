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

#include <WITCH/STR/psu.h>
#include <WITCH/STR/pss.h>
#include <WITCH/STR/psh.h>

#define ETC_HTTP_set_prefix HTTP
#include <WITCH/ETC/HTTP/HTTP.h>

#define ETC_HTTP_chunked_set_prefix HTTP_Chunked
#include <WITCH/ETC/HTTP/chunked.h>

/* external json parser */
#include "jsmn.h"

/* for utf16 to utf8 */
#include "utf.h"

#define TotalCPType 2

typedef enum {
  CPType_Telegram,
  CPType_Discord
}CPType;

#define set_NameSizeLimit 64

typedef struct {
  CPType CPFrom;
  uint32_t CPReferenceCount;
  sint64_t UserID;
  uint8_t Name[set_NameSizeLimit];
  uint32_t NameSize;
  void* Text;
  uintptr_t TextSize;
}Message_t;

#define BLL_set_prefix MessageList
#define BLL_set_NodeDataType Message_t
#include <WITCH/BLL/BLL.h>

MessageList_t MessageList;

struct {
  MessageList_NodeReference_t Verify;
  MessageList_NodeReference_t Sent;
}CPMessageAt[TotalCPType];

typedef enum {
  HTTP_TransferType_Unknown,
  HTTP_TransferType_ContentLength,
  HTTP_TransferType_Chunked
}HTTP_TransferType_t;

typedef struct {
  union {
    uint64_t ContentLength;
    HTTP_Chunked_Parser_t Chunked;
  };
  HTTP_TransferType_t Type; // HTTP_TransferType_Unknown
}HTTP_Transfer_t;

typedef enum {
  TelegramPacketType_GetUpdate,
  TelegramPacketType_SendMessage
}TelegramPacketType;

#define BLL_set_prefix TelegramPacketTypeList
#define BLL_set_NodeDataType TelegramPacketType
#include <WITCH/BLL/BLL.h>

typedef struct {
  union {
    HTTP_decode_t HTTP_decode; // init
    struct {
      VEC_t jvec;
      jsmn_parser jparser;
      jsmntok_t jtokens[128];
      bool IsItOK;
    };
  };
  uint16_t HTTPCode;
  bool ContentTypeJSON; // false
  bool WeGetHTTPNow; // true
  uint32_t ContentLength;

  TelegramPacketTypeList_t PacketTypeList;
}pd_Telegram_t;

typedef enum {
  DiscordPacketType_CreateMessage
}DiscordPacketType;

#define BLL_set_prefix DiscordPacketTypeList
#define BLL_set_NodeDataType DiscordPacketType
#include <WITCH/BLL/BLL.h>

typedef struct {
  union {
    HTTP_decode_t HTTP_decode; // init
    struct {
      VEC_t jvec;
      jsmn_parser jparser;
      jsmntok_t jtokens[128];
    };
  };
  uint16_t HTTPCode;
  bool ContentTypeJSON; // false
  bool WeGetHTTPNow; // true
  HTTP_Transfer_t Transfer;

  DiscordPacketTypeList_t PacketTypeList;
}pd_Discord_t;

typedef struct {
  NET_TCP_t* tcp;
  NET_TCP_peer_t* peer;

  NET_TCP_extid_t extid;
  NET_TCP_layerid_t LayerStateID;
  NET_TCP_layerid_t LayerReadID;
}tcppile_t;

typedef struct {
  EV_t listener;

  EV_timer_t InitialTimer;

  tcppile_t Telegram;
  uint64_t TelegramUpdateID;

  tcppile_t Discord;
}pile_t;
pile_t pile;

void Send_telegram_Message(const Message_t* Message) {
  pd_Telegram_t* pd = (pd_Telegram_t*)NET_TCP_GetPeerData(pile.Telegram.peer, pile.Telegram.extid);

  {
    TelegramPacketTypeList_NodeReference_t nr = TelegramPacketTypeList_NewNodeLast(&pd->PacketTypeList);
    TelegramPacketTypeList_Node_t* n = TelegramPacketTypeList_GetNodeByReference(&pd->PacketTypeList, nr);
    n->data = TelegramPacketType_SendMessage;
  }

  VEC_t buf;
  VEC_init(&buf, 1, A_resize);

  VEC_print(&buf, "GET /bot6619857197:AAEkKjUxc1pJ_91tlAD-74z41Z3zVrtvGWc/sendMessage?chat_id=-956834021&text=");

  VEC_print(&buf, "%lx:%llx %.*s%%0a", CPType_Telegram, Message->UserID, Message->NameSize, Message->Name);

  bool EscapeCame = false;
  for (uintptr_t i = 0; i < Message->TextSize; i++) {
    if (EscapeCame == true) {
      EscapeCame = false;
      if (((uint8_t*)Message->Text)[i] == '\\') {
        VEC_print(&buf, "\\");
      }
      else if (((uint8_t*)Message->Text)[i] == 'n') {
        VEC_print(&buf, "%%0a");
      }
      else if (((uint8_t*)Message->Text)[i] == 'u') {
        uintptr_t hsize = i + 1;
        for (; hsize < Message->TextSize; hsize++) {
          if (STR_ischar_hexdigit(((uint8_t*)Message->Text)[hsize]) == false) {
            break;
          }
        }
        hsize -= i + 1;
        uint16_t hcode = STR_psh32_digit(&((uint8_t*)Message->Text)[i + 1], hsize);
        i += hsize; // i++ will came in end.
        uint8_t utf8c[8];
        size_t utf8size = utf16_to_utf8(&hcode, 1, utf8c, 8);
        VEC_print(&buf, "%.*s", utf8size, utf8c);
      }
    }
    else if (((uint8_t*)Message->Text)[i] < 0x80) {
      if (((uint8_t*)Message->Text)[i] == '\\') {
        EscapeCame = true;
      }
      else if (
        STR_ischar_digit(((uint8_t*)Message->Text)[i]) == false &&
        STR_ischar_char(((uint8_t*)Message->Text)[i]) == false
        ) {
        VEC_print(&buf, "%%%lx", ((uint8_t*)Message->Text)[i]);
      }
      else {
        VEC_print(&buf, "%c", ((uint8_t*)Message->Text)[i]);
      }
    }
    else {
      VEC_print(&buf, "%c", ((uint8_t*)Message->Text)[i]);
    }
  }

  VEC_print(&buf,
    " HTTP/1.1\r\n"
    "Host: api.telegram.org\r\n"
    "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8\r\n"
    "Accept-Language: en-US,en;q=0.5\r\n"
    "DNT: 1\r\n"
    "Connection: keep-alive\r\n\r\n");

  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = buf.ptr;
  Queue.DynamicPointer.size = buf.Current;
  NET_TCP_write_loop(
    pile.Telegram.peer,
    NET_TCP_GetWriteQueuerReferenceFirst(pile.Telegram.peer),
    NET_TCP_QueueType_DynamicPointer_e,
    &Queue);

  VEC_free(&buf);
}

void Send_Discord_Message(const Message_t* Message) {
  pd_Discord_t* pd = (pd_Discord_t*)NET_TCP_GetPeerData(pile.Discord.peer, pile.Discord.extid);

  {
    DiscordPacketTypeList_NodeReference_t nr = DiscordPacketTypeList_NewNodeLast(&pd->PacketTypeList);
    DiscordPacketTypeList_Node_t* n = DiscordPacketTypeList_GetNodeByReference(&pd->PacketTypeList, nr);
    n->data = DiscordPacketType_CreateMessage;
  }

  VEC_t buf;
  VEC_init(&buf, 1, A_resize);

  VEC_t jbuf;
  VEC_init(&jbuf, 1, A_resize);

  VEC_print(&jbuf,
    "{"
    "\"content\": \"");

  VEC_print(&jbuf, "%lx:%llx %.*s\\n", Message->CPFrom, Message->UserID, Message->NameSize, Message->Name);

  bool EscapeCame = false;
  for (uintptr_t i = 0; i < Message->TextSize; i++) {
    if (EscapeCame == true) {
      EscapeCame = false;
      if (((uint8_t*)Message->Text)[i] == '\\') {
        VEC_print(&jbuf, "\\");
      }
      else if (((uint8_t*)Message->Text)[i] == 'n') {
        VEC_print(&jbuf, "\\n");
      }
      else if (((uint8_t*)Message->Text)[i] == 'u') {
        uintptr_t hsize = i + 1;
        for (; hsize < Message->TextSize; hsize++) {
          if (STR_ischar_hexdigit(((uint8_t*)Message->Text)[hsize]) == false) {
            break;
          }
        }
        hsize -= i + 1;
        uint16_t hcode = STR_psh32_digit(&((uint8_t*)Message->Text)[i + 1], hsize);
        i += hsize; // i++ will came in end.
        uint8_t utf8c[8];
        size_t utf8size = utf16_to_utf8(&hcode, 1, utf8c, 8);
        VEC_print(&jbuf, "%.*s", utf8size, utf8c);
      }
    }
    else if (((uint8_t*)Message->Text)[i] < 0x80) {
      if (((uint8_t*)Message->Text)[i] == '\\') {
        EscapeCame = true;
      }
      else if (
        STR_ischar_digit(((uint8_t*)Message->Text)[i]) == false &&
        STR_ischar_char(((uint8_t*)Message->Text)[i]) == false
        ) {
        VEC_print(&jbuf, "%%%lx", ((uint8_t*)Message->Text)[i]);
      }
      else {
        VEC_print(&jbuf, "%c", ((uint8_t*)Message->Text)[i]);
      }
    }
    else {
      VEC_print(&jbuf, "%c", ((uint8_t*)Message->Text)[i]);
    }
  }

  VEC_print(&jbuf,
    "\","
    "\"tts\": false"
    "}");

  VEC_print(&buf,
    "POST /api/v9/channels/1141090206901092458/messages HTTP/1.1\r\n"
    "Host: discord.com\r\n"
    "Authorization: Bot MTE0MTA4MjI2ODQ1MDk1MTIxOQ.G1lCr-.xRksTEheNKasUAQgkbSWG4jC1mGm8DRJRW3tDw\r\n"
    "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8\r\n"
    "Accept-Language: en-US,en;q=0.5\r\n"
    "Content-Type: application/json\r\n"
    "DNT: 1\r\n"
    "Connection: keep-alive\r\n"
    "Content-Length: %u\r\n\r\n"
    "%.*s",
    jbuf.Current, jbuf.Current, jbuf.ptr);

  VEC_free(&jbuf);

  print("raw send:\n%.*s\n", buf.Current, buf.ptr);

  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = buf.ptr;
  Queue.DynamicPointer.size = buf.Current;
  NET_TCP_write_loop(
    pile.Discord.peer,
    NET_TCP_GetWriteQueuerReferenceFirst(pile.Discord.peer),
    NET_TCP_QueueType_DynamicPointer_e,
    &Queue);

  VEC_free(&buf);
}

void BroadcastMessages_Telegram() {
  MessageList_NodeReference_t* mnr = &CPMessageAt[CPType_Telegram].Sent;
  while (MessageList_IsNodeReferenceEqual(*mnr, MessageList.dst) == false) {
    MessageList_Node_t* mn = MessageList_GetNodeByReference(&MessageList, *mnr);
    Send_telegram_Message(&mn->data);
    *mnr = mn->NextNodeReference;
  }
}

void BroadcastMessages_Discord() {
  MessageList_NodeReference_t* mnr = &CPMessageAt[CPType_Discord].Sent;
  while (MessageList_IsNodeReferenceEqual(*mnr, MessageList.dst) == false) {
    MessageList_Node_t* mn = MessageList_GetNodeByReference(&MessageList, *mnr);
    Send_Discord_Message(&mn->data);
    *mnr = mn->NextNodeReference;
  }
}

void VerifyMessage(CPType cp) {
  MessageList_NodeReference_t* nr = &CPMessageAt[cp].Verify;
  MessageList_Node_t* n = MessageList_GetNodeByReference(&MessageList, *nr);
  MessageList_NodeReference_t nnr = n->NextNodeReference;
  if (--n->data.CPReferenceCount == 0) {
    A_resize(n->data.Text, 0);
    MessageList_unlrec(&MessageList, *nr);
  }
  print("usage %lx\n", MessageList_Usage(&MessageList));
  *nr = nnr;
}

void BroadcastNewMessages() {
  if (pile.Telegram.peer != NULL) {
    BroadcastMessages_Telegram();
  }
  if (pile.Discord.peer != NULL) {
    BroadcastMessages_Discord();
  }
}

void ConnectServer_Telegram() {
  uint32_t ips[] = {
    0x959aa7dc
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
  sint32_t err = NET_TCP_connect(pile.Telegram.tcp, &FillerPeer, &addr, &sockopt, 1);
  if (err != 0) {
    PR_abort();
  }
}
void ConnectServer_Discord() {
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
  sint32_t err = NET_TCP_connect(pile.Discord.tcp, &FillerPeer, &addr, &sockopt, 1);
  if (err != 0) {
    PR_abort();
  }
}

void InitHTTPVariables_Telegram() {
  pd_Telegram_t* pd = (pd_Telegram_t*)NET_TCP_GetPeerData(pile.Telegram.peer, pile.Telegram.extid);

  HTTP_decode_init(&pd->HTTP_decode);
  pd->ContentTypeJSON = false;
  pd->WeGetHTTPNow = true;
  pd->ContentLength = (uint32_t)-1;
}

void InitHTTPVariables_Discord() {
  pd_Discord_t* pd = (pd_Discord_t*)NET_TCP_GetPeerData(pile.Discord.peer, pile.Discord.extid);

  HTTP_decode_init(&pd->HTTP_decode);
  pd->ContentTypeJSON = false;
  pd->WeGetHTTPNow = true;

  pd->Transfer.Type = HTTP_TransferType_Unknown;
}

void Send_telegram_GetUpdate() {
  pd_Telegram_t* pd = (pd_Telegram_t*)NET_TCP_GetPeerData(pile.Telegram.peer, pile.Telegram.extid);

  {
    TelegramPacketTypeList_NodeReference_t nr = TelegramPacketTypeList_NewNodeLast(&pd->PacketTypeList);
    TelegramPacketTypeList_Node_t* n = TelegramPacketTypeList_GetNodeByReference(&pd->PacketTypeList, nr);
    n->data = TelegramPacketType_GetUpdate;
  }

  VEC_t buf;
  VEC_init(&buf, 1, A_resize);

  VEC_print(&buf,
    "GET /bot6619857197:AAEkKjUxc1pJ_91tlAD-74z41Z3zVrtvGWc/getUpdates?timeout=8&limit=2&offset=%llu HTTP/1.1\r\n"
    "Host: api.telegram.org\r\n"
    "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8\r\n"
    "Accept-Language: en-US,en;q=0.5\r\n"
    "DNT: 1\r\n"
    "Connection: keep-alive\r\n\r\n",
    pile.TelegramUpdateID + 1);

  NET_TCP_Queue_t Queue;
  Queue.DynamicPointer.ptr = buf.ptr;
  Queue.DynamicPointer.size = buf.Current;
  NET_TCP_write_loop(
    pile.Telegram.peer,
    NET_TCP_GetWriteQueuerReferenceFirst(pile.Telegram.peer),
    NET_TCP_QueueType_DynamicPointer_e,
    &Queue);

  VEC_free(&buf);
}

#include "Telegram/tcp_cb.h"
#include "Discord/tcp_cb.h"

void cb_InitialTimer(EV_t* listener, EV_timer_t* evt, uint32_t flag) {
  ConnectServer_Telegram();
  ConnectServer_Discord();

  EV_timer_stop(listener, evt);
}

int main() {
  MessageList_Open(&MessageList); // TOOD close it
  for (uint32_t i = 0; i < TotalCPType; i++) {
    CPMessageAt[i].Verify = MessageList.dst;
    CPMessageAt[i].Sent = MessageList.dst;
  }

  pile.TelegramUpdateID = 0;

  EV_open(&pile.listener);

  {
    pile.Telegram.tcp = NET_TCP_alloc(&pile.listener);
    pile.Telegram.peer = NULL;

    {
      TLS_ctx_t tctx;
      bool r = TLS_ctx_generate(&tctx);
      if (r != 0) {
        PR_abort();
      }
      NET_TCP_TLS_add(pile.Telegram.tcp, &tctx);
    }

    pile.Telegram.extid = NET_TCP_EXT_new(pile.Telegram.tcp, 0, sizeof(pd_Telegram_t));
    pile.Telegram.LayerStateID = NET_TCP_layer_state_open(
      pile.Telegram.tcp,
      pile.Telegram.extid,
      (NET_TCP_cb_state_t)cb_connstate_Telegram);
    pile.Telegram.LayerReadID = NET_TCP_layer_read_open(
      pile.Telegram.tcp,
      pile.Telegram.extid,
      (NET_TCP_cb_read_t)cb_read_Telegram,
      0,
      0,
      0
    );

    NET_TCP_open(pile.Telegram.tcp);
  }
  {
    pile.Discord.tcp = NET_TCP_alloc(&pile.listener);
    pile.Discord.peer = NULL;

    {
      TLS_ctx_t tctx;
      bool r = TLS_ctx_generate(&tctx);
      if (r != 0) {
        PR_abort();
      }
      NET_TCP_TLS_add(pile.Discord.tcp, &tctx);
    }

    pile.Discord.extid = NET_TCP_EXT_new(pile.Discord.tcp, 0, sizeof(pd_Discord_t));
    pile.Discord.LayerStateID = NET_TCP_layer_state_open(
      pile.Discord.tcp,
      pile.Discord.extid,
      (NET_TCP_cb_state_t)cb_connstate_Discord);
    pile.Discord.LayerReadID = NET_TCP_layer_read_open(
      pile.Discord.tcp,
      pile.Discord.extid,
      (NET_TCP_cb_read_t)cb_read_Discord,
      0,
      0,
      0
    );

    NET_TCP_open(pile.Discord.tcp);
  }

  EV_timer_init(&pile.InitialTimer, 1, (EV_timer_cb_t)cb_InitialTimer);
  EV_timer_start(&pile.listener, &pile.InitialTimer);

  EV_start(&pile.listener);

  return 0;
}
