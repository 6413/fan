#pragma pack(push, 1)

struct Protocol_SessionID_t {
  typedef uint32_t Type;
  Type i;
  operator Type&() { return i; }
  Protocol_SessionID_t() = default;
#ifndef __ecps_client
  Protocol_SessionID_t(auto);
  Protocol_SessionID_t(auto, auto);
#endif
  Protocol_SessionID_t(Type v) : i(v) {}
  bool operator==(Protocol_SessionID_t SessionID){
    return (Type)*this == (Type)SessionID;
  }
};
struct Protocol_AccountID_t {
  typedef uint32_t Type;
  Type i;
  operator Type& () { return i; }
  Protocol_AccountID_t() = default;
  Protocol_AccountID_t(Type v) : i(v) {}
  static Protocol_AccountID_t GetInvalid(){
    Protocol_AccountID_t r;
    r = (Type)-1;
    return r;
  }
};
struct Protocol_ChannelID_t {
  typedef uint16_t Type;
  Type i;
  operator Type& () { return i; }
  Protocol_ChannelID_t() = default;
  Protocol_ChannelID_t(Type v) : i(v) {}
#ifndef __ecps_client
  Protocol_ChannelID_t(auto p);
#endif
  void invalidate() { i = (uint16_t)-1; }
};
struct Protocol_ChannelSessionID_t {
  typedef uint32_t Type;
  Type i;
  operator Type& () { return i; }
  Protocol_ChannelSessionID_t() = default;
  Protocol_ChannelSessionID_t(Type v) : i(v) {}
#ifndef __ecps_client
  Protocol_ChannelSessionID_t(auto p);
#endif
};
struct Protocol_SessionChannelID_t {
  typedef Protocol_ChannelID_t::Type Type;
  Type i;
  operator Type& () { return i; }
  Protocol_SessionChannelID_t() = default;
  Protocol_SessionChannelID_t(Type v) : i(v) {}
};

typedef uint16_t Protocol_CI_t;

struct ProtocolBasePacket_t{
  uint32_t ID;
  Protocol_CI_t Command;
};

struct channel_list_info_t {
  Protocol_ChannelID_t channel_id;
  uint8_t type;
  uint32_t user_count;
  std::string name;
  bool is_password_protected;
  Protocol_SessionID_t host_session_id;
};

struct session_info_t {
  Protocol_SessionID_t session_id;
  Protocol_ChannelSessionID_t channel_session_id;
  Protocol_AccountID_t account_id;
  std::string username;
  bool is_host;
  uint64_t joined_at;
};

struct tcp {
  struct ProtocolBasePacket_t {
    uint32_t ID;
    Protocol_CI_t Command;
  };
};

struct udp {
  struct BasePacket_t {
    Protocol_SessionID_t SessionID;
    uint32_t ID;
    uint64_t IdentifySecret;
    Protocol_CI_t Command;
  };
};

struct ProtocolChannel {
  struct ScreenShare {
    struct ChannelFlag {
      using _t = uint8_t;
      inline static constexpr _t InputControl = 0x01;
      inline static constexpr _t ResetIDR = 0x01 << 1;
    };
    struct StreamHeadFlag {
      using _t = uint8_t;
      inline static _t KeyFrame = 0x01;
    };
  };
};

struct Protocol {
  inline static constexpr uint64_t InformInvalidIdentifyAt = 1000000000;
  inline static constexpr uint32_t ChannelType_Amount = 1;
  enum {
    ChannelType_ScreenShare_e
  };
  const uint8_t* ChannelType_Text[ChannelType_Amount] = {
    (const uint8_t*)"ScreenShare"
  };

  enum class KickedFromChannel_Reason_t : uint8_t {
    Unknown,
    ChannelIsClosed
  };
  inline static const char* KickedFromChannel_Reason_String[] = {
    "Unknown",
    "ChannelIsClosed"
  };

  enum class JoinChannel_Error_Reason_t : uint8_t {
    InvalidChannelType,
    InvalidChannelID
  };
  inline static const char* JoinChannel_Error_Reason_String[] = {
    "InvalidChannelType",
    "InvalidChannelID"
  };
  enum class LoginType_t : uint8_t {
    Anonymous
  };
};

struct ProtocolUDP {
  struct BasePacket_t {
    Protocol_SessionID_t SessionID;
    uint32_t ID;
    uint64_t IdentifySecret;
    Protocol_CI_t Command;
  };
  struct C2S_t : __dme_inherit(C2S_t) {
    __dme(KeepAlive);
    __dme(Channel_ScreenShare_Host_StreamData,
      Protocol_ChannelID_t ChannelID;
      Protocol_ChannelSessionID_t ChannelSessionID;
    );
    __dme(Channel_ScreenShare_RecoveryRequest,
      Protocol_ChannelID_t ChannelID;
      Protocol_ChannelSessionID_t ChannelSessionID;
    );
  }C2S;
  struct S2C_t : __dme_inherit(S2C_t) {
    __dme(KeepAlive);
    __dme(Channel_ScreenShare_View_StreamData,
      Protocol_ChannelID_t ChannelID;
    );
  }S2C;
};

struct Protocol_C2S_t : __dme_inherit(Protocol_C2S_t){
  __dme(KeepAlive);
  
  __dme(Request_Login,
    Protocol::LoginType_t Type;
  );
  
  __dme(CreateChannel,
    uint8_t Type;
  );
  
  __dme(JoinChannel,
    Protocol_ChannelID_t ChannelID;
  );
  
  __dme(QuitChannel,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
  );
  
  __dme(Response_UDPIdentifySecret,
    uint64_t UDPIdentifySecret;
  );
  
  __dme(RequestChannelList,
    uint8_t pad; // for some reason command doesnt get processed without pad in server
  );
  
  __dme(RequestChannelSessionList,
    Protocol_ChannelID_t ChannelID;
  );
  
  // Screen sharing control messages
  __dme(Channel_ScreenShare_Share_InformationToViewSetFlag,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
    ProtocolChannel::ScreenShare::ChannelFlag::_t Flag;
  );
  
  __dme(Channel_ScreenShare_Share_InformationToViewMouseCoordinate,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
    fan::vec2ui pos;
  );
  
  __dme(Channel_ScreenShare_View_ApplyToHostMouseCoordinate,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
    fan::vec2si pos;
  );
  
  __dme(Channel_ScreenShare_View_ApplyToHostMouseMotion,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
    fan::vec2si Motion;
  );
  
  __dme(Channel_ScreenShare_View_ApplyToHostMouseButton,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
    uint8_t key;
    bool state;
    fan::vec2si pos;
  );
  
  __dme(Channel_ScreenShare_View_ApplyToHostKeyboard,
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
    uint16_t Scancode;
    bool State;
  );

  __dme(Channel_ScreenShare_ViewToShare,
    Protocol_ChannelID_t ChannelID;
    uint16_t Flag;
  );
  __dme(Channel_ScreenShare_ShareToView,
    Protocol_ChannelID_t ChannelID;
    uint16_t Flag;
  );
}Protocol_C2S;

#ifndef __ecps_client
  struct ecps_backend_t;
#endif

static fan::event::task_t default_s2c_cb(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base);

using S2C_callback_inherit_t = std::function<
  fan::event::task_t(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base)
>;
struct S2C_callback_t : S2C_callback_inherit_t {
  using S2C_callback_inherit_t::S2C_callback_inherit_t;
  S2C_callback_t() : S2C_callback_inherit_t(default_s2c_cb) {};
};

struct Protocol_S2C_t : __dme_inherit(Protocol_S2C_t, S2C_callback_t) {
   __dme(KeepAlive);
  
  __dme(InformInvalidIdentify,
    uint64_t ClientIdentify;
    uint64_t ServerIdentify;
  );
  
  __dme(Response_Login,
    Protocol_AccountID_t AccountID;
    Protocol_SessionID_t SessionID;
  );
  
  __dme(CreateChannel_OK,
    uint8_t Type;
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
  );
  
  __dme(CreateChannel_Error,
    Protocol::JoinChannel_Error_Reason_t Reason;
  );
  
  __dme(JoinChannel_OK,
    uint8_t Type;
    Protocol_ChannelID_t ChannelID;
    Protocol_ChannelSessionID_t ChannelSessionID;
  );
  
  __dme(JoinChannel_Error,
    Protocol::JoinChannel_Error_Reason_t Reason;
  );
  
  __dme(KickedFromChannel,
    Protocol_ChannelID_t ChannelID;
    Protocol::KickedFromChannel_Reason_t Reason;
  );
  
  __dme(Request_UDPIdentifySecret);
  
  __dme(UseThisUDPIdentifySecret,
    uint64_t UDPIdentifySecret;
  );
  
  struct ChannelInfo_t {
    Protocol_ChannelID_t ChannelID;        // Channel's unique ID
    uint8_t Type;                          // Channel type (ScreenShare, etc.)
    uint32_t UserCount;                    // Current number of users in channel
    char Name[64];                         // Channel name (null-terminated)
    uint8_t IsPasswordProtected;           // 1 if password required, 0 if not
    Protocol_SessionID_t HostSessionID;    // Session ID of channel host/creator
  };
  
  struct SessionInfo_t {
    Protocol_SessionID_t SessionID;           // User's session ID
    Protocol_ChannelSessionID_t ChannelSessionID; // User's channel session ID
    Protocol_AccountID_t AccountID;           // User's account ID
    char Username[32];                       // Username (null-terminated)
    uint8_t IsHost;                         // 1 if user is channel host, 0 if not
    uint64_t JoinedAt;                      // Timestamp when user joined channel
  };
  
  __dme(ChannelList,
    uint16_t ChannelCount;
  );
  
  __dme(ChannelSessionList,
    Protocol_ChannelID_t ChannelID;        // Which channel this list is for
    uint16_t SessionCount;                 // Number of sessions/users in the channel
  );
  
  __dme(Channel_ScreenShare_View_InformationToViewSetFlag,
    Protocol_ChannelID_t ChannelID;
    ProtocolChannel::ScreenShare::ChannelFlag::_t Flag;
  );
  
  __dme(Channel_ScreenShare_View_InformationToViewMouseCoordinate,
    Protocol_ChannelID_t ChannelID;
    fan::vec2ui pos;
  );
  
  __dme(Channel_ScreenShare_Share_ApplyToHostMouseCoordinate,
    Protocol_ChannelID_t ChannelID;
    fan::vec2si pos;
  );
  
  __dme(Channel_ScreenShare_Share_ApplyToHostMouseMotion,
    Protocol_ChannelID_t ChannelID;
    fan::vec2si Motion;
  );
  
  __dme(Channel_ScreenShare_Share_ApplyToHostMouseButton,
    Protocol_ChannelID_t ChannelID;
    uint8_t key;
    bool state;
    fan::vec2si pos;
  );
  
  __dme(Channel_ScreenShare_Share_ApplyToHostKeyboard,
    Protocol_ChannelID_t ChannelID;
    uint16_t Scancode;
    uint8_t State;
  );

  __dme(Channel_ScreenShare_ViewToShare,
    Protocol_ChannelID_t ChannelID;
    uint16_t Flag;
  );
  __dme(Channel_ScreenShare_ShareToView,
    Protocol_ChannelID_t ChannelID;
    uint16_t Flag;
  );
}Protocol_S2C;

#pragma pack(push, 1)

struct ScreenShare_StreamHeader_Body_t {
  uint8_t sc[3];

  void SetSequence(uint16_t Sequence) {
    *(uint16_t*)&sc[0] ^= *(uint16_t*)&sc[0] & 0xf0ff;
    *(uint16_t*)&sc[0] |= Sequence & 0x00ff | (Sequence & 0x0f00) << 4;
  }
  uint16_t GetSequence() {
    return *(uint16_t*)&sc[0] & 0x00ff | (*(uint16_t*)&sc[0] & 0xf000) >> 4;
  }
  void SetCurrent(uint16_t Current) {
    *(uint16_t*)&sc[1] ^= *(uint16_t*)&sc[1] & 0xff0f;
    *(uint16_t*)&sc[1] |= Current & 0x000f | (Current & 0x0ff0) << 4;
  }
  uint16_t GetCurrent() {
    return *(uint16_t*)&sc[1] & 0x000f | (*(uint16_t*)&sc[1] & 0xff00) >> 4;
  }
};
struct ScreenShare_StreamHeader_Head_t{
  ScreenShare_StreamHeader_Body_t Body;
  uint8_t pf[2];
  #if set_VerboseProtocol_HoldStreamTimes == 1
    struct _VerboseTime_t{
      uint64_t ScreenRead = 0;
      uint64_t SourceOptimize = 0;
      uint64_t Encode = 0;
      uint64_t WriteQueue = 0;
      uint64_t ThreadFrameEnd = 0;
      uint64_t NetworkWrite = 0;
    }_VerboseTime;
  #endif

  void SetPossible(uint16_t Possible){
    *(uint16_t *)&pf[0] ^= *(uint16_t *)&pf[0] & 0xf0ff;
    *(uint16_t *)&pf[0] |= Possible & 0x00ff | (Possible & 0x0f00) << 4;
  }
  uint16_t GetPossible(){
    return *(uint16_t *)&pf[0] & 0x00ff | (*(uint16_t *)&pf[0] & 0xf000) >> 4;
  }
  void SetFlag(uint8_t Flag){
    *(uint8_t *)&pf[1] ^= *(uint16_t *)&pf[2] & 0x0f;
    *(uint8_t *)&pf[1] |= Flag & 0x0f;
  }
  uint8_t GetFlag(){
    return *(uint8_t *)&pf[1] & 0x0f;
  }
};

#pragma pack(pop)