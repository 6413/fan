#pragma pack(push, 1)

struct Protocol_SessionID_t {
  typedef uint32_t Type;
  Type i;
  operator Type&() { return i; }
  Protocol_SessionID_t() = default;
  Protocol_SessionID_t(Type v) : i(v) {}
};
struct Protocol_AccountID_t {
  typedef uint32_t Type;
  Type i;
  operator Type& () { return i; }
  Protocol_AccountID_t() = default;
  Protocol_AccountID_t(Type v) : i(v) {}
};
struct Protocol_ChannelID_t {
  typedef uint16_t Type;
  Type i;
  operator Type& () { return i; }
  Protocol_ChannelID_t() = default;
  Protocol_ChannelID_t(Type v) : i(v) {}
};
struct Protocol_ChannelSessionID_t {
  typedef uint32_t Type;
  Type i;
  operator Type& () { return i; }
  Protocol_ChannelSessionID_t() = default;
  Protocol_ChannelSessionID_t(Type v) : i(v) {}
};
struct Protocol_SessionChannelID_t {
  typedef Protocol_ChannelID_t::Type Type;
  Type i;
  operator Type& () { return i; }
  Protocol_SessionChannelID_t() = default;
  Protocol_SessionChannelID_t(Type v) : i(v) {}
};

typedef uint16_t Protocol_CI_t;

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
}Protocol_C2S;

using S2C_callback_inherit_t = std::function<
  fan::event::task_t(ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base)
>;
struct S2C_callback_t : S2C_callback_inherit_t {
  using S2C_callback_inherit_t::S2C_callback_inherit_t;
  S2C_callback_t() : S2C_callback_inherit_t(
    [](ecps_backend_t& backend, const tcp::ProtocolBasePacket_t& base) -> fan::event::task_t {
      fan::print("unhandled callback");
      co_return;
    }
  ) { };
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