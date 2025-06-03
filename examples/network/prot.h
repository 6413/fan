#pragma pack(push, 1)

struct Protocol_AccountID_t {
  uint32_t i;
};
struct Protocol_SessionID_t {
  uint32_t i;
};

struct tcp {
  struct ProtocolBasePacket_t {
    uint32_t ID;
    uint16_t Command;
  };
};

struct udp {
  struct BasePacket_t {
    Protocol_SessionID_t SessionID;
    uint32_t ID;
    uint64_t IdentifySecret;
    uint16_t Command;
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

struct Protocol_C2S_t : __dme_inherit(Protocol_C2S_t){
  __dme(KeepAlive);
  __dme(Request_Login,
    Protocol::LoginType_t Type;
  );
  __dme(CreateChannel,
    uint8_t Type;
  );
  __dme(JoinChannel,
    uint16_t ChannelID;
  );
  __dme(QuitChannel,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
  );
  __dme(Response_UDPIdentifySecret,
    uint64_t UDPIdentifySecret;
  );
  __dme(Channel_ScreenShare_Share_InformationToViewSetFlag,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
    ProtocolChannel::ScreenShare::ChannelFlag::_t Flag;
  );
  __dme(Channel_ScreenShare_Share_InformationToViewMouseCoordinate,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
    fan::vec2ui pos;
  );
  __dme(Channel_ScreenShare_View_ApplyToHostMouseCoordinate,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
    fan::vec2si pos;
  );
  __dme(Channel_ScreenShare_View_ApplyToHostMouseMotion,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
    fan::vec2si Motion;
  );
  __dme(Channel_ScreenShare_View_ApplyToHostMouseButton,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
    uint8_t key;
    bool state;
    fan::vec2si pos;
  );
  __dme(Channel_ScreenShare_View_ApplyToHostKeyboard,
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
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

struct Protocol_S2C_t : __dme_inherit(Protocol_S2C_t, S2C_callback_t){
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
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
  );
  __dme(CreateChannel_Error,
    Protocol::JoinChannel_Error_Reason_t Reason;
  );
  __dme(JoinChannel_OK,
    uint8_t Type;
    uint16_t ChannelID;
    uint32_t ChannelSessionID;
  );
  __dme(JoinChannel_Error,
    Protocol::JoinChannel_Error_Reason_t Reason;
  );
  __dme(KickedFromChannel,
    uint16_t ChannelID;
    Protocol::KickedFromChannel_Reason_t Reason;
  );
  __dme(Request_UDPIdentifySecret);
  __dme(UseThisUDPIdentifySecret,
    uint64_t UDPIdentifySecret;
  );
  __dme(Channel_ScreenShare_View_InformationToViewSetFlag,
    uint16_t ChannelID;
    ProtocolChannel::ScreenShare::ChannelFlag::_t Flag;
  );
  __dme(Channel_ScreenShare_View_InformationToViewMouseCoordinate,
    uint16_t ChannelID;
    fan::vec2ui pos;
  );
  __dme(Channel_ScreenShare_Share_ApplyToHostMouseCoordinate,
    uint16_t ChannelID;
    fan::vec2si pos;
  );
  __dme(Channel_ScreenShare_Share_ApplyToHostMouseMotion,
    uint16_t ChannelID;
    fan::vec2si Motion;
  );
  __dme(Channel_ScreenShare_Share_ApplyToHostMouseButton,
    uint16_t ChannelID;
    uint8_t key;
    bool state;
    fan::vec2si pos;
  );
  __dme(Channel_ScreenShare_Share_ApplyToHostKeyboard,
    uint16_t ChannelID;
    uint16_t Scancode;
    uint8_t State;
  );
}Protocol_S2C;