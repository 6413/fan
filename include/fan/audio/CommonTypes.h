namespace _constants {
  const uint32_t opus_decode_sample_rate = 48000;
  namespace Opus{
    const uint32_t SegmentFrameAmount20 = 960;
    const uint32_t SupportedChannels = 2;
    const uint32_t CacheDecoderPerChannel = 0x08;
    const uint32_t CacheSegmentAmount = 0x400;
    const uint32_t DecoderWarmUpAmount = 0x04;
  }

  const f32_t OneSampleTime = (f32_t)1 / opus_decode_sample_rate;

  const uint32_t CallFrameCount = 480;
  const uint32_t ChannelAmount = 2;
  const uint32_t FrameCacheAmount = Opus::SegmentFrameAmount20;
  const uint64_t FrameCacheTime = opus_decode_sample_rate / CallFrameCount * 1; // 1 second
}

struct piece_t;

typedef uint8_t _DecoderID_t;
typedef uint32_t _SegmentID_t;
typedef uint16_t _CacheID_t;

struct _DecoderHead_t{
  _CacheID_t CacheID;
};

#define BLL_set_prefix _DecoderList
#define BLL_set_type_node _DecoderID_t
#include _WITCH_PATH(BLL/BLL.h)

#define BLL_set_prefix _CacheList
#define BLL_set_type_node _CacheID_t
#define BLL_set_node_data \
  f32_t Samples[_constants::FrameCacheAmount * _constants::ChannelAmount]; \
  _DecoderID_t DecoderID; \
  piece_t *piece; \
  _SegmentID_t SegmentID;
#include _WITCH_PATH(BLL/BLL.h)

#pragma pack(push, 1)

struct _SACHead_t{
  uint8_t Sign;
  uint16_t Checksum;
  uint8_t ChannelAmount;
  uint16_t BeginCut;
  uint16_t EndCut;
  uint32_t TotalSegments;
};

struct _SACSegment_t{
  uint32_t Offset;
  uint16_t Size;
  _CacheID_t CacheID;
};

#pragma pack(pop)

struct PropertiesSoundPlay_t {
  struct {
    uint32_t Loop : 1 = false;
    uint32_t FadeIn : 1 = false;
    uint32_t FadeOut : 1 = false;
  }Flags;
  f32_t FadeFrom;
  f32_t FadeTo;
};
struct PropertiesSoundStop_t {
  f32_t FadeOutTo = 0;
};

#define BLL_set_prefix _PlayInfoList
#define BLL_set_type_node uint32_t
#define BLL_set_node_data \
  piece_t *piece; \
  uint32_t GroupID; \
  uint32_t PlayID; \
  PropertiesSoundPlay_t properties; \
  uint64_t offset;
#define BLL_set_ResizeListAfterClear 1
#include _WITCH_PATH(BLL/BLL.h)

enum class _MessageType_t {
  SoundPlay,
  SoundStop,
  PauseGroup,
  ResumeGroup,
  StopGroup
};
struct _Message_t {
  _MessageType_t Type;
  union {
    struct {
      uint32_t PlayInfoReference;
    }SoundPlay;
    struct {
      uint32_t PlayInfoReference;
      PropertiesSoundStop_t Properties;
    }SoundStop;
    struct {
      uint32_t GroupID;
    }PauseGroup;
    struct {
      uint32_t GroupID;
    }ResumeGroup;
    struct {
      uint32_t GroupID;
    }StopGroup;
  }Data;
};

struct _Group_t {
  _PlayInfoList_NodeReference_t FirstReference;
  _PlayInfoList_NodeReference_t LastReference;
};
struct _Play_t {
  _PlayInfoList_NodeReference_t Reference;
};

struct _audio_common_t{
  TH_mutex_t PlayInfoListMutex;
  _PlayInfoList_t PlayInfoList;

  uint32_t GroupAmount;
  _Group_t* GroupList;

  VEC_t PlayList;

  TH_mutex_t MessageQueueListMutex;
  VEC_t MessageQueueList;

  _DecoderList_t DecoderList[_constants::Opus::SupportedChannels];

  _CacheList_t CacheList;
};

struct piece_t {
  _audio_common_t *audio_common;
  uint8_t ChannelAmount;
  uint16_t BeginCut;
  uint32_t TotalSegments;
  uint8_t *SACData;
  uint64_t FrameAmount;

  uint64_t GetFrameAmount(){
    return FrameAmount - BeginCut;
  }
};

/* will used for userspace functions. casted to _audio_common_t internally */
struct audio_t;
