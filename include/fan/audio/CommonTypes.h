namespace _constants {
  const uint32_t opus_decode_sample_rate = 48000;

  const f32_t OneSampleTime = (f32_t)1 / opus_decode_sample_rate;

  const uint32_t CallFrameCount = 480;
  const uint32_t ChannelAmount = 2;
  const uint32_t FrameCacheAmount = 4800;
  const uint64_t FrameCacheTime = opus_decode_sample_rate / CallFrameCount * 1; // 1 second
}

struct _OpusHolder_t {
  OggOpusFile* decoder;
  const OpusHead* head;
};

struct piece_t;

#define BLL_set_prefix _FrameCacheList
#define BLL_set_type_node uint32_t
#define BLL_set_node_data \
  uint64_t LastAccessTime; \
  f32_t Frames[_constants::FrameCacheAmount][_constants::ChannelAmount]; \
  piece_t *piece; \
  uint32_t PieceCacheIndex;
#define BLL_set_ResizeListAfterClear 1
#include _WITCH_PATH(BLL/BLL.h)

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

struct _PieceCache_t {
  _FrameCacheList_NodeReference_t ref;
};

struct _audio_common_t{
  TH_mutex_t PlayInfoListMutex;
  _PlayInfoList_t PlayInfoList;

  uint32_t GroupAmount;
  _Group_t* GroupList;

  VEC_t PlayList;

  TH_mutex_t MessageQueueListMutex;
  VEC_t MessageQueueList;

  _FrameCacheList_t FrameCacheList;

  uint64_t Tick;
};

struct piece_t {
  _audio_common_t *audio_common;
  _OpusHolder_t holder;
  uint64_t raw_size;
  _PieceCache_t *Cache;
};

/* will used for userspace functions. casted to _audio_common_t internally */
struct audio_t;
