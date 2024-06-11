#ifndef shaper_set_MaxShapeTypes
#define shaper_set_MaxShapeTypes 0x80
#endif

#ifndef shaper_set_MaxKeyInPack
#define shaper_set_MaxKeyInPack 0x80
#endif

/* in bytes */
#ifndef shaper_set_MaxKeySize
#define shaper_set_MaxKeySize 8
#endif

#ifndef shaper_set_MaxKeyType
#define shaper_set_MaxKeyType 0xff
#endif

#ifndef shaper_set_MaxKeyPackAmount
#define shaper_set_MaxKeyPackAmount 0xff
#endif

#ifdef shaper_set_MaxElementPerBlock
#error did you meant to define shaper_set_MaxMaxElementPerBlock
#endif
#ifndef shaper_set_MaxMaxElementPerBlock
#define shaper_set_MaxMaxElementPerBlock 0x8000
#endif

/* in bytes */
#ifdef shaper_set_MaxKeySizesSum
#define shaper_set_MaxKeySizesSum 0xff
#endif

#ifndef shaper_set_MaxShapeRenderDataSize
#define shaper_set_MaxShapeRenderDataSize 0xffff
#endif
#ifndef shaper_set_MaxShapeDataSize
#define shaper_set_MaxShapeDataSize 0xffff
#endif

#ifndef shaper_set_RenderDataOffsetType
#define shaper_set_RenderDataOffsetType uint32_t
#endif

#ifndef shaper_set_fan
#define shaper_set_fan 0
#endif
#ifdef gloco
#error use cpp not c
#endif
#define gloco gloco_is_not_allowed

struct shaper_t {
public: /* -------------------------------------------------------------------------------- */

#if shaper_set_MaxShapeTypes <= 0xff
  typedef uint8_t ShapeTypeAmount_t;
#elif shaper_set_MaxShapeTypes <= 0xffff
  typedef uint16_t ShapeTypeAmount_t;
#elif shaper_set_MaxShapeTypes <= 0xffffffff
  typedef uint32_t ShapeTypeAmount_t;
#else
#error ?
#endif
#if shaper_set_MaxShapeTypes <= 0x100
  typedef uint8_t ShapeTypeIndex_t;
#elif shaper_set_MaxShapeTypes <= 0x10000
  typedef uint16_t ShapeTypeIndex_t;
#elif shaper_set_MaxShapeTypes <= 0x100000000
  typedef uint32_t ShapeTypeIndex_t;
#else
#error ?
#endif

#if shaper_set_MaxKeyInPack <= 0xff
  typedef uint8_t KeyAmountInPack_t;
#else
#error ?
#endif
#if shaper_set_MaxKeyInPack <= 0x100
  typedef uint8_t KeyIndexInPack_t;
#else
#error ?
#endif

#if shaper_set_MaxMaxElementPerBlock <= 0xff
  typedef uint8_t MaxElementPerBlock_t;
#elif shaper_set_MaxMaxElementPerBlock <= 0xffff
  typedef uint16_t MaxElementPerBlock_t;
#elif shaper_set_MaxMaxElementPerBlock <= 0xffffffff
  typedef uint32_t MaxElementPerBlock_t;
#else
#error ?
#endif
#if shaper_set_MaxMaxElementPerBlock <= 0x100
  typedef uint8_t ElementIndexInBlock_t;
#elif shaper_set_MaxMaxElementPerBlock <= 0x10000
  typedef uint16_t ElementIndexInBlock_t;
#elif shaper_set_MaxMaxElementPerBlock <= 0x100000000
  typedef uint32_t ElementIndexInBlock_t;
#else
#error ?
#endif

  typedef uint8_t KeyData_t;

#if shaper_set_MaxKeySize <= 0xff
  typedef uint8_t KeySizeInBytes_t;
#else
#error ?
#endif
#if shaper_set_MaxKeySize * 8 <= 0xff
  typedef uint8_t KeySizeInBits_t;
#elif shaper_set_MaxKeySize * 8 <= 0xffff
  typedef uint16_t KeySizeInBits_t;
#else
#error ?
#endif

#if shaper_set_MaxKeyType <= 0xff
  typedef uint8_t KeyTypeAmount_t;
#else
#error ?
#endif
#if shaper_set_MaxKeyType <= 0x100
  typedef uint8_t KeyTypeIndex_t;
#else
#error ?
#endif

#if shaper_set_MaxKeyPackAmount <= 0xff
  typedef uint8_t KeyPackAmount_t;
#else
#error ?
#endif
#if shaper_set_MaxKeyPackAmount <= 0x100
  typedef uint8_t KeyPackIndex_t;
#else
#error ?
#endif

#if shaper_set_MaxKeySizesSum <= 0xff
  typedef uint8_t KeySizesSumInBytes_t;
#else
#error ?
#endif

#if shaper_set_MaxShapeRenderDataSize <= 0xff
  typedef uint8_t ShapeRenderDataSize_t;
#elif shaper_set_MaxShapeRenderDataSize <= 0xffff
  typedef uint16_t ShapeRenderDataSize_t;
#elif shaper_set_MaxShapeRenderDataSize <= 0xffffffff
  typedef uint32_t ShapeRenderDataSize_t;
#else
#error ?
#endif

#if shaper_set_MaxShapeDataSize <= 0xff
  typedef uint8_t ShapeDataSize_t;
#elif shaper_set_MaxShapeDataSize <= 0xffff
  typedef uint16_t ShapeDataSize_t;
#elif shaper_set_MaxShapeDataSize <= 0xffffffff
  typedef uint32_t ShapeDataSize_t;
#else
#error ?
#endif

  typedef uint8_t ShapeRenderData_t;
  typedef uint8_t ShapeData_t;

private: /* ------------------------------------------------------------------------------- */

  /* key tree and block manager node reference type */
  typedef uint16_t ktbmnr_t;

  /* block list node reference */
  /*
    total element a ShapeType can have is =
      (pow(2, (sizeof(blid_t) * 8)) * MaxElementPerBlock)
  */
  typedef uint16_t blid_t;

#define BDBT_set_prefix KeyTree
#define BDBT_set_type_node ktbmnr_t
#define BDBT_set_lcpp
#define BDBT_set_KeySize 0
#ifdef shaper_set_MaxKeySize
#define BDBT_set_MaxKeySize (shaper_set_MaxKeySize * 8)
#endif
#define BDBT_set_AreWeInsideStruct 1
#include <BDBT/BDBT.h>
  KeyTree_t KeyTree;
  typedef KeyTree_Key_t Key_t;

public: /* -------------------------------------------------------------------------------- */

  typedef Key_t::BitOrder_t KeyBitOrder_t;
  constexpr static KeyBitOrder_t KeyBitOrderLow = Key_t::BitOrderLow;
  constexpr static KeyBitOrder_t KeyBitOrderHigh = Key_t::BitOrderHigh;
  constexpr static KeyBitOrder_t KeyBitOrderAny = Key_t::BitOrderAny;

private: /* ------------------------------------------------------------------------------- */

  struct KeyType_t {
    KeySizeInBytes_t Size;
    KeyBitOrder_t BitOrder;

    /* size in bits */
    KeySizeInBits_t sibit() {
      return (KeySizeInBits_t)Size * 8;
    }
  };
  KeyType_t* KeyTypes;
  KeyTypeAmount_t KeyTypeAmount;

#define BLL_set_prefix BlockList
#define BLL_set_BufferUpdateInfo \
    auto st = OFFSETLESS(bll, ShapeType_t, BlockList); \
    st->shaper->_BlockListBufferChange(st->sti, New);
#define BLL_set_Link 1
#define BLL_set_LinkSentinel 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_type_node blid_t
#define BLL_set_Mark 1
#include <BLL/BLL.h>
  struct ShapeType_t {
    /* this will be used from BlockList callbacks with offsetless */

    ShapeTypeIndex_t sti;

    shaper_t* shaper;

    /*
      (RenderDataSize + DataSize + sizeof(ShapeList_t::nr_t)) * (MaxElementPerBlock_m1 + 1) +
      sizeof(BlockUnique_t)
    */
    BlockList_t BlockList;

    KeyPackIndex_t KeyPackIndex;

    ElementIndexInBlock_t MaxElementPerBlock_m1; /* minus 1, (p.MaxElementPerBlock[n] - 1) */
    ShapeRenderDataSize_t RenderDataSize;
    ShapeDataSize_t DataSize;

#if shaper_set_fan
    fan::opengl::core::vao_t m_vao;
    fan::opengl::core::vbo_t m_vbo;

    struct a_t {
      a_t() {
        x = 5;
      }
      int x;
    }a;

    std::vector<shape_gl_init_t> locations;
    struct b_t {
      b_t() {
        y = 5;
      }
      int y;
    }b;
    fan::opengl::context_t::shader_nr_t shader;
#endif

    MaxElementPerBlock_t MaxElementPerBlock() {
      return (MaxElementPerBlock_t)MaxElementPerBlock_m1 + 1;
    }
  };
#define BLL_set_prefix ShapeTypes
#define BLL_set_Link 0
#define BLL_set_Recycle 0
#define BLL_set_IntegerNR 1
#define BLL_set_CPP_Node_ConstructDestruct 1
#define BLL_set_CPP_CopyAtPointerChange 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeDataType ShapeType_t
#define BLL_set_type_node ShapeTypeAmount_t
#include <BLL/BLL.h>
  ShapeTypes_t ShapeTypes;

#pragma pack(push, 1)
  struct bm_BaseData_t {
    ShapeTypeAmount_t sti;
    BlockList_NodeReference_t FirstBlockNR;
    BlockList_NodeReference_t LastBlockNR;
    ElementIndexInBlock_t LastBlockElementCount;
  };
#pragma pack(pop)

#define BLL_set_prefix bm
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_type_node ktbmnr_t
#include <BLL/BLL.h>
  /* sizeof(bm_BaseData_t) + KeySizesSum */

#pragma pack(push, 1)
  struct shape_t {
    ShapeTypeIndex_t sti;
    bm_t::nr_t bmid;
    BlockList_t::nr_t blid;
    ElementIndexInBlock_t ElementIndex;
  };
#pragma pack(pop)

#define BLL_set_prefix ShapeList
#define BLL_set_Link 0
#define BLL_set_NodeDataType shape_t
#define BLL_set_AreWeInsideStruct 1
  // actually it needs to be uint24_t
#define BLL_set_type_node uint32_t
#include <BLL/BLL.h>
  ShapeList_t ShapeList;

  struct KeyPack_t {
    KeyAmountInPack_t KeyAmount;
    KeyTypeIndex_t* KeyIndexes;
    KeySizesSumInBytes_t KeySizesSum;
    KeyTree_NodeReference_t KeyTree_root;
    bm_t bm;
  };
  KeyPack_t* KeyPacks;
  KeyPackAmount_t KeyPackAmount;

  /* function internal datas */
  struct fid_t {
    union {
      void* base;
      /* used in remove */
      KeyTree_NodeReference_t* knrs;
      /* used in traverse for store keys */
      KeyData_t* KeyDatas;
    };

    Key_t::Traverse_t* tra;

    void Open(const shaper_t* shaper) {
      KeyAmountInPack_t MaxKeyAmount = 0;
      KeySizesSumInBytes_t MaxKeySizesSum = 0;
      for (KeyPackIndex_t kpi = 0; kpi < shaper->KeyPackAmount; kpi++) {
        if (MaxKeyAmount < shaper->KeyPacks[kpi].KeyAmount) {
          MaxKeyAmount = shaper->KeyPacks[kpi].KeyAmount;
        }
        if (MaxKeySizesSum < shaper->KeyPacks[kpi].KeySizesSum) {
          MaxKeySizesSum = shaper->KeyPacks[kpi].KeySizesSum;
        }
      }
      base = A_resize(NULL,
        sizeof(KeyTree_NodeReference_t) * MaxKeyAmount > MaxKeySizesSum ?
        sizeof(KeyTree_NodeReference_t) * MaxKeyAmount :
        MaxKeySizesSum
      );

      tra = (Key_t::Traverse_t*)A_resize(NULL, MaxKeyAmount * sizeof(Key_t::Traverse_t));
    }
    void Close(const shaper_t* shaper) {
      A_resize(tra, 0);

      A_resize(base, 0);
    }
  }fid;

  /*
    block data format is this. %30
    RenderData   Data         ShapeID
    [|||       ] [|||       ] [|||       ]
  */

  /* TODO those are made uintptr_t to not overflow, maybe there is better way? */
  ShapeRenderData_t* _GetRenderData(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ) {
    return &((ShapeRenderData_t*)ShapeTypes[sti].BlockList[blid])[
      (uintptr_t)ShapeTypes[sti].RenderDataSize * ElementIndex
    ];
  }
public:
  ShapeData_t* _GetData(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ) {
    return &_GetRenderData(
      sti,
      blid,
      ShapeTypes[sti].MaxElementPerBlock()
    )[(uintptr_t)ShapeTypes[sti].DataSize * ElementIndex];
  }
  ShapeList_t::nr_t& _GetShapeID(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ) {
    return *(ShapeList_t::nr_t*)&_GetData(
      sti,
      blid,
      ShapeTypes[sti].MaxElementPerBlock()
    )[sizeof(ShapeList_t::nr_t) * ElementIndex];
  }
private:

#define BLL_set_prefix BlockEditQueue
#define BLL_set_Link 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeData \
    ShapeTypeIndex_t sti; \
    BlockList_t::nr_t blid;
#define BLL_set_type_node uint16_t
#include <BLL/BLL.h>
  BlockEditQueue_t BlockEditQueue;

  struct BlockUnique_t {
    shaper_set_RenderDataOffsetType MinEdit;
    shaper_set_RenderDataOffsetType MaxEdit;
    BlockEditQueue_t::nr_t beid{ true };

    void clear() {
      beid.sic();
      MinEdit = (decltype(MinEdit))-1;
      MaxEdit = 0;
    }
    void constructor() {
      clear();
    }
    void destructor(shaper_t& shaper) {
      if (!beid.iic()) {
        shaper.BlockEditQueue.unlrec(beid);
      }
    }
  };

public: /* -------------------------------------------------------------------------------- */

  using BlockID_t = BlockList_t::nr_t;
  using bmid_t = bm_t::nr_t;
  using ShapeID_t = ShapeList_t::nr_t;

#if shaper_set_fan
  fan::opengl::context_t::shader_nr_t GetShader(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].shader;
  }
  fan::opengl::core::vao_t GetVAO(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].m_vao;
  }
  fan::opengl::core::vbo_t GetVBO(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].m_vbo;
  }
  std::vector<shape_gl_init_t>& GetLocations(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].locations;
  }
#endif

  ShapeTypeIndex_t GetSTI(ShapeID_t ShapeID) {
    return ShapeList[ShapeID].sti;
  }
  BlockID_t GetBLID(ShapeID_t ShapeID) {
    return ShapeList[ShapeID].blid;
  }
  ElementIndexInBlock_t GetElementIndex(ShapeID_t ShapeID) {
    return ShapeList[ShapeID].ElementIndex;
  }
  KeyPackIndex_t GetKPI(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].KeyPackIndex;
  }

  KeySizesSumInBytes_t GetKeySizesSum(ShapeTypeIndex_t sti) {
    auto& st = ShapeTypes[sti];
    auto& kp = KeyPacks[st.KeyPackIndex];
    return kp.KeySizesSum;
  }

  ShapeRenderDataSize_t GetRenderDataSize(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].RenderDataSize;
  }
  //void *GetRenderData(shaper_t &shaper){
  //    return shaper._GetRenderData(sti, From, 0);
  //}
  //void *GetData(shaper_t &shaper){
  //  return shaper._GetData(sti, From, 0);
  //}
  ShapeDataSize_t GetDataSize(ShapeTypeIndex_t sti) {
    return ShapeTypes[sti].DataSize;
  }

  void* GetKeys(ShapeID_t ShapeID) {
    auto& s = ShapeList[ShapeID];
    auto& st = ShapeTypes[s.sti];
    auto& kp = KeyPacks[st.KeyPackIndex];
    bm_BaseData_t* bmbase = (bm_BaseData_t*)kp.bm[s.bmid];
    return &bmbase[1];
  }
  void WriteKeys(ShapeID_t ShapeID, void* dst) {
    auto& s = ShapeList[ShapeID];
    auto& st = ShapeTypes[s.sti];
    auto& kp = KeyPacks[st.KeyPackIndex];
    bm_BaseData_t* bmbase = (bm_BaseData_t*)kp.bm[s.bmid];
    __MemoryCopy(&bmbase[1], dst, kp.KeySizesSum);
  }
  ShapeRenderData_t* GetRenderData(
    ShapeID_t ShapeID
  ) {
    auto& s = ShapeList[ShapeID];
    return _GetRenderData(s.sti, s.blid, s.ElementIndex);
  }
  ShapeData_t* GetData(
    ShapeID_t ShapeID
  ) {
    auto& s = ShapeList[ShapeID];
    return _GetData(s.sti, s.blid, s.ElementIndex);
  }
  BlockUnique_t& GetBlockUnique(
    ShapeTypeIndex_t sti,
    BlockID_t BlockID
  ) {
    return *(BlockUnique_t*)&_GetShapeID(
      sti,
      BlockID,
      ShapeTypes[sti].MaxElementPerBlock()
    );
  }

  shaper_set_RenderDataOffsetType GetRenderDataOffset(
    ShapeTypeIndex_t sti,
    BlockID_t blid
  ) {
    auto& st = ShapeTypes[sti];
    return (shaper_set_RenderDataOffsetType)blid.gint() *
      st.MaxElementPerBlock() *
      st.RenderDataSize;
  }

  struct BlockProperties_t {
    MaxElementPerBlock_t MaxElementPerBlock;
    decltype(ShapeType_t::RenderDataSize) RenderDataSize;
    decltype(ShapeType_t::DataSize) DataSize;

#if shaper_set_fan
    std::vector<shape_gl_init_t> locations;
    fan::opengl::context_t::shader_nr_t shader;
#endif
  };

  void Open() {
    KeyTypeAmount = 0;
    KeyTypes = NULL;

    KeyPackAmount = 0;
    KeyPacks = NULL;

    ShapeTypes.Open();

    KeyTree.Open();
    BlockEditQueue.Open();
    ShapeList.Open();

    fid.Open(this);
  }
  void Close() {
    fid.Close(this);

    ShapeList.Close();
    BlockEditQueue.Close();
    KeyTree.Close();

    for (auto& st : ShapeTypes) {
      st.BlockList.Close();
    }
    ShapeTypes.Close();

    for (KeyPackIndex_t kpi = 0; kpi < KeyPackAmount; kpi++) {
      A_resize(KeyPacks[kpi].KeyIndexes, 0);
      KeyPacks[kpi].bm.Close();
    }
    A_resize(KeyPacks, 0);

    A_resize(KeyTypes, 0);
  }

  void AddKey(KeyTypeIndex_t KeyTypeIndex, KeySizeInBytes_t Size, KeyBitOrder_t BitOrder) {
    if (KeyTypeIndex >= KeyTypeAmount) {
      KeyTypeAmount = KeyTypeIndex;
      KeyTypeAmount++;
      KeyTypes = (KeyType_t*)A_resize(
        KeyTypes,
        (uintptr_t)KeyTypeAmount * sizeof(KeyType_t)
      );
    }

    KeyTypes[KeyTypeIndex].Size = Size;
    KeyTypes[KeyTypeIndex].BitOrder = BitOrder;
  }

  void SetKeyOrder(KeyTypeIndex_t KeyTypeIndex, KeyBitOrder_t BitOrder) {
    KeyTypes[KeyTypeIndex].BitOrder = BitOrder;
  }

  /* fid struct requires different size for different key packs */
  /* because of it, fid gets closed and opened in this function */
  void AddKeyPack(
    KeyPackIndex_t kpi,
    KeyAmountInPack_t KeyAmount,
    KeyTypeIndex_t* KeyIndexes
  ) {
    fid.Close(this);

    if (kpi >= KeyPackAmount) {
      KeyPacks = (KeyPack_t*)A_resize(
        KeyPacks,
        ((uintptr_t)kpi + 1) * sizeof(KeyPack_t)
      );
      for (; KeyPackAmount < (KeyPackAmount_t)kpi + 1; KeyPackAmount++) { /* filler open */
        KeyPacks[KeyPackAmount].KeyAmount = 0;
        KeyPacks[KeyPackAmount].KeyIndexes = NULL;
        KeyPacks[KeyPackAmount].KeySizesSum = 0;
        KeyPacks[KeyPackAmount].KeyTree_root = KeyTree.NewNode();
        KeyPacks[KeyPackAmount].bm.Open(1);
      }
    }

    KeyPacks[kpi].KeyAmount = KeyAmount;

    KeyPacks[kpi].KeyIndexes = A_resize(
      KeyPacks[kpi].KeyIndexes,
      (uintptr_t)KeyAmount * sizeof(KeyTypeIndex_t)
    );
    __MemoryCopy(
      KeyIndexes,
      KeyPacks[kpi].KeyIndexes,
      (uintptr_t)KeyAmount * sizeof(KeyTypeIndex_t)
    );

    KeyPacks[kpi].KeySizesSum = 0;
    for (uintptr_t i = 0; i < KeyAmount; i++) {
      KeyPacks[kpi].KeySizesSum += KeyTypes[KeyIndexes[i]].Size;
    }

    KeyPacks[kpi].bm.Close();
    KeyPacks[kpi].bm.Open(sizeof(bm_BaseData_t) + KeyPacks[kpi].KeySizesSum);

    fid.Open(this);
  }


  loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, shaper);
  }

  void AddShapeType(
    ShapeTypeIndex_t sti,
    KeyPackIndex_t kpi,
    const BlockProperties_t bp
  ) {
    while (sti >= ShapeTypes.Usage()) {
      auto csti = ShapeTypes.NewNode();
      auto& st = ShapeTypes[csti];

      /* filler init */

      st.sti = csti;
      st.shaper = this;
      st.BlockList.Open(1);
    }

    auto& st = ShapeTypes[sti];

    st.BlockList.Close();
    st.BlockList.Open(
      (
        (uintptr_t)bp.RenderDataSize + bp.DataSize + sizeof(ShapeList_t::nr_t)
        ) * (bp.MaxElementPerBlock) + sizeof(BlockUnique_t)
    );

    st.KeyPackIndex = kpi;

    st.MaxElementPerBlock_m1 = bp.MaxElementPerBlock - 1;
    st.RenderDataSize = bp.RenderDataSize;
    st.DataSize = bp.DataSize;

#if shaper_set_fan
    auto& context = get_loco()->get_context();

    st.m_vao.open(context);
    st.m_vbo.open(context, fan::opengl::GL_ARRAY_BUFFER);
    st.m_vao.bind(context);
    st.m_vbo.bind(context);
    st.shader = bp.shader;
    st.locations = bp.locations;
    uint64_t ptr_offset = 0;
    for (const auto& location : st.locations) {
      context.opengl.glEnableVertexAttribArray(location.index);
      context.opengl.glVertexAttribPointer(location.index, location.size, location.type, fan::opengl::GL_FALSE, location.stride, (void*)ptr_offset);
      context.opengl.glVertexAttribDivisor(location.index, 1);
      switch (location.type) {
      case fan::opengl::GL_FLOAT: {
        ptr_offset += location.size * sizeof(f32_t);
        break;
      }
      case fan::opengl::GL_UNSIGNED_INT: {
        ptr_offset += location.size * sizeof(fan::opengl::GLuint);
        break;
      }
      default: {
        fan::throw_error_impl();
      }
      }
    }
#endif
  }

  void ProcessBlockEditQueue() {
#if shaper_set_fan
    fan::opengl::context_t& context = get_loco()->get_context();
#endif

    auto beid = BlockEditQueue.GetNodeFirst();
    while (beid != BlockEditQueue.dst) {
      auto& be = BlockEditQueue[beid];
      auto& st = ShapeTypes[be.sti];
      auto& bu = GetBlockUnique(be.sti, be.blid);

#if shaper_set_fan

      st.m_vao.bind(context);
      fan::opengl::core::edit_glbuffer(
        get_loco()->get_context(),
        st.m_vbo.m_buffer,
        _GetRenderData(be.sti, be.blid, 0) + bu.MinEdit,
        GetRenderDataOffset(be.sti, be.blid) + bu.MinEdit,
        bu.MaxEdit - bu.MinEdit,
        fan::opengl::GL_ARRAY_BUFFER
      );
#endif

      bu.clear();

      beid = beid.Next(&BlockEditQueue);
    }
    BlockEditQueue.Close();
    BlockEditQueue.Open();
  }

  void ElementIsPartiallyEdited(
    ShapeTypeIndex_t sti,
    shaper_t::BlockList_t::nr_t blid,
    uint16_t eiib,
    uint32_t byte_start,
    uint32_t byte_count
  ) {
    ShapeType_t& st = ShapeTypes[sti];
    BlockUnique_t& bu = GetBlockUnique(sti, blid);

#if shaper_set_fan
    bu.MinEdit = std::min(
      bu.MinEdit,
      (shaper_set_RenderDataOffsetType)eiib * st.RenderDataSize + byte_start
    );
    bu.MaxEdit = std::max(
      bu.MaxEdit,
      (shaper_set_RenderDataOffsetType)eiib * st.RenderDataSize + byte_start + byte_count
    );
#endif

    if (!bu.beid.iic()) {
      return;
    }

    bu.beid = BlockEditQueue.NewNodeLast();
    auto& be = BlockEditQueue[bu.beid];
    be.sti = sti;
    be.blid = blid;
  }
  void ElementIsFullyEdited(
    ShapeTypeIndex_t sti,
    shaper_t::BlockList_t::nr_t blid,
    uint16_t eiib
  ) {
    auto& st = ShapeTypes[sti];
    ElementIsPartiallyEdited(sti, blid, eiib, 0, st.RenderDataSize);
  }

  void _RenderDataReset(ShapeTypeIndex_t sti) {
    auto& st = ShapeTypes[sti];

    /* TODO remove all block edit queue stuff */

    BlockList_t::nrtra_t traverse;
    traverse.Open(&st.BlockList);
#if shaper_set_fan
    st.m_vao.bind(get_loco()->get_context());
#endif
    while (traverse.Loop(&st.BlockList)) {
#if shaper_set_fan
      fan::opengl::core::edit_glbuffer(
        get_loco()->get_context(),
        st.m_vbo.m_buffer,
        _GetRenderData(sti, traverse.nr, 0),
        GetRenderDataOffset(sti, traverse.nr),
        st.RenderDataSize * st.MaxElementPerBlock(),
        fan::opengl::GL_ARRAY_BUFFER
      );
#endif
    }
    traverse.Close(&st.BlockList);
  }
  void _BlockListBufferChange(ShapeTypeIndex_t sti, uintptr_t New) {
    auto& st = ShapeTypes[sti];

#if shaper_set_fan
    st.m_vbo.bind(get_loco()->get_context());
    fan::opengl::core::write_glbuffer(
      get_loco()->get_context(),
      st.m_vbo.m_buffer,
      0,
      New * st.RenderDataSize * st.MaxElementPerBlock(),
      fan::opengl::GL_DYNAMIC_DRAW,
      fan::opengl::GL_ARRAY_BUFFER
    );
    _RenderDataReset(sti);
#endif
  }

  BlockList_t::nr_t _newblid(
    ShapeTypeIndex_t sti,
    bm_BaseData_t* bmbase
  ) {
    auto& st = ShapeTypes[sti];

    bmbase->LastBlockElementCount = 0;
    auto blid = st.BlockList.NewNode();
    bmbase->LastBlockNR = blid;

    GetBlockUnique(sti, blid).constructor();

    return blid;
  }
  void _deleteblid(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid
  ) {
    auto& st = ShapeTypes[sti];
    GetBlockUnique(sti, blid).destructor(*this);

    st.BlockList.Recycle(blid);
  }

  ShapeID_t add(
    ShapeTypeIndex_t sti,
    const void* KeyDataArray,
    const void* RenderData,
    const void* Data
  ) {
    bm_NodeReference_t bmid;
    bm_BaseData_t* bmbase;

    auto& st = ShapeTypes[sti];
    auto& kp = KeyPacks[st.KeyPackIndex];

    auto _KeyDataArray = (KeyData_t*)KeyDataArray;

    KeyTree_NodeReference_t nr = kp.KeyTree_root;
    for (KeyIndexInPack_t kiip = 0; kiip < kp.KeyAmount; kiip++) {
      auto kt = &KeyTypes[kp.KeyIndexes[kiip]];
      Key_t::KeySize_t bdbt_ki;
      Key_t::q(&KeyTree, kt->sibit(), _KeyDataArray, &bdbt_ki, &nr);
      if (bdbt_ki != kt->sibit()) {
        /* query failed to find rest so lets make new block manager */

        bmid = kp.bm.NewNode();
        bmbase = (bm_BaseData_t*)kp.bm[bmid];
        bmbase->sti = sti;
        bmbase->FirstBlockNR = _newblid(sti, bmbase);
        __MemoryCopy(KeyDataArray, &bmbase[1], kp.KeySizesSum);

        do {
          KeyTree_NodeReference_t out;
          if (kiip != kp.KeyAmount - 1) {
            out = KeyTree.NewNode();
          }
          else {
            out = *(KeyTree_NodeReference_t*)&bmid;
          }

          Key_t::a(&KeyTree, kt->sibit(), _KeyDataArray, bdbt_ki, nr, out);
          bdbt_ki = 0;

          nr = out;

          _KeyDataArray += kt->Size;

          if (++kiip == kp.KeyAmount) {
            break;
          }

          kt = &KeyTypes[kp.KeyIndexes[kiip]];
        } while (1);

        goto gt_NoNewBlockManager;
      }

      _KeyDataArray += kt->Size;
    }

    bmid = *(bm_NodeReference_t*)&nr;
    bmbase = (bm_BaseData_t*)kp.bm[bmid];

    if (bmbase->LastBlockElementCount == st.MaxElementPerBlock_m1) {
      auto lbnr = bmbase->LastBlockNR;
      auto newblid = _newblid(sti, bmbase);
      st.BlockList.linkNextOfOrphan(lbnr, newblid);
    }
    else {
      bmbase->LastBlockElementCount++;
    }

  gt_NoNewBlockManager:

    auto shapeid = ShapeList.NewNode();
    ShapeList[shapeid].sti = sti;
    ShapeList[shapeid].bmid = bmid;
    ShapeList[shapeid].blid = bmbase->LastBlockNR;
    ShapeList[shapeid].ElementIndex = bmbase->LastBlockElementCount;

    __MemoryCopy(
      RenderData,
      _GetRenderData(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount),
      st.RenderDataSize
    );
    __MemoryCopy(
      Data,
      _GetData(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount),
      st.DataSize
    );
    _GetShapeID(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount) = shapeid;

    ElementIsFullyEdited(sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount);

    return shapeid;
  }

  void remove(
    ShapeList_t::nr_t shapeid
  ) {
    ShapeTypeIndex_t sti;
    ShapeType_t* st;
    KeyPack_t* kp;
    bm_t::nr_t bmid;
    bm_BaseData_t* bmbase;
    BlockList_t::nr_t gblid;
    ElementIndexInBlock_t eiib;

    {
      auto& s = ShapeList[shapeid];
      sti = s.sti;
      st = &ShapeTypes[s.sti];
      kp = &KeyPacks[st->KeyPackIndex];
      bmid = s.bmid;
      bmbase = (bm_BaseData_t*)kp->bm[bmid];
      gblid = s.blid;
      eiib = s.ElementIndex;

      auto& lshape = ShapeList[
        _GetShapeID(s.sti, bmbase->LastBlockNR, bmbase->LastBlockElementCount)
      ];

      __MemoryMove(
        _GetRenderData(s.sti, s.blid, s.ElementIndex),
        _GetRenderData(s.sti, lshape.blid, lshape.ElementIndex),
        st->RenderDataSize
      );
      __MemoryMove(
        _GetData(s.sti, s.blid, s.ElementIndex),
        _GetData(s.sti, lshape.blid, lshape.ElementIndex),
        st->DataSize
      );
      _GetShapeID(s.sti, s.blid, s.ElementIndex) =
        _GetShapeID(s.sti, lshape.blid, lshape.ElementIndex);

      lshape.blid = s.blid;
      lshape.ElementIndex = s.ElementIndex;

      ShapeList.Recycle(shapeid);
    }

    /* shapeid we deleted may be same as last so lets no longer access shape plus lshape */

    /* we just deleted last so lets check if we can just decrease count */
    if (bmbase->LastBlockElementCount != 0) {
      bmbase->LastBlockElementCount--;
      ElementIsFullyEdited(sti, gblid, eiib);
      return;
    }

    /* looks like we deleted first element in block so we need to delete last block */
    /* before do that we need to be sure if that we dont delete first block in block manager */
    if (bmbase->FirstBlockNR != bmbase->LastBlockNR) {
      if (gblid != bmbase->LastBlockNR) {
        ElementIsFullyEdited(sti, gblid, eiib);
      }
      auto blid = bmbase->LastBlockNR;
      bmbase->LastBlockNR = blid.Prev(&st->BlockList);
      _deleteblid(sti, blid);
      bmbase->LastBlockElementCount = st->MaxElementPerBlock_m1;
      return;
    }


    /* good luck we need to remove block manager completely */
    /* aaaaaaaaaaaa */

    _deleteblid(sti, bmbase->LastBlockNR);

    auto KeyDataArray = (KeyData_t*)&bmbase[1];
    auto knr = kp->KeyTree_root;

    for (KeyIndexInPack_t kiip = 0; kiip < kp->KeyAmount; kiip++) {
      auto* kt = &KeyTypes[kp->KeyIndexes[kiip]];
      fid.knrs[kiip] = knr;
      Key_t::cq(&KeyTree, kt->sibit(), KeyDataArray, &knr);
      KeyDataArray += kt->Size;
    }

    /* TODO this part can be faster if used some different function instead of .r */
    for (KeyIndexInPack_t kiip = kp->KeyAmount - 1;;) {
      auto* kt = &KeyTypes[kp->KeyIndexes[kiip]];
      KeyDataArray -= kt->Size;
      Key_t::r(&KeyTree, kt->sibit(), KeyDataArray, fid.knrs[kiip]);
      if (KeyTree.inrhc(fid.knrs[kiip])) {
        break;
      }
      if (kiip == 0) {
        break;
      }
      KeyTree.Recycle(fid.knrs[kiip]);
      kiip--;
    }

    kp->bm.Recycle(bmid);
  }

  struct KeyPackTraverse_t {
    KeyPackIndex_t kpi;

    void Init(shaper_t& shaper) {
      kpi = (KeyPackIndex_t)-1;
    }
    bool Loop(shaper_t& shaper) {
      return ++kpi <= shaper.KeyPackAmount - 1;
    }
  };

  struct KeyTraverse_t {
  private: /* ----------------------------------------------------------------------------- */

    uint8_t State;
    KeyIndexInPack_t KeyIndexInPack;
    KeyPack_t* kp;
    KeyType_t* kt;

  public: /* ------------------------------------------------------------------------------ */

    KeyData_t* KeyData;

    void Init(
      shaper_t& shaper,
      const KeyPackIndex_t kpi
    ) {
      State = 0;
      kp = &shaper.KeyPacks[kpi];
    }
    bool Loop(
      shaper_t& shaper,
      KeyTypeIndex_t& kti
    ) {
    gt_reswitch:

      switch (State) {
      case 0: {
        KeyIndexInPack = 0;
        KeyData = shaper.fid.KeyDatas;

        if (!kp->KeyAmount) {
          /* TODO bmid */
          kti = (KeyTypeIndex_t)-1;
          State = 1;
          return true;
        }

        kt = &shaper.KeyTypes[kp->KeyIndexes[0]];

        shaper.fid.tra[0].i0(
          kp->KeyTree_root,
          kt->BitOrder
        );

        if (shaper.fid.tra[0].t0(
          &shaper.KeyTree,
          kt->sibit(),
          KeyData,
          kt->BitOrder
        ) == false) {
          return false;
        }

        kti = kp->KeyIndexes[0];

        State = 2;

        return true;
      }
      case 1: {
        return false;
      }
      case 2: {
        if (KeyIndexInPack == kp->KeyAmount - 1) {
          kti = (KeyTypeIndex_t)-1;
          State = 3;
          return true;
        }
        KeyData += shaper.KeyTypes[kp->KeyIndexes[KeyIndexInPack]].Size;
        KeyIndexInPack++;

        kt = &shaper.KeyTypes[kp->KeyIndexes[KeyIndexInPack]];

        shaper.fid.tra[KeyIndexInPack].i0(
          shaper.fid.tra[KeyIndexInPack - 1].Output,
          kt->BitOrder
        );

      gt_retra:

        if (shaper.fid.tra[KeyIndexInPack].t0(
          &shaper.KeyTree,
          kt->sibit(),
          KeyData,
          kt->BitOrder
        ) == false) {
          State = 4;
          goto gt_reswitch;
        }

        kti = kp->KeyIndexes[KeyIndexInPack];

        return true;
      }
      case 3: {
        if (shaper.fid.tra[KeyIndexInPack].t0(
          &shaper.KeyTree,
          kt->sibit(),
          KeyData,
          kt->BitOrder
        ) == false) {
          State = 4;
          goto gt_reswitch;
        }
        kti = kp->KeyIndexes[KeyIndexInPack];
        State = 5;
        return true;
      }
      case 4: {
        if (KeyIndexInPack == 0) {
          return false;
        }
        KeyIndexInPack--;
        KeyData -= shaper.KeyTypes[kp->KeyIndexes[KeyIndexInPack]].Size;
        kt = &shaper.KeyTypes[kp->KeyIndexes[KeyIndexInPack]];
        State = 2;
        goto gt_retra;
      }
      case 5: {
        kti = (KeyTypeIndex_t)-1;
        State = 3;
        return true;
      }
      }
      __unreachable();
    }
    bmid_t bmid(shaper_t& shaper) {
      return *(bmid_t*)&shaper.fid.tra[KeyIndexInPack].Output;
    }
  };

  struct BlockTraverse_t {
  private: /* ----------------------------------------------------------------------------- */

    ShapeTypeIndex_t sti;
    BlockList_t::nr_t From;
    BlockList_t::nr_t To;
    ElementIndexInBlock_t LastBlockElementCount;

  public: /* ------------------------------------------------------------------------------ */

    ShapeTypeIndex_t Init(shaper_t& shaper, KeyPackIndex_t kpi, bmid_t bmid) {
      auto& kp = shaper.KeyPacks[kpi];
      bm_BaseData_t* bmbase = (bm_BaseData_t*)kp.bm[bmid];
      sti = bmbase->sti;
      From = bmbase->FirstBlockNR;
      To = bmbase->LastBlockNR;
      LastBlockElementCount = bmbase->LastBlockElementCount;
      return bmbase->sti;
    }
    bool Loop(shaper_t& shaper) {
      if (From == To) {
        return false;
      }
      From = From.Next(&shaper.ShapeTypes[sti].BlockList);
      return true;
    }
    MaxElementPerBlock_t GetAmount(shaper_t& shaper) {
      if (From == To) {
        return (MaxElementPerBlock_t)LastBlockElementCount + 1;
      }
      return shaper.ShapeTypes[sti].MaxElementPerBlock();
    }
    shaper_set_RenderDataOffsetType GetRenderDataOffset(shaper_t& shaper) {
      return shaper.GetRenderDataOffset(sti, From);
    }
    BlockList_t::nr_t GetBlockID() {
      return From;
    }
    void* GetData(shaper_t& shaper) {
      return shaper._GetData(sti, From, 0);
    }
  };
};

#undef gloco
#undef shaper_set_fan

#undef shaper_set_RenderDataOffsetType
#undef shaper_set_MaxShapeDataSize
#undef shaper_set_MaxShapeRenderDataSize
#undef shaper_set_MaxKeySizesSum
#undef shaper_set_MaxMaxElementPerBlock
#undef shaper_set_MaxKeyPackAmount
#undef shaper_set_MaxKeyType
#undef shaper_set_MaxKeySize
#undef shaper_set_MaxKeyInPack
#undef shaper_set_MaxShapeTypes