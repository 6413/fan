#ifndef shaper_set_ShapeTypeChange
  #define shaper_set_ShapeTypeChange
#endif

#ifndef shaper_set_MaxShapeTypes
  #define shaper_set_MaxShapeTypes 0x80
#endif
//////////
/* in bytes */
#ifndef shaper_set_MaxKeySize
  #define shaper_set_MaxKeySize 8
#endif

#ifndef shaper_set_MaxKeyType
  #define shaper_set_MaxKeyType 0xff
#endif

#ifdef shaper_set_MaxElementPerBlock
  #error did you meant to define shaper_set_MaxMaxElementPerBlock
#endif
#ifndef shaper_set_MaxMaxElementPerBlock
  #define shaper_set_MaxMaxElementPerBlock 0x8000
#endif

/* in bytes */
#ifndef shaper_set_MaxKeySizesSum
  #define shaper_set_MaxKeySizesSum 0x80
#endif

#ifndef shaper_set_MaxKeyAmountInBM
  #define shaper_set_MaxKeyAmountInBM 0x3f
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

struct shaper_t{
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

  #if shaper_set_MaxKeySizesSum <= 0xff
    typedef uint8_t KeySizesSumInBytes_t;
  #else
    #error ?
  #endif

  /* TOOD that 1 needs to be KeyTypeIndex_t */
  #if shaper_set_MaxKeySizesSum + shaper_set_MaxKeyAmountInBM * 1 <= 0xff
    typedef uint8_t KeyPackSize_t;
  #else
    #error ?
  #endif

  #if shaper_set_MaxKeyAmountInBM <= 0xff
    typedef uint8_t KeyIndexInBM_t;
  #else
    #error ?
  #endif
  #if shaper_set_MaxKeyAmountInBM <= 0x100
    typedef uint8_t KeyAmountInBM_t;
  #else
    #error ?
  #endif
  /* they are internal. used for point key's itself and value of it */
  #if shaper_set_MaxKeyAmountInBM * 2 <= 0xff
    typedef uint8_t _KeyIndexInBM_t;
  #else
    #error ?
  #endif
  #if shaper_set_MaxKeyAmountInBM * 2 <= 0x100
    typedef uint8_t _KeyAmountInBM_t;
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
  typedef uint16_t _blid_t;

  #define BDBT_set_prefix _KeyTree
  #define BDBT_set_type_node ktbmnr_t
  #define BDBT_set_lcpp
  #ifdef shaper_set_MaxKeySize
    #define BDBT_set_MaxKeySize (shaper_set_MaxKeySize * 8)
  #endif
  #define BDBT_set_AreWeInsideStruct 1
  #include <BDBT/BDBT.h>
  _KeyTree_t _KeyTree;
  _KeyTree_NodeReference_t _KeyTree_root;

  public: /* -------------------------------------------------------------------------------- */

  typedef _KeyTree_BitOrder_t KeyBitOrder_t;
  constexpr static KeyBitOrder_t KeyBitOrderLow = _KeyTree_BitOrderLow;
  constexpr static KeyBitOrder_t KeyBitOrderHigh = _KeyTree_BitOrderHigh;
  constexpr static KeyBitOrder_t KeyBitOrderAny = _KeyTree_BitOrderAny;

  public: /* ------------------------------------------------------------------------------- */

  struct KeyType_t{
    KeySizeInBytes_t Size;
    KeyBitOrder_t BitOrder;

    /* size in bits */
    KeySizeInBits_t sibit(){
      return (KeySizeInBits_t)Size * 8;
    }
  };
  KeyType_t *_KeyTypes;
  KeyTypeAmount_t KeyTypeAmount;

  bool _InformCapacity;
  #define BLL_set_Clear 1
  #define BLL_set_nrtra 1
  #define BLL_set_prefix BlockList
  #define BLL_set_CapacityUpdateInfo \
    auto st = OFFSETLESS(bll, ShapeType_t, BlockList); \
    if(st->shaper->_InformCapacity){ \
      st->shaper->_BlockListCapacityChange(st->sti, old_capacity, new_capacity); \
    }
  #define BLL_set_Link 1
  #define BLL_set_LinkSentinel 0
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_type_node _blid_t
  #include <BLL/BLL.h>
  using blid_t = BlockList_t::nr_t;
  struct ShapeType_t{
    ShapeType_t() {
      std::construct_at(&renderer.gl, gl_t{});
    }
    /* this will be used from BlockList callbacks with offsetless */
    shaper_t *shaper;
    ShapeTypeAmount_t sti;

    /* 
      (RenderDataSize + DataSize + sizeof(ShapeList_t::nr_t)) * (MaxElementPerBlock_m1 + 1) +
      sizeof(BlockUnique_t)
    */
    BlockList_t BlockList;

    ElementIndexInBlock_t MaxElementPerBlock_m1; /* minus 1, (p.MaxElementPerBlock[n] - 1) */
    ShapeRenderDataSize_t RenderDataSize;
    ShapeDataSize_t DataSize;

    #if shaper_set_fan
    struct gl_t {
      gl_t() = default;
      fan::opengl::core::vao_t m_vao;
      fan::opengl::core::vbo_t m_vbo;

      std::vector<shape_gl_init_t> locations;
      fan::graphics::shader_nr_t shader;
      bool instanced = true;
      GLsizei vertex_count = 6;
    };
    #if defined(fan_vulkan)
    struct vk_t {
      vk_t() = default;
      fan::vulkan::context_t::pipeline_t pipeline;
      fan::vulkan::context_t::ssbo_t shape_data;
      uint32_t vertex_count = 6;
    };
#endif
    struct renderer_t {
      enum class type_t {
        none,
        gl,
#if defined(fan_vulkan)
        vk
#endif
      };

      type_t type = type_t::gl;

      renderer_t() {}

      renderer_t(const renderer_t& other) {
        type = other.type;
        switch (type) {
        case type_t::gl:
          new (&gl) gl_t(other.gl);
          break;
#if defined(fan_vulkan)
        case type_t::vk:
          new (&vk) vk_t(other.vk);
          break;
#endif
        case type_t::none:
          break;
        }
      }

      renderer_t& operator=(const renderer_t& other) {
        if (this == &other) return *this;

        destroy();

        type = other.type;
        switch (type) {
        case type_t::gl:
          new (&gl) gl_t(other.gl);
          break;
#if defined(fan_vulkan)
        case type_t::vk:
          new (&vk) vk_t(other.vk);
          break;
#endif
        case type_t::none:
          break;
        }

        return *this;
      }

      ~renderer_t() {
        destroy();
      }

      void destroy() {
        switch (type) {
        case type_t::gl:
          gl.~gl_t();
          break;
#if defined(fan_vulkan)
        case type_t::vk:
          vk.~vk_t();
          break;
#endif
        case type_t::none:
          break;
        }
        type = type_t::none;
      }

      union {
        gl_t gl;
#if defined(fan_vulkan)
        vk_t vk;
#endif
      };
    }renderer;

    #endif
    //fan::vulkan::context_t::descriptor_t<vulkan_buffer_count>
    MaxElementPerBlock_t MaxElementPerBlock(){
      return (MaxElementPerBlock_t)MaxElementPerBlock_m1 + 1;
    }
  };
  #define BLL_set_prefix ShapeTypes
  #define BLL_set_Link 0
  #define BLL_set_Recycle 0
  #define BLL_set_IntegerNR 1
  #define BLL_set_CPP_ConstructDestruct 1
  #define BLL_set_CPP_Node_ConstructDestruct 1
  #define BLL_set_CPP_CopyAtPointerChange 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_NodeDataType ShapeType_t
  #define BLL_set_Usage 1
  #define BLL_set_type_node ShapeTypeAmount_t
  #include <BLL/BLL.h>
  ShapeTypes_t ShapeTypes;

  #define BLL_set_prefix BlockManager
  #define BLL_set_CPP_Node_ConstructDestruct 1
  #define BLL_set_Link 0
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_type_node ktbmnr_t
  #define BLL_set_NodeData \
    KeyPackSize_t KeyPackSize; \
    uint8_t *KeyPack; \
    ShapeTypeAmount_t sti; \
    BlockList_NodeReference_t FirstBlockNR; \
    BlockList_NodeReference_t LastBlockNR; \
    ElementIndexInBlock_t LastBlockElementCount; \
    ~BlockManager_NodeData_t(){ \
      A_resize(KeyPack, 0); \
    }
  #include <BLL/BLL.h>
  BlockManager_t BlockManager;

  #pragma pack(push, 1)
    struct shape_t{
      ShapeTypeIndex_t sti;
      BlockManager_t::nr_t bmid;
      BlockList_t::nr_t blid;
      ElementIndexInBlock_t ElementIndex;
    };
  #pragma pack(pop)

  #define BLL_set_Usage 1
  #define BLL_set_prefix ShapeList
  #define BLL_set_Link 0
  #define BLL_set_NodeDataType shape_t
  #define BLL_set_AreWeInsideStruct 1
  // actually it needs to be uint24_t
  #define BLL_set_type_node uint32_t
  #include <BLL/BLL.h>
  ShapeList_t ShapeList;

  #define BLL_set_Clear 1
  #define BLL_set_prefix BlockEditQueue
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_NodeData \
    ShapeTypeIndex_t sti; \
    BlockList_t::nr_t blid;
  #define BLL_set_type_node uint16_t
  #include <BLL/BLL.h>
  BlockEditQueue_t BlockEditQueue;

  struct BlockUnique_t{
    shaper_set_RenderDataOffsetType MinEdit;
    shaper_set_RenderDataOffsetType MaxEdit;
    BlockEditQueue_t::nr_t beid;

    void clear() {
      beid.sic();
      MinEdit = (decltype(MinEdit))-1;
      MaxEdit = 0;
    }
    void constructor(){
      clear();
    }
    void destructor(shaper_t &shaper){
      if(!beid.iic()){
        shaper.BlockEditQueue.unlrec(beid);
      }
    }
  };

  /*
    block data format is this. %30
    RenderData   Data         ShapeID
    [|||       ] [|||       ] [|||       ]
  */

  /* TODO those are made uintptr_t to not overflow, maybe there is better way? */
  ShapeRenderData_t *_GetRenderData(
    BlockList_t &BlockList,
    ShapeRenderDataSize_t RenderDataSize,
    ShapeDataSize_t DataSize,
    MaxElementPerBlock_t MaxElementPerBlock,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ){
    return &((ShapeRenderData_t *)BlockList[blid])[
      (uintptr_t)RenderDataSize * ElementIndex
    ];
  }
  ShapeData_t *_GetData(
    BlockList_t &BlockList,
    ShapeRenderDataSize_t RenderDataSize,
    ShapeDataSize_t DataSize,
    MaxElementPerBlock_t MaxElementPerBlock,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ){
    return &_GetRenderData(
      BlockList,
      RenderDataSize,
      DataSize,
      MaxElementPerBlock,
      blid,
      MaxElementPerBlock
    )[(uintptr_t)DataSize * ElementIndex];
  }
  ShapeList_t::nr_t *_GetShapeID(
    BlockList_t &BlockList,
    ShapeRenderDataSize_t RenderDataSize,
    ShapeDataSize_t DataSize,
    MaxElementPerBlock_t MaxElementPerBlock,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ){
    return (ShapeList_t::nr_t *)&_GetData(
      BlockList,
      RenderDataSize,
      DataSize,
      MaxElementPerBlock,
      blid,
      MaxElementPerBlock
    )[sizeof(ShapeList_t::nr_t) * ElementIndex];
  }
  BlockUnique_t &_GetBlockUnique(
    BlockList_t &BlockList,
    ShapeRenderDataSize_t RenderDataSize,
    ShapeDataSize_t DataSize,
    MaxElementPerBlock_t MaxElementPerBlock,
    BlockList_t::nr_t blid
  ){
    return *(BlockUnique_t *)_GetShapeID(
      BlockList,
      RenderDataSize,
      DataSize,
      MaxElementPerBlock,
      blid,
      MaxElementPerBlock
    );
  }

  ShapeRenderData_t *GetRenderData(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ){
    auto &st = ShapeTypes[sti];
    return _GetRenderData(
      st.BlockList,
      st.RenderDataSize,
      st.DataSize,
      st.MaxElementPerBlock(),
      blid,
      ElementIndex
    );
  }
  ShapeData_t *GetData(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ){
    auto &st = ShapeTypes[sti];
    return _GetData(
      st.BlockList,
      st.RenderDataSize,
      st.DataSize,
      st.MaxElementPerBlock(),
      blid,
      ElementIndex
    );
  }
  ShapeList_t::nr_t *GetShapeID(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid,
    uintptr_t ElementIndex
  ){
    auto &st = ShapeTypes[sti];
    return _GetShapeID(
      st.BlockList,
      st.RenderDataSize,
      st.DataSize,
      st.MaxElementPerBlock(),
      blid,
      ElementIndex
    );
  }
  BlockUnique_t &GetBlockUnique(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid
  ){
    auto &st = ShapeTypes[sti];
    return _GetBlockUnique(
      st.BlockList,
      st.RenderDataSize,
      st.DataSize,
      st.MaxElementPerBlock(),
      blid
    );
  }

  public: /* -------------------------------------------------------------------------------- */

  
  using bmid_t = BlockManager_t::nr_t;

  #if shaper_set_fan
    fan::graphics::shader_nr_t& GetShader(ShapeTypeIndex_t sti) {
      auto& d = ShapeTypes[sti];
      if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
        return d.renderer.gl.shader;
      }
      #if defined(fan_vulkan)
      else if (get_loco()->get_renderer() == loco_t::renderer_t::vulkan) {
        return d.renderer.vk.pipeline.shader_nr;
      }
      #endif
      fan::throw_error("");
      static fan::graphics::shader_nr_t doesnt_happen;
      return doesnt_happen;
    }
    fan::opengl::core::vao_t GetVAO(ShapeTypeIndex_t sti) {
      auto& st = ShapeTypes[sti];
      if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
        return st.renderer.gl.m_vao;
      }
      fan::throw_error("Unsupported renderer type");
      fan::opengl::core::vao_t doesnt_happen;
      return doesnt_happen;
    }
    fan::opengl::core::vbo_t GetVBO(ShapeTypeIndex_t sti) {
      auto& st = ShapeTypes[sti];
      if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
        return st.renderer.gl.m_vbo;
      }
      fan::throw_error("Unsupported renderer type");
      fan::opengl::core::vbo_t doesnt_happen;
      return doesnt_happen;
    }
    std::vector<shape_gl_init_t>& GetLocations(ShapeTypeIndex_t sti) {
      auto& st = ShapeTypes[sti];
      if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
        return st.renderer.gl.locations;
      }
      fan::throw_error("Unsupported renderer type");
      static std::vector<shape_gl_init_t> doesnt_happen;
      return doesnt_happen;
    }
    ShapeTypes_NodeData_t& GetShapeTypes(ShapeTypeIndex_t sti) {
      return ShapeTypes[sti];
    }
  #endif

  struct ShapeID_t : ShapeList_t::nr_t{
    
    ShapeID_t() = default;

    ShapeTypeIndex_t sti(shaper_t &shaper){
      return shaper.ShapeList[*this].sti;
    }
    bmid_t bmid(shaper_t &shaper){
      return shaper.ShapeList[*this].bmid;
    }
    blid_t blid(shaper_t &shaper){
      return shaper.ShapeList[*this].blid;
    }
    ElementIndexInBlock_t eiib(shaper_t &shaper){
      return shaper.ShapeList[*this].ElementIndex;
    }

    ShapeRenderData_t *GetRenderData(shaper_t &shaper) const {
      auto &s = shaper.ShapeList[*this];
      return shaper.GetRenderData(s.sti, s.blid, s.ElementIndex);
    }
    ShapeData_t *GetData(shaper_t &shaper) const {
      auto &s = shaper.ShapeList[*this];
      return shaper.GetData(s.sti, s.blid, s.ElementIndex);
    }

    using ShapeList_t::nr_t::nr_t;
    constexpr ShapeID_t(ShapeList_t::nr_t nr) : ShapeList_t::nr_t(nr) {}
  };

  ShapeRenderDataSize_t GetRenderDataSize(ShapeTypeIndex_t sti){
    return ShapeTypes[sti].RenderDataSize;
  }
  ShapeDataSize_t GetDataSize(ShapeTypeIndex_t sti){
    return ShapeTypes[sti].DataSize;
  }

  KeyPackSize_t GetKeyOffset(
    KeyIndexInBM_t kiibm,
    KeyPackSize_t oo /* output offset */
  ){
    return kiibm * sizeof(KeyTypeIndex_t) + sizeof(KeyTypeIndex_t) + oo;
  }
  KeyPackSize_t GetKeysSize(ShapeID_t sid){
    auto &s = ShapeList[sid];
    auto &bm = BlockManager[s.bmid];
    return bm.KeyPackSize;
  }
  uint8_t *GetKeys(ShapeID_t ShapeID){
    auto &s = ShapeList[ShapeID];
    auto &bm = BlockManager[s.bmid];
    return bm.KeyPack;
  }
  void WriteKeys(ShapeID_t ShapeID, void *dst){
    auto &s = ShapeList[ShapeID];
    auto &bm = BlockManager[s.bmid];
    __builtin_memcpy(dst, bm.KeyPack, bm.KeyPackSize);
  }

  shaper_set_RenderDataOffsetType GetRenderDataOffset(
    ShapeTypeIndex_t sti,
    blid_t blid
  ){
    auto &st = ShapeTypes[sti];
    return (shaper_set_RenderDataOffsetType)blid.gint() *
      st.MaxElementPerBlock() *
      st.RenderDataSize;
  }

  struct BlockProperties_t{
    BlockProperties_t() {
      std::construct_at(&renderer.gl, gl_t{});
    }
    ~BlockProperties_t() {
#if shaper_set_fan
      std::destroy_at(&renderer.gl);
#endif
    }
    MaxElementPerBlock_t MaxElementPerBlock;
    decltype(ShapeType_t::RenderDataSize) RenderDataSize;
    decltype(ShapeType_t::DataSize) DataSize;

    #if shaper_set_fan
    struct gl_t {
      gl_t() = default;
      std::vector<shape_gl_init_t> locations;
      fan::graphics::shader_nr_t shader;
      bool instanced = true;
      GLuint draw_mode = GL_TRIANGLES;
      GLsizei vertex_count = 6;
    };
#if defined(fan_vulkan)
    struct vk_t {
      vk_t() = default;
      fan::vulkan::context_t::pipeline_t pipeline;
      fan::vulkan::context_t::ssbo_t shape_data;
      uint32_t vertex_count = 6;
    };
#endif
    struct renderer_t {
      renderer_t() {}
      ~renderer_t() {}
      union {
        gl_t gl;
      #if defined(fan_vulkan)
        vk_t vk;
      #endif
      };
    }renderer;

    #endif
  };

  void Open(){
    KeyTypeAmount = 0;
    _KeyTypes = NULL;
    _InformCapacity = 1;

    _KeyTree.Open();
    _KeyTree_root = _KeyTree.NewNode();
    BlockManager.Open();
    BlockEditQueue.Open();
    ShapeList.Open();
  }
  void Close(){
    ShapeList.Close();
    BlockEditQueue.Close();
    BlockManager.Close();
    _KeyTree.Close();
    A_resize(_KeyTypes, 0);
  }

  void AddKey(KeyTypeIndex_t KeyTypeIndex, KeySizeInBytes_t Size, KeyBitOrder_t BitOrder){
    if(KeyTypeIndex >= KeyTypeAmount){
      KeyTypeAmount = KeyTypeIndex;
      KeyTypeAmount++;
      _KeyTypes = (KeyType_t *)A_resize(
        _KeyTypes,
        (uintptr_t)KeyTypeAmount * sizeof(KeyType_t)
      );
    }

    _KeyTypes[KeyTypeIndex].Size = Size;
    _KeyTypes[KeyTypeIndex].BitOrder = BitOrder;
  }

  void SetKeyOrder(KeyTypeIndex_t KeyTypeIndex, KeyBitOrder_t BitOrder){
    _KeyTypes[KeyTypeIndex].BitOrder = BitOrder;
  }

  loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, shaper);
  }

  void _ShapeTypeChange(
    ShapeTypeIndex_t sti,
    KeyPackSize_t keypack_size,
    uint8_t *keypack,
    MaxElementPerBlock_t element_count,
    const void *old_renderdata,
    const void *old_data,
    void *new_renderdata,
    void *new_data
  ){
    shaper_set_ShapeTypeChange
  }

  void SetShapeType(
    ShapeTypeIndex_t sti,
    const BlockProperties_t& bp
  ){
    while(sti >= ShapeTypes.Usage()){
      auto csti = ShapeTypes.NewNode();
      auto &st = ShapeTypes[csti];

      /* filler init */
      st.shaper = this;
      st.sti = csti;
      st.BlockList.Open(1);
      #if fan_debug >= fan_debug_high
      st.sti = -1;
      #endif
    }

    auto &st = ShapeTypes[sti];

    _InformCapacity = 0;
    {
      BlockList_t BlockList = st.BlockList;
      st.BlockList.Open(
        (
          (uintptr_t)bp.RenderDataSize + bp.DataSize + sizeof(ShapeList_t::nr_t)
        ) * (bp.MaxElementPerBlock) + sizeof(BlockUnique_t)
      );

      /* slow and steady */
      while(st.BlockList.SizeNormalized() != BlockList.SizeNormalized()){
        st.BlockList.NewNode();
      }

      {
        BlockList_t::RecycleTraverse_t rt;
        BlockList_t::nr_t node_id;
        rt.Open(&BlockList, &node_id);
        while(rt.Loop(&BlockList, &node_id)){
          st.BlockList.Recycle(node_id);
        }
        rt.Close(&BlockList);
      }

      {
        BlockManager_t::nrtra_t t;
        BlockManager_t::nr_t bm_id;
        t.Open(&BlockManager, &bm_id);
        while(t.Loop(&BlockManager, &bm_id)){
          auto &bm = BlockManager[bm_id];
          if(bm.sti != sti){
            continue;
          }

          auto block_id = bm.FirstBlockNR;
          while(1){
            MaxElementPerBlock_t element_count = st.MaxElementPerBlock_m1;
            if(block_id == bm.LastBlockNR){
              element_count = bm.LastBlockElementCount;
            }
            element_count++;

            if(st.MaxElementPerBlock() != bp.MaxElementPerBlock){
              /* not supported yet */
              __abort();
            }

            _ShapeTypeChange(
              sti,
              bm.KeyPackSize,
              bm.KeyPack,
              element_count,
              (const void *)_GetRenderData(
                BlockList,
                st.RenderDataSize,
                st.DataSize,
                st.MaxElementPerBlock(),
                block_id,
                0
              ),
              (const void *)_GetData(
                BlockList,
                st.RenderDataSize,
                st.DataSize,
                st.MaxElementPerBlock(),
                block_id,
                0
              ),
              (void *)_GetRenderData(
                st.BlockList,
                bp.RenderDataSize,
                bp.DataSize,
                bp.MaxElementPerBlock,
                block_id,
                0
              ),
              (void *)_GetData(
                st.BlockList,
                bp.RenderDataSize,
                bp.DataSize,
                bp.MaxElementPerBlock,
                block_id,
                0
              )
            );

            if(block_id == bm.LastBlockNR){
              break;
            }
            block_id = block_id.Next(&BlockList);
          }
        }
        t.Close(&BlockManager);
      }

      BlockList.Close();
    }
    _InformCapacity = 1;

    st.sti = sti;
    st.MaxElementPerBlock_m1 = bp.MaxElementPerBlock - 1;
    st.RenderDataSize = bp.RenderDataSize;
    st.DataSize = bp.DataSize;

  #if shaper_set_fan
    if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
      st.renderer.gl = ShapeType_t::gl_t{};
      get_loco()->gl.add_shape_type(st, bp);
    }
    #if defined(fan_vulkan)
    else if (get_loco()->get_renderer() == loco_t::renderer_t::vulkan) {
      ShapeType_t::vk_t d;
      auto& bpr = bp.renderer.vk;
      d.pipeline = bpr.pipeline;
      d.shape_data = bpr.shape_data;
      d.vertex_count = bpr.vertex_count;
      st.renderer = d;
      //st.renderer.emplace<ShapeType_t::vk_t>();
    }
#endif
  #endif
  }

  void ProcessBlockEditQueue(){
    #if shaper_set_fan
    fan::opengl::context_t &context = get_loco()->context.gl;
    #endif

    auto beid = BlockEditQueue.GetNodeFirst();
    while(beid != BlockEditQueue.dst){
      auto &be = BlockEditQueue[beid];
      auto &st = ShapeTypes[be.sti];
      auto &bu = GetBlockUnique(be.sti, be.blid);

      #if shaper_set_fan
      if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
        auto& gl = st.renderer.gl;
        gl.m_vao.bind(context);
        fan::opengl::core::edit_glbuffer(
          context,
          gl.m_vbo.m_buffer,
          GetRenderData(be.sti, be.blid, 0) + bu.MinEdit,
          GetRenderDataOffset(be.sti, be.blid) + bu.MinEdit,
          bu.MaxEdit - bu.MinEdit,
          GL_ARRAY_BUFFER
        );
      }
      #if defined(fan_vulkan)
      else if (get_loco()->get_renderer() == loco_t::renderer_t::vulkan) {
        auto& vk = st.renderer.vk;
        auto wrote = bu.MaxEdit - bu.MinEdit;
        for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
          memcpy(
            vk.shape_data.data[frame] + (GetRenderDataOffset(be.sti, be.blid) + bu.MinEdit), // data  + offset
            GetRenderData(be.sti, be.blid, 0) + bu.MinEdit,
            wrote
          );
        }
        
      }
#endif
      #endif

      bu.clear();

      beid = beid.Next(&BlockEditQueue);
    }

    BlockEditQueue.Clear();
  }

  void ElementIsPartiallyEdited(
    ShapeTypeIndex_t sti,
    blid_t blid,
    uint16_t eiib,
    shaper_set_RenderDataOffsetType byte_start,
    shaper_set_RenderDataOffsetType byte_count
  ){
    ShapeType_t &st = ShapeTypes[sti];
    BlockUnique_t &bu = GetBlockUnique(sti, blid);

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

    if(!bu.beid.iic()){
      return;
    }

    bu.beid = BlockEditQueue.NewNodeLast();
    auto &be = BlockEditQueue[bu.beid];
    be.sti = sti;
    be.blid = blid;
  }
  void ElementIsFullyEdited(
    ShapeTypeIndex_t sti,
    blid_t blid,
    uint16_t eiib
  ){
    auto &st = ShapeTypes[sti];
    ElementIsPartiallyEdited(sti, blid, eiib, 0, st.RenderDataSize);
  }

  void _RenderDataReset(ShapeTypeIndex_t sti){
    auto &st = ShapeTypes[sti];
    #if shaper_set_fan
    /* TODO remove all block edit queue stuff */
    BlockList_t::nrtra_t traverse;
    BlockList_t::nr_t node_id;
    traverse.Open(&st.BlockList, &node_id);
    
    if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
      auto& gl = st.renderer.gl;
      fan::opengl::context_t &context = get_loco()->context.gl;
      gl.m_vao.bind(context);
      while(traverse.Loop(&st.BlockList, &node_id)){
        fan::opengl::core::edit_glbuffer(
          context,
          gl.m_vbo.m_buffer,
          GetRenderData(sti, node_id, 0),
          GetRenderDataOffset(sti, node_id),
          st.RenderDataSize * st.MaxElementPerBlock(),
          GL_ARRAY_BUFFER
        );
      }
    }
    #if defined(fan_vulkan)
    else if (get_loco()->get_renderer() == loco_t::renderer_t::vulkan){
      auto& vk = st.renderer.vk;
      while (traverse.Loop(&st.BlockList, &node_id)) {
        for (uint32_t frame = 0; frame < fan::vulkan::max_frames_in_flight; frame++) {
          memcpy(vk.shape_data.data[frame], GetRenderData(sti, node_id, 0), st.RenderDataSize * st.MaxElementPerBlock());
        }
      }
    }
    #endif
    traverse.Close(&st.BlockList);
    #endif
  }
  void _BlockListCapacityChange(
    ShapeTypeIndex_t sti,
    uintptr_t old_capacity,
    uintptr_t new_capacity
  ){
    auto &st = ShapeTypes[sti];

    #if shaper_set_fan
    if (get_loco()->get_renderer() == loco_t::renderer_t::opengl) {
      auto& gl = st.renderer.gl;
      gl.m_vbo.bind(get_loco()->get_context().gl);
      fan::opengl::core::write_glbuffer(
        get_loco()->get_context().gl,
        gl.m_vbo.m_buffer,
        0,
        new_capacity * st.RenderDataSize * st.MaxElementPerBlock(),
        GL_DYNAMIC_DRAW,
        GL_ARRAY_BUFFER
      );
      _RenderDataReset(sti);
    }
    else if (get_loco()->get_renderer() == loco_t::renderer_t::vulkan){
      //memcpy(vk.shape_data.data, _GetRenderData(sti, node_id, 0) + GetRenderDataOffset(sti, node_id), st.RenderDataSize * st.MaxElementPerBlock());
      //vk.shape_data.allocate(gloco->context.vk, New * st.RenderDataSize * st.MaxElementPerBlock());
      //vk.shape_data.m_descriptor.update(gloco->context.vk, 1, 0, 1, 0);
      //fan::throw_error("");
      _RenderDataReset(sti);
    }
    #endif
  }

  BlockList_t::nr_t _newblid(
    ShapeTypeIndex_t sti,
    BlockManager_NodeData_t *bm
  ){
    auto &st = ShapeTypes[sti];

    bm->LastBlockElementCount = 0;
    auto blid = st.BlockList.NewNode();
    bm->LastBlockNR = blid;

    GetBlockUnique(sti, blid).constructor();

    return blid;
  }
  void _deleteblid(
    ShapeTypeIndex_t sti,
    BlockList_t::nr_t blid
  ){
    auto &st = ShapeTypes[sti];
    //
    //if (std::holds_alternative<ShapeType_t::gl_t>(st.renderer)) {
    //  //auto& gl = std::get<ShapeType_t::gl_t>(st.renderer);
    //  //get_loco()->shader_erase(gl.shader);
    //}
    //else {
    //  //
    //  //auto& vk = std::get<ShapeType_t::vk_t>(st.renderer);
    //  //get_loco()->context.vk.shader_erase(vk.pipeline.shader_nr);
    //  //vk.pipeline.close(get_loco()->context.vk);
    //  //vk.shape_data.close(get_loco()->context.vk);
    //}

    GetBlockUnique(sti, blid).destructor(*this);

    st.BlockList.Recycle(blid);
  }

  static void _kti_SetLastBit(KeyTypeIndex_t &kti){
    kti |= (KeyTypeIndex_t)1 << sizeof(KeyTypeIndex_t) * 8 - 1; /* 0x80... */
  }
  static KeyTypeIndex_t _kti_GetNormal(KeyTypeIndex_t kti){
    KeyTypeIndex_t p = 0;
    _kti_SetLastBit(p);
    return kti & ~p;
  }
  static bool _kti_GetLastBit(KeyTypeIndex_t kti){
    return _kti_GetNormal(kti) != kti;
  }

  void PrepareKeysForAdd(
      const void *KeyPack,
      KeyPackSize_t LastKeyOffset
    ){
      auto _KeyPack = (KeyData_t *)KeyPack;
      _kti_SetLastBit(*(KeyTypeIndex_t *)&_KeyPack[LastKeyOffset]);
  }
  ShapeID_t add(
    ShapeTypeIndex_t sti,
    const void *KeyPack,
    KeyPackSize_t KeyPackSize,
    const void *RenderData,
    const void *Data
  ){
    auto _KeyPack = (KeyData_t *)KeyPack;

    bmid_t bmid;
    BlockManager_NodeData_t *bm;

    auto &st = ShapeTypes[sti];
  #if fan_debug >= fan_debug_high
    if (st.sti == (decltype(st.sti))-1) {
      fan::throw_error("");
    }
  #endif

    _KeyTree_NodeReference_t nr = _KeyTree_root;
    KeyPackSize_t ikp = 0;
    _KeyTree_KeySize_t bdbt_ki;
    KeyType_t *kt;
    uint8_t step;

    /* DEBUG_HINT if this loop goes above KeyPackSize, your KeyPack is bad */
    while(ikp != KeyPackSize){

      auto kti = (KeyTypeIndex_t *)&_KeyPack[ikp];
      _KeyTree_QueryNoPointer(&_KeyTree, true, sizeof(*kti) * 8, kti, &bdbt_ki, &nr);
      if(bdbt_ki != sizeof(*kti) * 8){
        step = 0;
        goto gt_newbm;
      }
      ikp += sizeof(*kti);

      kt = &_KeyTypes[_kti_GetNormal(*kti)];
      _KeyTree_QueryNoPointer(&_KeyTree, true, kt->sibit(), &_KeyPack[ikp], &bdbt_ki, &nr);
      if(bdbt_ki != kt->sibit()){
        step = 1;
        goto gt_newbm;
      }
      ikp += kt->Size;
    }

    bmid = *(BlockManager_t::nr_t *)&nr;
    bm = &BlockManager[bmid];

    if(bm->LastBlockElementCount == st.MaxElementPerBlock_m1){
      auto lbnr = bm->LastBlockNR;
      auto nblid = _newblid(sti, bm);
      st.BlockList.linkNextOfOrphan(lbnr, nblid);
    }
    else{
      bm->LastBlockElementCount++;
    }

    goto gt_nonewbm;

    gt_newbm:

    bmid = BlockManager.NewNode();
    bm = &BlockManager[bmid];
    bm->KeyPackSize = KeyPackSize;
    bm->KeyPack = (uint8_t *)A_resize(NULL, bm->KeyPackSize);
    __builtin_memcpy(bm->KeyPack, KeyPack, bm->KeyPackSize);
    bm->sti = sti;
    bm->FirstBlockNR = _newblid(sti, bm);

    /* DEBUG_HINT if this loop goes above KeyPackSize, your KeyPack is bad */
    while(ikp != KeyPackSize){
      _KeyTree_NodeReference_t out;
      if(step == 0){
        auto kti = (KeyTypeIndex_t *)&_KeyPack[ikp];
        kt = &_KeyTypes[_kti_GetNormal(*kti)];

        out = _KeyTree.NewNode();

        _KeyTree_Add(&_KeyTree, true, sizeof(*kti) * 8, kti, bdbt_ki, nr, out);

        ikp += sizeof(*kti);
      }
      else if(step == 1){
        if(ikp + kt->Size != KeyPackSize){
          out = _KeyTree.NewNode();
        }
        else{
          out = *(_KeyTree_NodeReference_t *)&bmid;
        }

        _KeyTree_Add(&_KeyTree, true, kt->sibit(), &_KeyPack[ikp], bdbt_ki, nr, out);

        ikp += kt->Size;
      }
      else{
        __unreachable();
      }
      bdbt_ki = 0;
      nr = out;
      step ^= 1;
    }

    gt_nonewbm:

    auto shapeid = ShapeList.NewNode();
    ShapeList[shapeid].sti = sti;
    ShapeList[shapeid].bmid = bmid;
    ShapeList[shapeid].blid = bm->LastBlockNR;
    ShapeList[shapeid].ElementIndex = bm->LastBlockElementCount;

    __builtin_memcpy(
      GetRenderData(sti, bm->LastBlockNR, bm->LastBlockElementCount),
      RenderData,
      st.RenderDataSize
    );
    __builtin_memcpy(
      GetData(sti, bm->LastBlockNR, bm->LastBlockElementCount),
      Data,
      st.DataSize
    );
    *GetShapeID(sti, bm->LastBlockNR, bm->LastBlockElementCount) = shapeid;

    ElementIsFullyEdited(sti, bm->LastBlockNR, bm->LastBlockElementCount);
    return shapeid;
  }

  void remove(
    ShapeList_t::nr_t sid
  ){
    auto &s = ShapeList[sid];
    auto sti = s.sti;
    auto &st = ShapeTypes[sti];
    auto bmid = s.bmid;
    auto &bm = BlockManager[bmid];
    auto lsid = *GetShapeID(sti, bm.LastBlockNR, bm.LastBlockElementCount);
    if(sid != lsid){
      ElementIsFullyEdited(sti, s.blid, s.ElementIndex);
    }

    auto &ls = ShapeList[lsid];
    
    __builtin_memmove(
      GetRenderData(sti, s.blid, s.ElementIndex),
      GetRenderData(sti, ls.blid, ls.ElementIndex),
      st.RenderDataSize
    );
    __builtin_memmove(
      GetData(sti, s.blid, s.ElementIndex),
      GetData(sti, ls.blid, ls.ElementIndex),
      st.DataSize
    );
    *GetShapeID(sti, s.blid, s.ElementIndex) =
      *GetShapeID(sti, ls.blid, ls.ElementIndex);

    ls.blid = s.blid;
    ls.ElementIndex = s.ElementIndex;

    ShapeList.Recycle(sid);

    /* sid we deleted may be same as last so lets no longer access s and ls */

    /* we just deleted last so lets check if we can just decrease count */
    if(bm.LastBlockElementCount != 0){
      bm.LastBlockElementCount--;
      return;
    }

    /* looks like we deleted last standing element in block so we need to delete block */

    /* before that, lets be sure its not last standing block in bm */
    if(bm.FirstBlockNR != bm.LastBlockNR){
      auto lblid = bm.LastBlockNR;
      bm.LastBlockNR = lblid.Prev(&st.BlockList);
      _deleteblid(sti, lblid);
      bm.LastBlockElementCount = st.MaxElementPerBlock_m1;
      return;
    }

    /* good luck we need to remove block manager completely */
    /* aaaaaaaaaaaa */

    _deleteblid(sti, bm.LastBlockNR);

    _KeyTree_NodeReference_t knrs[shaper_set_MaxKeyAmountInBM * 2];
    KeySizeInBytes_t ks[shaper_set_MaxKeyAmountInBM * 2];

    auto KeyPack = bm.KeyPack;
    KeyPackSize_t ikp = 0;
    _KeyIndexInBM_t _kiibm = 0;
    {
      auto knr = _KeyTree_root;
      while (ikp != bm.KeyPackSize) {
        auto kti = (KeyTypeIndex_t*)&KeyPack[ikp];
        knrs[_kiibm] = knr;
        ks[_kiibm++] = sizeof(*kti);
        _KeyTree_ConfidentQuery(&_KeyTree, true, sizeof(*kti) * 8, kti, &knr);
        ikp += sizeof(*kti);
        auto& kt = _KeyTypes[_kti_GetNormal(*kti)];
        knrs[_kiibm] = knr;
        ks[_kiibm++] = kt.Size;
        _KeyTree_ConfidentQuery(&_KeyTree, true, kt.sibit(), &KeyPack[ikp], &knr);
        ikp += kt.Size;
      }
    }

    /* TODO this part can be faster if used some different function instead of .r */
    --_kiibm;
    while(1){
      auto size = ks[_kiibm];
      ikp -= size;
      {
        auto knr = knrs[_kiibm];
        _KeyTree_Remove(&_KeyTree, true, (KeySizeInBits_t)size * 8, &KeyPack[ikp], &knr);
      }
      if(_KeyTree.inrhc(knrs[_kiibm])){
        break;
      }
      if(_kiibm == 0){
        break;
      }
      _KeyTree.Recycle(knrs[_kiibm]);
      _kiibm--;
    }

    BlockManager.Recycle(bmid);
  }

  #pragma pack(push, 1)
  struct KeyTraverse_t{
    uint8_t State;
    KeyIndexInBM_t kiibm;
    _KeyTree_NodeReference_t knr;
    KeyType_t *kt;
    bool isbm;

    KeyTypeIndex_t kd0[shaper_set_MaxKeyAmountInBM];
    KeyData_t kd1[shaper_set_MaxKeySizesSum][shaper_set_MaxKeyAmountInBM];
    _KeyTree_Traverse_t tra0[shaper_set_MaxKeyAmountInBM];
    _KeyTree_Traverse_t behindtra; /* little maneuver */
    _KeyTree_Traverse_t tra1[shaper_set_MaxKeyAmountInBM];

    void Init(
      shaper_t &shaper
    ){
      State = 0;
      kiibm = 0;
      behindtra.Output = shaper._KeyTree_root;
    }
    bool Loop(
      shaper_t &shaper
    ){
      gt_reswitch:

      switch(State){
        case 0:{
         _KeyTree_TraverseInit(
            &tra0[kiibm],
            KeyBitOrderLow,
            tra1[(uintptr_t)kiibm - 1].Output /* tra1 index is underflowable on purpose */
          );
          State = 1;
        }
        case 1:{
          if(_KeyTree_Traverse(
            &shaper._KeyTree,
            &tra0[kiibm],
            true,
            KeyBitOrderLow,
            sizeof(*kd0) * 8,
            &kd0[kiibm]
          ) == false){
            if(kiibm == 0){
              return false;
            }
            --kiibm;
            isbm = false;
            kt = &shaper._KeyTypes[kd0[kiibm]];
            State = 2;
            goto gt_reswitch;
          }
          isbm = _kti_GetLastBit(kd0[kiibm]);
          kt = &shaper._KeyTypes[_kti_GetNormal(kd0[kiibm])];
          _KeyTree_TraverseInit(
            &tra1[kiibm],
            kt->BitOrder,
            tra0[kiibm].Output
          );
          State = 2;
        }
        case 2:{
          if(_KeyTree_Traverse(
            &shaper._KeyTree,
            &tra1[kiibm],
            true,
            kt->BitOrder,
            kt->sibit(),
            &kd1[kiibm]
          ) == false){
            State = 1;
            goto gt_reswitch;
          }
          if(!isbm){
            ++kiibm;
            State = 0;
          }
          return true;
        }
      }
      __unreachable();
    }
    KeyData_t *kd(){
      return kd1[kiibm - !isbm];
    }
    KeyTypeIndex_t kti(shaper_t &shaper){
      return ((uintptr_t)kt - (uintptr_t)shaper._KeyTypes) / sizeof(_KeyTypes[0]);
    }
    bmid_t bmid(){
      return *(bmid_t *)&tra1[kiibm].Output;
    }
  };
  #pragma pack(pop)

  struct BlockTraverse_t{
    private: /* ----------------------------------------------------------------------------- */

    ShapeTypeIndex_t sti;
    BlockList_t::nr_t From;
    BlockList_t::nr_t To;
    ElementIndexInBlock_t LastBlockElementCount;

    public: /* ------------------------------------------------------------------------------ */

    ShapeTypeIndex_t Init(shaper_t &shaper, bmid_t bmid){
      auto &bm = shaper.BlockManager[bmid];
      sti = bm.sti;
      From = bm.FirstBlockNR;
      To = bm.LastBlockNR;
      LastBlockElementCount = bm.LastBlockElementCount;
      return bm.sti;
    }
    bool Loop(shaper_t &shaper){
      if(From == To){
        return false;
      }
      From = From.Next(&shaper.ShapeTypes[sti].BlockList);
      return true;
    }
    MaxElementPerBlock_t GetAmount(shaper_t &shaper){
      if(From == To){
        return (MaxElementPerBlock_t)LastBlockElementCount + 1;
      }
      return shaper.ShapeTypes[sti].MaxElementPerBlock();
    }
    shaper_set_RenderDataOffsetType GetRenderDataOffset(shaper_t &shaper){
      return shaper.GetRenderDataOffset(sti, From);
    }
    void *GetRenderData(shaper_t &shaper){
      return shaper.GetRenderData(sti, From, 0);
    }
    BlockList_t::nr_t GetBlockID() {
      return From;
    }
    void *GetData(shaper_t &shaper){
      return shaper.GetData(sti, From, 0);
    }
  };
};

#undef gloco
#undef shaper_set_fan

#undef shaper_set_RenderDataOffsetType
#undef shaper_set_MaxShapeDataSize
#undef shaper_set_MaxShapeRenderDataSize
#undef shaper_set_MaxKeyAmountInBM
#undef shaper_set_MaxKeySizesSum
#undef shaper_set_MaxMaxElementPerBlock
#undef shaper_set_MaxKeyType
#undef shaper_set_MaxKeySize
#undef shaper_set_MaxShapeTypes
#undef shaper_set_ShapeTypeChange