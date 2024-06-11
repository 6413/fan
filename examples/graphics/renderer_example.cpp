#include <WITCH/WITCH.h>

struct shaper_t {

private: /* ------------------------------------------------------------------------------- */

  /* key tree and block manager node reference type */
  typedef uint16_t ktbmnr_t;

  /* block list node reference */
  /* total element a shape can have is = (pow(2, (sizeof(blnr_t) * 8)) * MaxElementPerBlock) */
  typedef uint16_t blnr_t;

#define BDBT_set_prefix kt
#define BDBT_set_type_node ktbmnr_t
#define BDBT_set_lcpp
#define BDBT_set_KeySize 0
#define BDBT_set_AreWeInsideStruct 1
#include <BDBT/BDBT.h>
  kt_t kt;
  kt_NodeReference_t kt_root;
  typedef kt_Key_t Key_t;

public: /* -------------------------------------------------------------------------------- */

#pragma pack(push, 1)
  struct KeyInfo_t {
    uint8_t Size;
    Key_t::BitOrder_t BitOrder;
  };
#pragma pack(pop)

  typedef Key_t::BitOrder_t KeyBitOrder_t;
  constexpr static KeyBitOrder_t KeyBitOrderLow = Key_t::BitOrderLow;
  constexpr static KeyBitOrder_t KeyBitOrderHigh = Key_t::BitOrderHigh;
  constexpr static KeyBitOrder_t KeyBitOrderAny = Key_t::BitOrderAny;

private: /* ------------------------------------------------------------------------------- */

#define BLL_set_prefix BlockList
#define BLL_set_Language 1
#define BLL_set_Link 1
#define BLL_set_LinkSentinel 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_type_node blnr_t
#include <WITCH/BLL/BLL.h>
  BlockList_t BlockList;
  /* (RenderDataSize + DataSize + sizeof(ShapeList_t::nr_t)) * (MaxElementPerBlock + 1) */

#pragma pack(push, 1)
  struct bm_BaseData_t {
    BlockList_NodeReference_t FirstBlockNR;
    BlockList_NodeReference_t LastBlockNR;
    uint8_t LastBlockElementCount;
  };
#pragma pack(pop)

#define BLL_set_prefix bm
#define BLL_set_Language 1
#define BLL_set_Link 0
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_type_node ktbmnr_t
#include <WITCH/BLL/BLL.h>
  bm_t bm;
  /* sizeof(bm_BaseData_t) + KeySizesSum */

#pragma pack(push, 1)
  struct shape_t {
    bm_t::nr_t bmid;
    BlockList_t::nr_t blid;
    uint8_t ElementIndex;
  };
#pragma pack(pop)

#define BLL_set_prefix ShapeList
#define BLL_set_Language 1
#define BLL_set_Link 0
#define BLL_set_NodeDataType shape_t
#define BLL_set_AreWeInsideStruct 1
  // actually it needs to be uint24_t
#define BLL_set_type_node uint32_t
#include <WITCH/BLL/BLL.h>
  ShapeList_t ShapeList;

  typedef uint8_t KeyIndex_t;

  KeyIndex_t KeyAmount;
  KeyInfo_t KeyInfos[8];
  uint8_t KeySizesSum;
  uint8_t MaxElementPerBlock; /* (p.MaxElementPerBlock - 1) */
  uint16_t RenderDataSize;
  uint16_t DataSize;

  /* function internal datas */
  struct fid_t {
    union {
      uint8_t* base;
      /* used in remove */
      kt_NodeReference_t* knrs;
      /* used in traverse for store keys */
      uint8_t* KeyDatas;
    };

    Key_t::Traverse_t* tra;
    Key_t::Traverse_t::ta_t** ta;

    void Open(const shaper_t* shaper) {
      base = new uint8_t[
        sizeof(kt_NodeReference_t) * shaper->KeyAmount > shaper->KeySizesSum ?
          sizeof(kt_NodeReference_t) * shaper->KeyAmount :
          shaper->KeySizesSum
      ];

      tra = new Key_t::Traverse_t[shaper->KeyAmount];

      ta = new Key_t::Traverse_t::ta_t * [shaper->KeyAmount];
      for (KeyIndex_t KeyIndex = 0; KeyIndex < shaper->KeyAmount; KeyIndex++) {
        ta[KeyIndex] = new Key_t::Traverse_t::ta_t[
          Key_t::Traverse_t::GetTraverseArraySize(shaper->KeyInfos[KeyIndex].Size * 8)
        ];
      }
    }
    void Close(const shaper_t* shaper) {
      for (KeyIndex_t KeyIndex = 0; KeyIndex < shaper->KeyAmount; KeyIndex++) {
        delete ta[KeyIndex];
      }
      delete ta;

      delete tra;

      delete base;
    }
  }fid;

  /*
    block data format is this. %30
    RenderData   Data         ShapeID
    [|||       ] [|||       ] [|||       ]
  */

  uint8_t* GetRenderData(BlockList_t::nr_t blid, uint8_t ElementIndex) {
    return &((uint8_t*)BlockList[blid])[RenderDataSize * ElementIndex];
  }
  uint8_t* GetData(BlockList_t::nr_t blid, uint8_t ElementIndex) {
    return &GetRenderData(
      blid,
      MaxElementPerBlock + 1
    )[DataSize * ElementIndex];
  }
  ShapeList_t::nr_t& GetShapeID(BlockList_t::nr_t blid, uint8_t ElementIndex) {
    return *(ShapeList_t::nr_t*)&GetData(
      blid,
      MaxElementPerBlock + 1
    )[sizeof(ShapeList_t::nr_t) * ElementIndex];
  }

public: /* -------------------------------------------------------------------------------- */

  struct OpenProperties_t {
    decltype(KeyAmount) KeyAmount;
    std::remove_reference<decltype(KeyInfos[0])>::type* KeyInfos;
    uint16_t MaxElementPerBlock;
    decltype(RenderDataSize) RenderDataSize;
    decltype(DataSize) DataSize;
  };

  void Open(
    const OpenProperties_t p
  ) {
    KeyAmount = p.KeyAmount;
    __MemoryCopy(p.KeyInfos, KeyInfos, KeyAmount * sizeof(KeyInfos[0]));
    if (p.MaxElementPerBlock > 0x100) {
      __abort();
    }
    MaxElementPerBlock = p.MaxElementPerBlock - 1;
    RenderDataSize = p.RenderDataSize;
    DataSize = p.DataSize;

    KeySizesSum = 0;
    for (uintptr_t i = 0; i < KeyAmount; i++) {
      KeySizesSum += KeyInfos[i].Size;
    }

    kt.Open();
    kt_root = kt_NewNode(&kt);
    BlockList.Open(
      (RenderDataSize + DataSize + sizeof(ShapeList_t::nr_t)) * (MaxElementPerBlock + 1)
    );
    bm.Open(sizeof(bm_BaseData_t) + KeySizesSum);
    ShapeList.Open();

    fid.Open(this);
  }
  void Close() {
    fid.Close(this);

    ShapeList.Close();
    bm.Close();
    BlockList.Close();
    kt.Close();
  }

  ShapeList_t::nr_t add(const void* KeyDataArray, const void* RenderData, const void* Data) {
    bm_NodeReference_t bmnr;
    bm_BaseData_t* bmbase;

    auto _KeyDataArray = (uint8_t*)KeyDataArray;

    kt_NodeReference_t nr = kt_root;
    for (KeyIndex_t KeyIndex = 0; KeyIndex < KeyAmount; KeyIndex++) {
      uintptr_t bdbt_ki;
      Key_t::q(&kt, KeyInfos[KeyIndex].Size * 8, _KeyDataArray, &bdbt_ki, &nr);
      if (bdbt_ki != KeyInfos[KeyIndex].Size * 8) {
        /* query failed to find rest so lets make new block manager */

        bmnr = bm.NewNode();
        bmbase = (bm_BaseData_t*)bm[bmnr];
        bmbase->FirstBlockNR = BlockList.NewNode();
        bmbase->LastBlockNR = bmbase->FirstBlockNR;
        bmbase->LastBlockElementCount = 0;
        __MemoryCopy(KeyDataArray, &bmbase[1], KeySizesSum);

        do {
          kt_NodeReference_t out;
          if (KeyIndex + 1 != KeyAmount) {
            out = kt_NewNode(&kt);
          }
          else {
            out = *(kt_NodeReference_t*)&bmnr;
          }

          Key_t::a(&kt, KeyInfos[KeyIndex].Size * 8, _KeyDataArray, bdbt_ki, nr, out);

          nr = out;

          _KeyDataArray += KeyInfos[KeyIndex].Size;
        } while (++KeyIndex < KeyAmount);

        goto gt_NoNewBlockManager;
      }

      _KeyDataArray += KeyInfos[KeyIndex].Size;
    }

    bmnr = *(bm_NodeReference_t*)&nr;
    bmbase = (bm_BaseData_t*)bm[bmnr];

    if (bmbase->LastBlockElementCount == MaxElementPerBlock) {
      bmbase->LastBlockElementCount = 0;
      auto blnr = BlockList.NewNode();
      BlockList.linkNextOfOrphan(bmbase->LastBlockNR, blnr);
      bmbase->LastBlockNR = blnr;
    }

  gt_NoNewBlockManager:

    auto shapeid = ShapeList.NewNode();
    ShapeList[shapeid].bmid = bmnr;
    ShapeList[shapeid].blid = bmbase->LastBlockNR;
    ShapeList[shapeid].ElementIndex = bmbase->LastBlockElementCount;

    __MemoryCopy(
      RenderData,
      GetRenderData(bmbase->LastBlockNR, bmbase->LastBlockElementCount),
      RenderDataSize
    );
    __MemoryCopy(
      Data,
      GetData(bmbase->LastBlockNR, bmbase->LastBlockElementCount),
      DataSize
    );
    GetShapeID(bmbase->LastBlockNR, bmbase->LastBlockElementCount) = shapeid;

    return shapeid;
  }

  void remove(ShapeList_t::nr_t shapeid) {
    bm_t::nr_t bmid;
    bm_BaseData_t* bmbase;
    {
      auto& shape = ShapeList[shapeid];
      bmid = shape.bmid;
      bmbase = (bm_BaseData_t*)bm[bmid];

      auto& lshape = ShapeList[GetShapeID(bmbase->LastBlockNR, bmbase->LastBlockElementCount)];

      __MemoryCopy(
        GetRenderData(lshape.blid, lshape.ElementIndex),
        GetRenderData(shape.blid, shape.ElementIndex),
        RenderDataSize
      );
      __MemoryCopy(
        GetData(lshape.blid, lshape.ElementIndex),
        GetData(shape.blid, shape.ElementIndex),
        DataSize
      );
      GetShapeID(shape.blid, shape.ElementIndex) = GetShapeID(lshape.blid, lshape.ElementIndex);

      lshape.blid = shape.blid;
      lshape.ElementIndex = shape.ElementIndex;

      ShapeList.Recycle(shapeid);
    }

    /* shapeid we deleted may be same as last so lets no longer access shape plus lshape */

    /* we just deleted last so lets check if we can just decrease count */
    if (bmbase->LastBlockElementCount != 0) {
      bmbase->LastBlockElementCount--;
      return;
    }

    /* looks like we deleted first element in block so we need to delete last block */
    /* before do that we need to be sure if that we dont delete first block in block manager */
    if (bmbase->FirstBlockNR != bmbase->LastBlockNR) {
      auto blid = bmbase->LastBlockNR;
      bmbase->LastBlockNR = blid.Prev(&BlockList);
      BlockList.unlrec(blid);
      bmbase->LastBlockElementCount = MaxElementPerBlock;
      return;
    }

    /* good luck we need to remove block manager completely */
    /* aaaaaaaaaaaa */

    BlockList.Recycle(bmbase->LastBlockNR);

    auto KeyDataArray = (uint8_t*)&bmbase[1];
    auto knr = kt_root;

    for (KeyIndex_t KeyIndex = 0; KeyIndex < KeyAmount; KeyIndex++) {
      fid.knrs[KeyIndex] = knr;
      Key_t::cq(&kt, KeyInfos[KeyIndex].Size * 8, KeyDataArray, &knr);
      KeyDataArray += KeyInfos[KeyIndex].Size;
    }

    /* TODO this part can be faster if used some different function instead of .r */
    for (KeyIndex_t KeyIndex = KeyAmount - 1;;) {
      KeyDataArray -= KeyInfos[KeyIndex].Size;
      Key_t::r(&kt, KeyInfos[KeyIndex].Size * 8, KeyDataArray, fid.knrs[KeyIndex]);
      if (kt_inrhc(&kt, fid.knrs[KeyIndex])) {
        break;
      }
      if (KeyIndex == 0) {
        break;
      }
      kt_Recycle(&kt, fid.knrs[KeyIndex]);
      KeyIndex--;
    }

    bm.Recycle(bmid);
  }

  struct tra_t {
    KeyIndex_t KeyIndex;
    uint8_t* KeyData;

    void Init(shaper_t& shaper) {
      KeyIndex = (KeyIndex_t)-1;
      KeyData = shaper.fid.KeyDatas - shaper.KeyInfos->Size;
    }
  };
  bool tra_loop(tra_t& tra) {

    if (++tra.KeyIndex == KeyAmount) {
      --tra.KeyIndex;
    }
    else {
      fid.tra[tra.KeyIndex].i0(
        fid.ta[tra.KeyIndex],
        tra.KeyIndex == 0 ? kt_root : fid.tra[tra.KeyIndex - 1].Output,
        KeyInfos[tra.KeyIndex].BitOrder
      );
      tra.KeyData += KeyInfos[tra.KeyIndex].Size;
    }

  gt_tra:

    bool r = fid.tra[tra.KeyIndex].t0(
      &kt,
      fid.ta[tra.KeyIndex],
      KeyInfos[tra.KeyIndex].Size * 8,
      tra.KeyData,
      KeyInfos[tra.KeyIndex].BitOrder
    );

    if (r == false) {
      if (tra.KeyIndex == 0) {
        return false;
      }
      tra.KeyIndex--;
      tra.KeyData -= KeyInfos[tra.KeyIndex].Size;
      goto gt_tra;
    }

    return true;
  }
  bm_t::nr_t tra_bm() {
    return fid.tra[KeyAmount - 1].Output;
  }
};

#pragma pack(push, 1)
struct ri_t {
  f32_t radius;
  // etc
};
#pragma pack(pop)

constexpr uint8_t KeyAmount = 4;
shaper_t::KeyInfo_t KeyInfos[KeyAmount] = {
  {.Size = 1, .BitOrder = shaper_t::KeyBitOrderHigh},
  {.Size = 2, .BitOrder = shaper_t::KeyBitOrderAny},
  {.Size = 4, .BitOrder = shaper_t::KeyBitOrderLow},
  {.Size = 1, .BitOrder = shaper_t::KeyBitOrderHigh}
};

#pragma pack(push, 1)
struct KeyPack_t {
  uint8_t k0;
  uint16_t k1;
  uint32_t k2;
  uint8_t k3;
};
#pragma pack(pop)

int main() {
  shaper_t shaper;
  shaper.Open({
    .KeyAmount = KeyAmount,
    .KeyInfos = KeyInfos,
    .MaxElementPerBlock = 0x100,
    .RenderDataSize = sizeof(ri_t),
    .DataSize = 0
    });

  KeyPack_t kp;
  kp.k0 = 5;
  kp.k1 = 25;
  kp.k2 = 13;
  kp.k3 = 79;
  ri_t ri;
  auto sid = shaper.add(&kp, &ri, &ri);

  shaper.remove(sid);

  shaper_t::tra_t tra;
  tra.Init(shaper);
  while (shaper.tra_loop(tra)) {
    uint32_t Key;
    if (tra.KeyIndex == 0) { Key = *(uint8_t*)tra.KeyData; }
    if (tra.KeyIndex == 1) { Key = *(uint16_t*)tra.KeyData; }
    if (tra.KeyIndex == 2) { Key = *(uint32_t*)tra.KeyData; }
    if (tra.KeyIndex == 3) { Key = *(uint8_t*)tra.KeyData; }
    printf("salsa %u %u\n", tra.KeyIndex, Key);
  }

  return 0;
}
