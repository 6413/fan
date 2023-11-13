struct ObjectFlag{
  constexpr static uint32_t Constant = 0x00000001;
};

#if ETC_BCOL_set_StoreExtraDataInsideObject == 1
  struct ObjectExtraData_t{
    ETC_BCOL_set_ExtraDataInsideObject
  };
#endif

struct ObjectData_t{
  uint32_t Flag;

  _vf Position;

  _vf Velocity;

  ShapeList_t ShapeList;

  #if ETC_BCOL_set_StoreExtraDataInsideObject == 1
    ObjectExtraData_t ExtraData;
  #endif
};

#define BLL_set_prefix ObjectList
#define BLL_set_SafeNext 2
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_Language 1
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType ObjectData_t
#include ETC_BCOL_Include(BLL/BLL.h)

typedef ObjectList_NodeReference_t ObjectID_t;

struct ObjectProperties_t{
  _vf Position;

  #if ETC_BCOL_set_StoreExtraDataInsideObject == 1
    ObjectExtraData_t ExtraData;
  #endif
};

ObjectList_t ObjectList;
