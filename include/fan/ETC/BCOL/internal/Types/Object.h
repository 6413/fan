enum{
  __ETC_BCOL_P(ObjectFlag_Constant) = 0x00000001
};

#if ETC_BCOL_set_StoreExtraDataInsideObject == 1
  typedef struct{
    ETC_BCOL_set_ExtraDataInsideObject
  }__ETC_BCOL_P(ObjectExtraData_t);
#endif

typedef struct{
  uint32_t Flag;

  __pfloat PositionY;
  __pfloat PositionX;

  __pfloat VelocityY;
  __pfloat VelocityX;

  VEC_t ShapeList; /* __ETC_BCOL_PP(ShapeData_t) */

  #if ETC_BCOL_set_StoreExtraDataInsideObject == 1
    __ETC_BCOL_P(ObjectExtraData_t) ExtraData;
  #endif
}__ETC_BCOL_PP(ObjectData_t);

#define BLL_set_prefix __ETC_BCOL_PP(ObjectList)
#define BLL_set_Language 1
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType __ETC_BCOL_PP(ObjectData_t)
#include _WITCH_PATH(BLL/BLL.h)

typedef __ETC_BCOL_PP(ObjectList_NodeReference_t) __ETC_BCOL_P(ObjectID_t);

typedef struct{
  __pfloat PositionY;
  __pfloat PositionX;

  #if ETC_BCOL_set_StoreExtraDataInsideObject == 1
    __ETC_BCOL_P(ObjectExtraData_t) ExtraData;
  #endif
}__ETC_BCOL_P(ObjectProperties_t);
