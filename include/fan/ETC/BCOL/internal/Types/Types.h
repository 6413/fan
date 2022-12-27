struct __ETC_BCOL_P(ShapeID_t){
  uint32_t ID;
  __ETC_BCOL_P(ShapeID_t)(auto);
};

enum{
  __ETC_BCOL_P(ShapeEnum_Circle),
  __ETC_BCOL_P(ShapeEnum_Square),
  __ETC_BCOL_P(ShapeEnum_Rectangle)
};

typedef struct{
  uint32_t ShapeEnum;
  __ETC_BCOL_P(ShapeID_t) ShapeID;
}__ETC_BCOL_PP(ShapeData_t);

typedef struct __ETC_BCOL_P(t);

#include _WITCH_PATH(ETC/BCOL/internal/Types/Object.h)

#if ETC_BCOL_set_SupportGrid == 1
  #include _WITCH_PATH(ETC/BCOL/internal/Types/Grid.h)
#endif

#include _WITCH_PATH(ETC/BCOL/internal/Types/Shape/Circle/Types.h)
#include _WITCH_PATH(ETC/BCOL/internal/Types/Shape/Square/Types.h)
#include _WITCH_PATH(ETC/BCOL/internal/Types/Shape/Rectangle/Types.h)

__ETC_BCOL_P(ShapeID_t)::__ETC_BCOL_P(ShapeID_t)(auto p){
  static_assert(
    __is_type_same<__ETC_BCOL_PP(ShapeList_Circle_NodeReference_t), decltype(p)> ||
    __is_type_same<__ETC_BCOL_PP(ShapeList_Square_NodeReference_t), decltype(p)> ||
    __is_type_same<__ETC_BCOL_PP(ShapeList_Rectangle_NodeReference_t), decltype(p)>);
  this->ID = p.NRI;
}

#if ETC_BCOL_set_DynamicToDynamic == 1
  enum{
    __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e) = 0x01
  };

  typedef struct{
    uint32_t Flag;
  }__ETC_BCOL_P(Contact_Shape_t);

  typedef void (*__ETC_BCOL_P(PreSolve_Shape_cb_t))(
    __ETC_BCOL_P(t) *,
    __ETC_BCOL_P(ObjectID_t),
    uint8_t /* ShapeEnum */,
    __ETC_BCOL_P(ShapeID_t),
    __ETC_BCOL_P(ObjectID_t),
    uint8_t /* ShapeEnum */,
    __ETC_BCOL_P(ShapeID_t),
    __ETC_BCOL_P(Contact_Shape_t) *
  );

  void
  __ETC_BCOL_P(Contact_Shape_EnableContact)
  (
    __ETC_BCOL_P(Contact_Shape_t) *Contact
  ){
    Contact->Flag |= __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e);
  }
  void
  __ETC_BCOL_P(Contact_Shape_DisableContact)
  (
    __ETC_BCOL_P(Contact_Shape_t) *Contact
  ){
    Contact->Flag ^= Contact->Flag & __ETC_BCOL_PP(Contact_Shape_Flag_EnableContact_e);
  }
#endif

struct __ETC_BCOL_P(t){
  __ETC_BCOL_PP(ObjectList_t) ObjectList;
  __ETC_BCOL_PP(ShapeList_Circle_t) ShapeList_Circle;
  __ETC_BCOL_PP(ShapeList_Square_t) ShapeList_Square;
  __ETC_BCOL_PP(ShapeList_Rectangle_t) ShapeList_Rectangle;

  #if ETC_BCOL_set_SupportGrid == 1
    __pfloat GridBlockSize;
    __ETC_BCOL_P(PreSolve_Grid_cb_t) PreSolve_Grid_cb;
  #endif
  #ifdef ETC_BCOL_set_PostSolve_Grid
    __ETC_BCOL_P(PostSolve_Grid_cb_t) PostSolve_Grid_cb;
  #endif

  #if ETC_BCOL_set_DynamicToDynamic == 1
    __ETC_BCOL_P(PreSolve_Shape_cb_t) PreSolve_Shape_cb;
  #endif
  #if ETC_BCOL_set_StepNumber == 1
    uint64_t StepNumber;
  #endif
};
