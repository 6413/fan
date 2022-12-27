typedef struct{
  /* relative to object */
  __pfloat PositionY;
  __pfloat PositionX;

  __pfloat Size;
}__ETC_BCOL_PP(ShapeData_Square_t);

#define BLL_set_prefix __ETC_BCOL_PP(ShapeList_Square)
#define BLL_set_Language 1
#define BLL_set_NodeReference_Overload_Declare \
  __ETC_BCOL_PP(ShapeList_Square_NodeReference_t)(__ETC_BCOL_P(ShapeID_t) ShapeID){ \
    this->NRI = ShapeID.ID; \
  }
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType __ETC_BCOL_PP(ShapeData_Square_t)
#include _WITCH_PATH(BLL/BLL.h)
