struct ShapeData_Rectangle_t{
  /* relative to object */
  _vf Position;

  _vf Size;
};

#define BLL_set_prefix ShapeList_Rectangle
#define BLL_set_Language 1
#define BLL_set_AreWeInsideStruct 1
#define BLL_set_NodeReference_Overload_Declare \
  ShapeList_Rectangle_NodeReference_t(ShapeID_t ShapeID){ \
    this->NRI = ShapeID.ID; \
  }
#define BLL_set_type_node uint32_t
#define BLL_set_NodeDataType ShapeData_Rectangle_t
#include ETC_BCOL_Include(BLL/BLL.h)
