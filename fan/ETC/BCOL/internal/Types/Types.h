struct __ETC_BCOL_P(t){
  typedef CONCAT3(f,ETC_BCOL_set_PreferredFloatSize,_t) _f;
  typedef CONCAT3(uint,ETC_BCOL_set_PreferredFloatSize,_t) _ui;
  typedef CONCAT3(sint,ETC_BCOL_set_PreferredFloatSize,_t) _si;
  typedef fan::vec_wrap_t<ETC_BCOL_set_Dimension, _f> _vf;
  typedef fan::vec_wrap_t<ETC_BCOL_set_Dimension, uint32_t> _vui32;
  typedef fan::vec_wrap_t<ETC_BCOL_set_Dimension, sint32_t> _vsi32;

  typedef uint32_t ShapeID_ID_t;

  struct ShapeID_t{
    ShapeID_ID_t ID;

    ShapeID_t() = default;
    ShapeID_t(auto p){
      static_assert(
        __is_type_same<ShapeList_Circle_NodeReference_t, decltype(p)> ||
        __is_type_same<ShapeList_Rectangle_NodeReference_t, decltype(p)>);
      this->ID = p.NRI;
    }
  };

  enum class ShapeEnum_t : uint8_t{
    Circle,
    Rectangle
  };

  struct ShapeData_t{
    ShapeEnum_t ShapeEnum;

    /* TODO why we have ShapeID here? */
    ShapeID_t ShapeID;
  };

  #define BVEC_set_prefix ShapeList
  #define BVEC_set_NodeType uint32_t
  #define BVEC_set_NodeData ShapeData_t
  #include ETC_BCOL_Include(BVEC/BVEC.h)

  #include "Shape/Circle/Types.h"
  #include "Shape/Rectangle/Types.h"

  #include "Object.h"

  struct ShapeInfoPack_t{
    ObjectID_t ObjectID;
    ShapeEnum_t ShapeEnum;
    ShapeID_t ShapeID;
  };

  #if ETC_BCOL_set_SupportGrid == 1
    #include "Grid.h"
  #endif

  ShapeList_Circle_t ShapeList_Circle;
  ShapeList_Rectangle_t ShapeList_Rectangle;

  #if ETC_BCOL_set_DynamicToDynamic == 1
    struct Contact_Shape_Flag{
      constexpr static uint32_t EnableContact = 0x01;
    };

    typedef void (*PreSolveAfter_Shape_cb_t)(
      __ETC_BCOL_P(t) *,
      const ShapeInfoPack_t *,
      const ShapeInfoPack_t *
    );

    struct Contact_Shape_t{
      uint32_t Flag = Contact_Shape_Flag::EnableContact;
      PreSolveAfter_Shape_cb_t AfterCB = [](
        __ETC_BCOL_P(t) *,
        const ShapeInfoPack_t *,
        const ShapeInfoPack_t *
      ){};
    };

    typedef void (*PreSolve_Shape_cb_t)(
      __ETC_BCOL_P(t) *,
      const ShapeInfoPack_t *,
      const ShapeInfoPack_t *,
      Contact_Shape_t *
    );

    void Contact_Shape_EnableContact(
      Contact_Shape_t *Contact
    ){
      Contact->Flag |= Contact_Shape_Flag::EnableContact;
    }
    void Contact_Shape_DisableContact
    (
      Contact_Shape_t *Contact
    ){
      Contact->Flag ^= Contact->Flag & Contact_Shape_Flag::EnableContact;
    }

    PreSolve_Shape_cb_t PreSolve_Shape_cb;
  #endif

  #if ETC_BCOL_set_SupportGrid == 1
    _f GridBlockSize;
    PreSolve_Grid_cb_t PreSolve_Grid_cb;
  #endif
  #ifdef ETC_BCOL_set_PostSolve_Grid
    PostSolve_Grid_cb_t PostSolve_Grid_cb;
  #endif

  #if ETC_BCOL_set_StepNumber == 1
    uint64_t StepNumber;
  #endif

  #include "../Collision/Collision.h"
  #include "../Object.h"
  #include "../BaseFunctions.h"
  #include "../Traverse.h"
  #include "../Shape/Shape.h"
  #include "../ObjectShape.h"
  #if ETC_BCOL_set_SupportGrid == 1
    #include "../Grid.h"
  #endif
  #include "../Step/Step.h"
  #include "../CompiledShapes.h"
  #include "../ImportHM.h"
};
