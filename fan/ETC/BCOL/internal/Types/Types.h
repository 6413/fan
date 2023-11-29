struct __BCOL_P(t){
  constexpr static uintptr_t _dc = BCOL_set_Dimension; /* dimension count */

  typedef CONCAT3(f,BCOL_set_PreferredFloatSize,_t) _f;
  typedef CONCAT3(uint,BCOL_set_PreferredFloatSize,_t) _ui;
  typedef CONCAT3(sint,BCOL_set_PreferredFloatSize,_t) _si;

  template <uintptr_t dc, typename t>
  using _v = fan::vec_wrap_t<dc, t>;

  typedef _v<_dc, _f> _vf;
  typedef _v<_dc, uint32_t> _vui32;
  typedef _v<_dc, sint32_t> _vsi32;
  typedef _v<3, _f> _3f;

  #include "../Math.h"

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
  #include BCOL_Include(BVEC/BVEC.h)

  #include "Shape/Circle/Types.h"
  #include "Shape/Rectangle/Types.h"

  #include "Object.h"

  struct ShapeInfoPack_t{
    ObjectID_t ObjectID;
    ShapeEnum_t ShapeEnum;
    ShapeID_t ShapeID;
  };

  #if BCOL_set_VisualSolve == 1
    struct VisualSolve_t{
      _vf at;
      _vf normal;
      _f multipler;
      _3f rgb;
      _f transparency;
      _f reflect;
      VisualSolve_t(){}
      VisualSolve_t(uint32_t p){
        at = _vf(0);
        normal = _vf(0);
        multipler = 0;
        rgb = 0;
        transparency = 0;
        reflect = 0;
      }
    };
  #endif

  #if BCOL_set_SupportGrid == 1
    #include "Grid.h"
  #endif

  ShapeList_Circle_t ShapeList_Circle;
  ShapeList_Rectangle_t ShapeList_Rectangle;

  #if BCOL_set_DynamicToDynamic == 1
    struct Contact_Shape_Flag{
      constexpr static uint32_t EnableContact = 0x01;
    };

    typedef void (*PreSolveAfter_Shape_cb_t)(
      __BCOL_P(t) *,
      const ShapeInfoPack_t *,
      const ShapeInfoPack_t *
    );

    struct Contact_Shape_t{
      uint32_t Flag = Contact_Shape_Flag::EnableContact;
      PreSolveAfter_Shape_cb_t AfterCB
      #if BCOL_set_HaveDefaultCB == 1
        = [](
          __BCOL_P(t) *,
          const ShapeInfoPack_t *,
          const ShapeInfoPack_t *
        ){}
      #endif
      ;
    };

    typedef void (*PreSolve_Shape_cb_t)(
      __BCOL_P(t) *,
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

    PreSolve_Shape_cb_t PreSolve_Shape_cb
    #if BCOL_set_HaveDefaultCB == 1
      = [](
        __BCOL_P(t) *,
        const ShapeInfoPack_t *,
        const ShapeInfoPack_t *,
        Contact_Shape_t *
      ){}
    #endif
    ;
  #endif

  #if BCOL_set_SupportGrid == 1
    _f GridBlockSize;
    PreSolve_Grid_cb_t PreSolve_Grid_cb
    #if BCOL_set_HaveDefaultCB == 1
      = [](
        __BCOL_P(t) *,
        const ShapeInfoPack_t *,
        _vsi32 /* Grid */,
        Contact_Grid_t *
      ){}
    #endif
    ;
  #endif
  #if BCOL_set_VisualSolve == 1
    typedef void (*VisualSolve_Shape_cb_t)(
      __BCOL_P(t) *,
      const ShapeInfoPack_t *,
      _vf, /* ray source */
      _vf, /* ray at */
      VisualSolve_t *
    );
    VisualSolve_Shape_cb_t VisualSolve_Shape_cb
    #if BCOL_set_HaveDefaultCB == 1
      = [](
        __BCOL_P(t) *,
        const ShapeInfoPack_t *,
        _vf,
        _vf,
        VisualSolve_t *
      ){}
    #endif
    ;
    #if BCOL_set_SupportGrid == 1
      VisualSolve_Grid_cb_t VisualSolve_Grid_cb
      #if BCOL_set_HaveDefaultCB == 1
        = [](
          __BCOL_P(t) *,
          _vsi32,
          _vf,
          _vf,
          VisualSolve_t *
        ){}
      #endif
      ;
    #endif
  #endif
  #ifdef BCOL_set_PostSolve_Grid
    PostSolve_Grid_cb_t PostSolve_Grid_cb
    #if BCOL_set_HaveDefaultCB == 1
      = [](
        __BCOL_P(t) *,
        const ShapeInfoPack_t *,
        _vsi32 /* Grid */,
        ContactResult_Grid_t *
      ){}
    #endif
    ;
  #endif

  #if BCOL_set_StepNumber == 1
    uint64_t StepNumber;
  #endif

  #include "../Collision/Collision.h"
  #include "../Object.h"
  #include "../BaseFunctions.h"
  #include "../Traverse.h"
  #include "../Shape/Shape.h"
  #include "../ObjectShape.h"
  #if BCOL_set_SupportGrid == 1
    #include "../Grid.h"
  #endif
  #include "../Step/Step.h"
  #include "../CompiledShapes.h"
  #include "../ImportHM.h"
  #include "../Ray.h"
};
