struct Contact_Grid_Flag{
  constexpr static uint32_t EnableContact = 0x01;
};

struct Contact_Grid_t{
  uint32_t Flag = Contact_Grid_Flag::EnableContact;
};

typedef void (*PreSolve_Grid_cb_t)(
  __BCOL_P(t) *,
  const ShapeInfoPack_t *,
  _vsi32 /* Grid */,
  Contact_Grid_t *
);

#if BCOL_set_VisualSolve == 1
  typedef void (*VisualSolve_Grid_cb_t)(
    __BCOL_P(t) *,
    _vsi32, /* grid index */
    _vf, /* ray source, normalized */
    _vf, /* ray at, normalized */
    VisualSolve_t *
  );
#endif

#ifdef BCOL_set_PostSolve_Grid
  struct ContactResult_Grid_t{
    #ifdef BCOL_set_PostSolve_Grid_CollisionNormal
      _vf Normal;
    #endif
  };

  #ifdef BCOL_set_PostSolve_Grid_CollisionNormal
    _vf ContactResult_Grid_GetNormal(
      ContactResult_Grid_t *ContactResult
    ){
      return ContactResult->Normal;
    }
  #endif

  typedef void (*PostSolve_Grid_cb_t)(
    __BCOL_P(t) *,
    const ShapeInfoPack_t *,
    _vsi32 /* Grid */,
    ContactResult_Grid_t *
  );
#endif
