enum{
  __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e) = 0x01
};

typedef struct{
  uint32_t Flag;
}__ETC_BCOL_P(Contact_Grid_t);

typedef void (*__ETC_BCOL_P(PreSolve_Grid_cb_t))(
  __ETC_BCOL_P(t) *,
  __ETC_BCOL_P(ObjectID_t),
  uint8_t /* ShapeEnum */,
  __ETC_BCOL_P(ShapeID_t),
  sint32_t /* GridY */,
  sint32_t /* GridX */,
  __ETC_BCOL_P(Contact_Grid_t) *
);

#ifdef ETC_BCOL_set_PostSolve_Grid
  typedef struct{
    #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
      __pfloat NormalY;
      __pfloat NormalX;
    #endif
  }__ETC_BCOL_P(ContactResult_Grid_t);

  #ifdef ETC_BCOL_set_PostSolve_Grid_CollisionNormal
    __pfloat
    __ETC_BCOL_P(ContactResult_Grid_GetNormalY)
    (
      __ETC_BCOL_P(ContactResult_Grid_t) *ContactResult
    ){
      return ContactResult->NormalY;
    }
    __pfloat
    __ETC_BCOL_P(ContactResult_Grid_GetNormalX)
    (
      __ETC_BCOL_P(ContactResult_Grid_t) *ContactResult
    ){
      return ContactResult->NormalX;
    }
  #endif

  typedef void (*__ETC_BCOL_P(PostSolve_Grid_cb_t))(
    __ETC_BCOL_P(t) *,
    __ETC_BCOL_P(ObjectID_t),
    uint8_t /* ShapeEnum */,
    __ETC_BCOL_P(ShapeID_t),
    sint32_t /* GridY */,
    sint32_t /* GridX */,
    __ETC_BCOL_P(ContactResult_Grid_t) *
  );
#endif
