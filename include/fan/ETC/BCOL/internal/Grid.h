void
__ETC_BCOL_P(Contact_Grid_EnableContact)
(
  __ETC_BCOL_P(Contact_Grid_t) *Contact
){
  Contact->Flag |= __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e);
}
void
__ETC_BCOL_P(Contact_Grid_DisableContact)
(
  __ETC_BCOL_P(Contact_Grid_t) *Contact
){
  Contact->Flag ^= Contact->Flag & __ETC_BCOL_PP(Contact_Grid_Flag_EnableContact_e);
}
