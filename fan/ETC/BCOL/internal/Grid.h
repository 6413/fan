void Contact_Grid_EnableContact(
  Contact_Grid_t *Contact
){
  Contact->Flag |= Contact_Grid_Flag::EnableContact;
}
void Contact_Grid_DisableContact(
  Contact_Grid_t *Contact
){
  Contact->Flag ^= Contact->Flag & Contact_Grid_Flag::EnableContact;
}
