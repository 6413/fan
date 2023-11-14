/* objects */

struct TraverseObjects_t{
  ObjectID_t ObjectID;
  struct{
    ObjectID_t NextObjectID;
  }priv;
};

void TraverseObjects_init(
  TraverseObjects_t *TraverseObjects
){
  TraverseObjects->ObjectID = this->ObjectList.src;
  auto ObjectNode = this->ObjectList.GetNodeByReference(TraverseObjects->ObjectID);
  TraverseObjects->priv.NextObjectID = ObjectNode->NextNodeReference;
}

bool TraverseObjects_loop(
  TraverseObjects_t *TraverseObjects
){
  if(TraverseObjects->priv.NextObjectID == this->ObjectList.dst){
    return false;
  }
  TraverseObjects->ObjectID = TraverseObjects->priv.NextObjectID;
  auto ObjectNode = this->ObjectList.GetNodeByReference(TraverseObjects->ObjectID);
  TraverseObjects->priv.NextObjectID = ObjectNode->NextNodeReference;
  return true;
}
