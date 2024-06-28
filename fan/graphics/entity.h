#pragma once

#include <cstdint>

namespace fan {
  namespace graphics {
    typedef uint32_t EntityID_i_t;
    struct EntityID_t {
      EntityID_i_t i;

      bool operator==(const EntityID_t& p) const {
        return i == p.i;
      }

      bool iic() {
        return i == (EntityID_i_t)-1;
      }

      void sic() {
        i = (EntityID_i_t)-1;
      }

      EntityID_t() = default;
      EntityID_t(bool p) {
        sic();
      }
    };

    struct EntityBehaviour_t;
    struct Entity_t;
    struct EntityList_t;

    struct EntityBehaviour_t {
      uint32_t IdentifyingAs;
      typedef void (*cb_force_remove_t)(EntityList_t*, EntityID_t);
      typedef void (*cb_delta_t)(EntityList_t*, EntityID_t, f32_t);
      cb_force_remove_t cb_force_remove;
      cb_delta_t cb_delta;
      union {
        void* UserPTR;
        uint64_t User64;
      };
    };

    struct Entity_t {
      EntityBehaviour_t* Behaviour;
      union {
        void* UserPTR;
        uint64_t User64;
      };
    };

    struct EntityList_t {
      #define BLL_set_prefix bll
      #define BLL_set_Language 1
      #define BLL_set_CPP_ConstructDestruct 1
      #define BLL_set_AreWeInsideStruct 1
      #define BLL_set_SafeNext 1
      #define BLL_set_type_node EntityID_i_t
      #define BLL_set_NodeDataType Entity_t
      #include <BLL/BLL.h>
      bll_t bll;

      Entity_t* Get(EntityID_t EntityID) {
        return &bll[*(bll_NodeReference_t*)&EntityID];
      }
      EntityID_t Add(Entity_t* Entity) {
        auto r = bll.NewNodeLast();
        bll[r] = *Entity;
        return *(EntityID_t*)&r;
      }
      void unlrec(EntityID_t EntityID) {
        bll.unlrec(*(bll_NodeReference_t*)&EntityID);
      }

      void ForceRemove(EntityID_t EntityID) {
        auto Entity = &bll[*(bll_NodeReference_t*)&EntityID];
        Entity->Behaviour->cb_force_remove(this, EntityID);
        unlrec(EntityID);
      }

      // force remove safe
      void frs(EntityID_t EntityID) {
        if (EntityID.iic() == false) {
          ForceRemove(EntityID);
        }
      }

      void Clear() {
        auto nr = bll.GetNodeFirst();
        while (nr != bll.dst) {
          auto Entity = &bll[nr];
          Entity->Behaviour->cb_force_remove(this, *(EntityID_t*)&nr);
          auto tnr = nr;
          nr = nr.Next(&bll);
          bll.unlrec(tnr);
        }
      }

      void Step(f32_t Time) {
        auto nr = bll.GetNodeFirst();
        while (nr != bll.dst) {
          auto Entity = &bll[nr];
          bll.StartSafeNext(nr);
          Entity->Behaviour->cb_delta(this, *(EntityID_t*)&nr, Time);
          nr = bll.EndSafeNext();
        }
      }

      EntityID_t Begin() {
        auto r = bll.GetNodeFirst();
        return *(EntityID_t*)&r;
      }
      EntityID_t End() {
        auto r = bll.dst;
        return *(EntityID_t*)&r;
      }
      EntityID_t Iterate(EntityID_t EntityID) {
        auto r = (*(bll_NodeReference_t*)&EntityID).Next(&bll);
        return *(EntityID_t*)&r;
      }

      EntityList_t() {

      }
      ~EntityList_t() {
        Clear();
      }
    };
  }
}