#include <cstdio>
#include <iostream>

	#define OFFSETLESS(ptr_m, t_m, d_m) \
		((t_m *)((uint8_t *)(ptr_m) - offsetof(t_m, d_m)))

#define _CONCAT(_0_m, _1_m) _0_m ## _1_m
#define CONCAT(_0_m, _1_m) _CONCAT(_0_m, _1_m)

#ifndef lstd_defastruct
  using lstd_current_type = void;

  #define lstd_defastruct(name, inside) \
    struct CONCAT(name,_t){ \
      using lstd_parent_type = lstd_current_type; \
      using lstd_current_type = CONCAT(name,_t); \
      struct lstd_parent_t{ \
        auto* operator->(){ \
          auto current = OFFSETLESS(this, CONCAT(name,_t), lstd_parent); \
          return OFFSETLESS(current, lstd_parent_type, name); \
        } \
      }lstd_parent; \
      inside \
    }name;
#endif

struct ship_t
{
  using lstd_current_type = ship_t;
  uint32_t salsa;
  struct Fuel_t
  {
    using lstd_parent_type = lstd_current_type;
    using lstd_current_type = Fuel_t;
    struct lstd_parent_t
    {
     inline lstd_parent_type * operator->()
      {
        ship_t::Fuel_t * current = (reinterpret_cast<ship_t::Fuel_t *>((reinterpret_cast<unsigned char *>((this)) - offsetof(ship_t::Fuel_t, lstd_parent))));
        return (reinterpret_cast<ship_t *>((reinterpret_cast<unsigned char *>((current)) - offsetof(ship_t, Fuel))));
      }
      
      // inline constexpr lstd_parent_t() noexcept = default;
    };
    
    struct lstd_parent_t lstd_parent;
    inline void f()
    {
      this->lstd_parent.operator->()->salsa = 5;
    }
    
    uint32_t x;
    // inline constexpr Fuel_t() noexcept = default;
  };
  
  struct Fuel_t Fuel;
  // inline constexpr ship_t() noexcept = default;
};



int main()
{
  ship_t ship = ship_t();
  ship.Fuel.f();
  return 0;
}