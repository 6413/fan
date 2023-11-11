struct t0{
  #if defined(hardcode0_t)
    hardcode0_t hardcode0_n;
  #endif
};
struct t1{
  #if defined(hardcode1_t)
    hardcode1_t hardcode1_n;
  #endif
};
struct t2{
  #if defined(hardcode2_t)
  hardcode2_t hardcode2_n;
  #endif
};

struct t3{
  #if defined(hardcode3_t)
  hardcode3_t hardcode3_n;
  #endif
};
struct t4{
  #if defined(hardcode4_t)
  hardcode4_t hardcode4_n;
  #endif
};

using parsed_masterpiece_t = fan::masterpiece_t<
  #if defined(hardcode0_t)
   hardcode0_t
  #endif
  #if defined(hardcode1_t)
  ,hardcode1_t
  #endif
  #if defined(hardcode2_t)
  ,hardcode2_t
  #endif
  #if defined(hardcode3_t)
  ,hardcode3_t
  #endif
  #if defined(hardcode4_t)
  ,hardcode4_t
  #endif
>;

#ifdef hardcode0_t
  #define hardcode0_f auto& CONCAT(get_, hardcode0_n)() { return *key.get_value<0>(); }
#else
  #define hardcode0_f
#endif
#ifdef hardcode1_t
  #define hardcode1_f auto& CONCAT(get_, hardcode1_n)() { return *key.get_value<1>(); }
#else
  #define hardcode1_f
#endif
#ifdef hardcode2_t
  #define hardcode2_f auto& CONCAT(get_, hardcode2_n)() { return *key.get_value<2>(); }
#else
  #define hardcode2_f
#endif
#ifdef hardcode3_t
  #define hardcode3_f auto& CONCAT(get_, hardcode3_n)() { return *key.get_value<3>(); }
#else
  #define hardcode3_f
#endif
#ifdef hardcode4_t
  #define hardcode4_f auto& CONCAT(get_, hardcode4_n)() { return *key.get_value<4>(); }
#else
  #define hardcode4_f
#endif

#define expand_get_functions hardcode0_f hardcode1_f hardcode2_f hardcode3_f hardcode4_f