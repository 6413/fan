struct t0{
  #if defined(hardcore0_t)
    hardcore0_t hardcore0_n;
  #endif
};
struct t1{
  #if defined(hardcore1_t)
    hardcore1_t hardcore1_n;
  #endif
};
struct t2{
  #if defined(hardcore2_t)
  hardcore2_t hardcore2_n;
  #endif
};

struct t3{
  #if defined(hardcore3_t)
  hardcore3_t hardcore3_n;
  #endif
};
struct t4{
  #if defined(hardcore4_t)
  hardcore4_t hardcore4_n;
  #endif
};

using parsed_masterpiece_t = fan::masterpiece_t<
  #if defined(hardcore0_t)
   hardcore0_t
  #endif
  #if defined(hardcore1_t)
  ,hardcore1_t
  #endif
  #if defined(hardcore2_t)
  ,hardcore2_t
  #endif
  #if defined(hardcore3_t)
  ,hardcore3_t
  #endif
  #if defined(hardcore4_t)
  ,hardcore4_t
  #endif
>;

#ifndef hardcore0_t
  #define hardcore0_t
  #define hardcore0_n
#endif
#ifndef hardcore1_t
  #define hardcore1_t
  #define hardcore1_n
#endif
#ifndef hardcore2_t
  #define hardcore2_t
  #define hardcore2_n
#endif
#ifndef hardcore3_t
  #define hardcore3_t
  #define hardcore3_n
#endif
#ifndef hardcore4_t
  #define hardcore4_t
  #define hardcore4_n
#endif

#define expand_all hardcore0_t hardcore0_n; hardcore1_t hardcore1_n; hardcore2_t hardcore2_n; hardcore3_t hardcore3_n; hardcore4_t hardcore4_n;