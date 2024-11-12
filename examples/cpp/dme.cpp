#include <fan/types/dme.h>

#define DME_VERIFY_PRINT 1

#if DME_VERIFY_PRINT == 1
#include <stdio.h>
#endif

#define VERIFY_DME 1

#if VERIFY_DME == 1
#include <cassert>
#endif


#pragma pack(push, 1)

struct data_we_want_t {
  int number;
};

struct dme_t : dme_inherit__(dme_t, data_we_want_t) {
  __dme2(a, {.number = 5});
  __dme2(b, {.number = 5});
  __dme2(c, {.number = 5});
  __dme2(d, {.number = 5});
  __dme2(e, {.number = 5});
  __dme2(f, {.number = 5});
  __dme2(g, {.number = 5});
  __dme2(h, {.number = 5});
  __dme2(i, {.number = 5});
  __dme2(j, {.number = 5});
  __dme2(k, {.number = 5});
  __dme2(l, {.number = 5});
  __dme2(m, {.number = 5});
  __dme2(n, {.number = 5});
  __dme2(o, {.number = 5});
  __dme2(p, {.number = 5});
  __dme2(q, {.number = 5});
  __dme2(r, {.number = 5});
  __dme2(s, {.number = 5});
  __dme2(t, {.number = 5});
  __dme2(u, {.number = 5});
  __dme2(v, {.number = 5});
  __dme2(w, {.number = 5});
  __dme2(x, {.number = 5});
  __dme2(y, {.number = 5});
  __dme2(z, {.number = 5});
  __dme2(a1, {.number = 5});
  __dme2(b1, {.number = 5});
  __dme2(c1, {.number = 5});
  __dme2(d1, {.number = 5});
  __dme2(e1, {.number = 5});
  __dme2(f1, {.number = 5});
  __dme2(g1, {.number = 5});
  __dme2(h1, {.number = 5});
  __dme2(i1, {.number = 5});
  __dme2(j1, {.number = 5});
  __dme2(k1, {.number = 5});
  __dme2(l1, {.number = 5});
  __dme2(m1, {.number = 5});
  __dme2(n1, {.number = 5});
  __dme2(o1, {.number = 5});
  __dme2(p1, {.number = 5});
  __dme2(q1, {.number = 5});
  __dme2(r1, {.number = 5});
  __dme2(s1, {.number = 5});
  __dme2(t1, {.number = 5});
  __dme2(u1, {.number = 5});
  __dme2(v1, {.number = 5});
  __dme2(w1, {.number = 5});
  __dme2(x1, {.number = 5});
  __dme2(y1, {.number = 5});
  __dme2(z1, {.number = 5});
  __dme2(a2, {.number = 5});
  __dme2(b2, {.number = 5});
  __dme2(c2, {.number = 5});
  __dme2(d2, {.number = 5});
  __dme2(e2, {.number = 5});
  __dme2(f2, {.number = 5});
  __dme2(g2, {.number = 5});
  __dme2(h2, {.number = 5});
  __dme2(i2, {.number = 5});
  __dme2(j2, {.number = 5});
  __dme2(k2, {.number = 5});
  __dme2(l2, {.number = 5});
  __dme2(m2, {.number = 5});
  __dme2(n2, {.number = 5});
  __dme2(o2, {.number = 5});
  __dme2(p2, {.number = 5});
  __dme2(q2, {.number = 5});
  __dme2(r2, {.number = 5});
  __dme2(s2, {.number = 5});
  __dme2(t2, {.number = 5});
  __dme2(u2, {.number = 5});
  __dme2(v2, {.number = 5});
  __dme2(w2, {.number = 5});
  __dme2(x2, {.number = 5});
  __dme2(y2, {.number = 5});
  __dme2(z2, {.number = 5});
  __dme2(a3, {.number = 5});
  __dme2(b3, {.number = 5});
  __dme2(c3, {.number = 5});
  __dme2(d3, {.number = 5});
  __dme2(e3, {.number = 5});
  __dme2(f3, {.number = 5});
  __dme2(g3, {.number = 5});
  __dme2(h3, {.number = 5});
  __dme2(i3, {.number = 5});
  __dme2(j3, {.number = 5});
  __dme2(k3, {.number = 5});
  __dme2(l3, {.number = 5});
  __dme2(m3, {.number = 5});
  __dme2(n3, {.number = 5});
  __dme2(o3, {.number = 5});
  __dme2(p3, {.number = 5});
  __dme2(q3, {.number = 5});
  __dme2(r3, {.number = 5});
  __dme2(s3, {.number = 5});
  __dme2(t3, {.number = 5});
  __dme2(u3, {.number = 5});
  __dme2(v3, {.number = 5});
  __dme2(w3, {.number = 5});
  __dme2(x3, {.number = 5});
  __dme2(y3, {.number = 5});
  __dme2(z3, {.number = 5});
  __dme2(a4, {.number = 5});
  __dme2(b4, {.number = 5});
  __dme2(c4, {.number = 5});
  __dme2(d4, {.number = 5});
  __dme2(e4, {.number = 5});
  __dme2(f4, {.number = 5});
  __dme2(g4, {.number = 5});
  __dme2(h4, {.number = 5});
  __dme2(i4, {.number = 5});
  __dme2(j4, {.number = 5});
  __dme2(k4, {.number = 5});
  __dme2(l4, {.number = 5});
  __dme2(m4, {.number = 5});
  __dme2(n4, {.number = 5});
  __dme2(o4, {.number = 5});
  __dme2(p4, {.number = 5});
  __dme2(q4, {.number = 5});
  __dme2(r4, {.number = 5});
  __dme2(s4, {.number = 5});
  __dme2(t4, {.number = 5});
  __dme2(u4, {.number = 5});
  __dme2(v4, {.number = 5});
  __dme2(w4, {.number = 5});
  __dme2(x4, {.number = 5});
  __dme2(y4, {.number = 5});
  __dme2(z4, {.number = 5});
  __dme2(a5, {.number = 5});
  __dme2(b5, {.number = 5});
  __dme2(c5, {.number = 5});
  __dme2(d5, {.number = 5});
  __dme2(e5, {.number = 5});
  __dme2(f5, {.number = 5});
  __dme2(g5, {.number = 5});
  __dme2(h5, {.number = 5});
  __dme2(i5, {.number = 5});
  __dme2(j5, {.number = 5});
  __dme2(k5, {.number = 5});
  __dme2(l5, {.number = 5});
  __dme2(m5, {.number = 5});
  __dme2(n5, {.number = 5});
  __dme2(o5, {.number = 5});
  __dme2(p5, {.number = 5});
  __dme2(q5, {.number = 5});
  __dme2(r5, {.number = 5});
  __dme2(s5, {.number = 5});
  __dme2(t5, {.number = 5});
  __dme2(u5, {.number = 5});
  __dme2(v5, {.number = 5});
  __dme2(w5, {.number = 5});
  __dme2(x5, {.number = 5});
  __dme2(y5, {.number = 5});
  __dme2(z5, {.number = 5});
  __dme2(a6, {.number = 5});
  __dme2(b6, {.number = 5});
  __dme2(c6, {.number = 5});
  __dme2(d6, {.number = 5});
  __dme2(e6, {.number = 5});
  __dme2(f6, {.number = 5});
  __dme2(g6, {.number = 5});
  __dme2(h6, {.number = 5});
  __dme2(i6, {.number = 5});
  __dme2(j6, {.number = 5});
  __dme2(k6, {.number = 5});
  __dme2(l6, {.number = 5});
  __dme2(m6, {.number = 5});
  __dme2(n6, {.number = 5});
  __dme2(o6, {.number = 5});
  __dme2(p6, {.number = 5});
  __dme2(q6, {.number = 5});
  __dme2(r6, {.number = 5});
  __dme2(s6, {.number = 5});
  __dme2(t6, {.number = 5});
  __dme2(u6, {.number = 5});
  __dme2(v6, {.number = 5});
  __dme2(w6, {.number = 5});
  __dme2(x6, {.number = 5});
  __dme2(y6, {.number = 5});
  __dme2(z6, {.number = 5});
  __dme2(a7, {.number = 5});
  __dme2(b7, {.number = 5});
  __dme2(c7, {.number = 5});
  __dme2(d7, {.number = 5});
  __dme2(e7, {.number = 5});
  __dme2(f7, {.number = 5});
  __dme2(g7, {.number = 5});
  __dme2(h7, {.number = 5});
  __dme2(i7, {.number = 5});
  __dme2(j7, {.number = 5});
  __dme2(k7, {.number = 5});
  __dme2(l7, {.number = 5});
  __dme2(m7, {.number = 5});
  __dme2(n7, {.number = 5});
  __dme2(o7, {.number = 5});
  __dme2(p7, {.number = 5});
  __dme2(q7, {.number = 5});
  __dme2(r7, {.number = 5});
  __dme2(s7, {.number = 5});
  __dme2(t7, {.number = 5});
  __dme2(u7, {.number = 5});
  __dme2(v7, {.number = 5});
  __dme2(w7, {.number = 5});
  __dme2(x7, {.number = 5});
  __dme2(y7, {.number = 5});
  __dme2(z7, {.number = 5});
  __dme2(a8, {.number = 5});
  __dme2(b8, {.number = 5});
  __dme2(c8, {.number = 5});
  __dme2(d8, {.number = 5});
  __dme2(e8, {.number = 5});
  __dme2(f8, {.number = 5});
  __dme2(g8, {.number = 5});
  __dme2(h8, {.number = 5});
  __dme2(i8, {.number = 5});
  __dme2(j8, {.number = 5});
  __dme2(k8, {.number = 5});
  __dme2(l8, {.number = 5});
  __dme2(m8, {.number = 5});
  __dme2(n8, {.number = 5});
  __dme2(o8, {.number = 5});
  __dme2(p8, {.number = 5});
  __dme2(q8, {.number = 5});
  __dme2(r8, {.number = 5});
  __dme2(s8, {.number = 5});
  __dme2(t8, {.number = 5});
  __dme2(u8, {.number = 5});
  __dme2(v8, {.number = 5});
  __dme2(w8, {.number = 5});
  __dme2(x8, {.number = 5});
  __dme2(y8, {.number = 5});
  __dme2(z8, {.number = 5});
};

struct dme2_t : dme_inherit__(dme2_t, data_we_want_t) {
  __dme2(a, {.number = 10});
  __dme2(b, {.number = 10 });
  __dme2(c, {.number = 10 });
  __dme2_data(d, {.number = 10 }, int x;);
};

#pragma pack(pop)

int main() {

  dme2_t c;
  c.b.AN();
  c.GetMemberAmount();

  dme_t a;
  dme_t b;

  //
#if DME_VERIFY_PRINT == 1
  printf("%llu %llu %llu %llu %d %d", a.a.I, b.a.I, b.z8.I, a.b.I, (int)a.z8, (int)c.c);
#endif

#if VERIFY_DME == 1
  assert(a.a.I == 0);
  assert(b.a.I == 0);
  assert((int)b.z8 == 233);
  assert((int)c.c == 2 && c.c.I == 2);
  assert((void*)c.NA(2) == (void*)&c.c);
  assert(c.GetMemberAmount() == sizeof(c) / sizeof(decltype(c)::dme_type_t));
  //decltype(c.d)::dt::;

  //printf("%s\n", typeid(std::tuple_element_t<0, std::tuple<decltype(a.b)::a_t>>).name());
#endif

  return a.a.I + b.a.I + b.z8 + a.b + a.z8 + c.c;
}
