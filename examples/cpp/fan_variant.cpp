#include <fan/pch.h>
#include <fan/types/variant.h>

fan_variant(test1_t, int x1, double y1, float a1, char b1, bool c1, std::string d1, int8_t e1, int16_t f1, int32_t g1, int64_t h1, uint8_t i1);
fan_variant(test2_t, float x2, double y2, char a2, bool b2, std::string c2, int8_t d2, int16_t e2, int32_t f2, int64_t g2, uint8_t h2, uint16_t i2);
fan_variant(test3_t, std::string x3, int y3, double a3, float b3, char c3, bool d3, int8_t e3, int16_t f3, int32_t g3, int64_t h3, uint8_t i3);
fan_variant(test4_t, double x4, float y4, char a4, bool b4, std::string c4, int8_t d4, int16_t e4, int32_t f4, int64_t g4, uint8_t h4, uint16_t i4);
fan_variant(test5_t, char x5, bool y5, std::string a5, int8_t b5, int16_t c5, int32_t d5, int64_t e5, uint8_t f5, uint16_t g5, uint32_t h5, uint64_t i5);
fan_variant(test6_t, int x6, std::string y6, float a6, char b6, bool c6, int8_t d6, int16_t e6, int32_t f6, int64_t g6, uint8_t h6, uint16_t i6);
fan_variant(test7_t, float x7, int y7, double a7, std::string b7, char c7, bool d7, int8_t e7, int16_t f7, int32_t g7, int64_t h7, uint8_t i7);
fan_variant(test8_t, std::string x8, double y8, float a8, char b8, bool c8, int8_t d8, int16_t e8, int32_t f8, int64_t g8, uint8_t h8, uint16_t i8);
fan_variant(test9_t, double x9, char y9, bool a9, std::string b9, int8_t c9, int16_t d9, int32_t e9, int64_t f9, uint8_t g9, uint16_t h9, uint32_t i9);


int main() {
  //
  test1_t var1;
  test2_t var2;
  test3_t var3;
  test4_t var4;
  test5_t var5;
  test6_t var6;
  test7_t var7;
  test8_t var8;
  test9_t var9;
  return 0;
}