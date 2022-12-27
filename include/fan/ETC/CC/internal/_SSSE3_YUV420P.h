#pragma once

#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <ammintrin.h>
#include <immintrin.h>
void _CC_YUV420P_RGB24_2x2(
	uint32_t width,
	uint32_t height,
	const uint32_t *sstride,
	const uint32_t *dstride,
	const uint8_t *const *sdata,
	uint8_t *const *ddata
){
	for(uint32_t iheight = 0; EXPECT(iheight != height, 1); iheight += 2){
		const uint8_t *sdata00 = sdata[0] + (iheight + 0) * sstride[0];
		const uint8_t *sdata01 = sdata[0] + (iheight + 1) * sstride[0];
		const uint8_t *sdata1 = sdata[1] + (iheight / 2) * sstride[1];
		const uint8_t *sdata2 = sdata[2] + (iheight / 2) * sstride[2];
		uint8_t *ddata00 = ddata[0] + (iheight + 0) * dstride[0];
		uint8_t *ddata01 = ddata[0] + (iheight + 1) * dstride[0];
		uint8_t *ddata01_limit = &ddata01[width * 3];
		while(EXPECT(ddata01 != ddata01_limit, 1)){
			sint8_t u = *sdata1++ - 128;
			sint8_t v = *sdata2++ - 128;

			sint16_t c1 = v * 102;
			sint16_t c3 = v * 52;
			sint16_t c2 = -(u * 25 + c3);
			sint16_t c4 = u * 129;

			__m128i sub = _mm_set_epi16(0, c4, c2, c1, 0, c4, c2, c1);

			sint16_t y00, y01, y10, y11;
			y00 = (*sdata00++ - 16) * 75;
			y01 = (*sdata00++ - 16) * 75;
			y10 = (*sdata01++ - 16) * 75;
			y11 = (*sdata01++ - 16) * 75;

			__m128i t;
			__m128i mask = _mm_setr_epi8(
				0x08, 0x0a, 0x0c, 0x00, 0x02, 0x04, 0x0f, 0x0f,
				0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f
			);

			t = _mm_set_epi16(0, y00, y00, y00, 0, y01, y01, y01);
			t = _mm_add_epi16(t, sub);
			t = _mm_srai_epi16(t, 6);
			t = _mm_max_epi16(t, _mm_setzero_si128());
			t = _mm_min_epi16(t, _mm_set_epi16(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff));
			t = _mm_shuffle_epi8(t, mask);
			_mm_storeu_si128((__m128i *)ddata00, t);

			t = _mm_set_epi16(0, y10, y10, y10, 0, y11, y11, y11);
			t = _mm_add_epi16(t, sub);
			t = _mm_srai_epi16(t, 6);
			t = _mm_max_epi16(t, _mm_setzero_si128());
			t = _mm_min_epi16(t, _mm_set_epi16(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff));
			t = _mm_shuffle_epi8(t, mask);
			_mm_storeu_si128((__m128i *)ddata01, t);

			ddata00 += 6;
			ddata01 += 6;
		}
	}
}
