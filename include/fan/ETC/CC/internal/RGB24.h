#pragma once

void _CC_RGB24_YUV420P_2x2(
	uint32_t width,
	uint32_t height,
	const uint32_t *sstride,
	const uint32_t *dstride,
	const uint8_t *const *sdata,
	uint8_t *const *ddata
){
	for(uint32_t iheight = 0; EXPECT(iheight < height, 1); iheight += 2){
		{
			const uint8_t *sdata00 = sdata[0] + (iheight + 0) * sstride[0];
			const uint8_t *sdata01 = sdata[0] + (iheight + 1) * sstride[0];
			uint8_t *ddata00 = ddata[0] + (iheight + 0) * dstride[0];
			uint8_t *ddata01 = ddata[0] + (iheight + 1) * dstride[0];
			uint8_t *ddata01_limit = &ddata01[width];
			while(EXPECT(ddata01 != ddata01_limit, 1)){
				*ddata00++ = ((66 * sdata00[0] + 129 * sdata00[1] + 25 * sdata00[2]) >> 8) + 16;
				*ddata01++ = ((66 * sdata01[0] + 129 * sdata01[1] + 25 * sdata01[2]) >> 8) + 16;
				sdata01 += 3;
				sdata00 += 3;
			}
		}
		{
			const uint8_t *sdata00 = sdata[0] + (iheight + 0) * sstride[0];
			const uint8_t *sdata01 = sdata[0] + (iheight + 1) * sstride[0];
			uint8_t *ddata10 = ddata[1] + (iheight / 2) * dstride[1];
			uint8_t *ddata20 = ddata[2] + (iheight / 2) * dstride[2];
			uint8_t *ddata20_limit = &ddata20[width / 2];
			while(EXPECT(ddata20 != ddata20_limit, 1)){
				uint16_t rs = 0;
				uint16_t gs = 0;
				uint16_t bs = 0;
				for(uint8_t i2 = 0; i2 < 2; i2++){
					rs += *sdata00++;
					gs += *sdata00++;
					bs += *sdata00++;
					rs += *sdata01++;
					gs += *sdata01++;
					bs += *sdata01++;
				}
				rs >>= 2;
				gs >>= 2;
				bs >>= 2;
				*ddata10++ = ((-38 * rs + -74 * gs + 112 * bs) >> 8) + 128;
				*ddata20++ = ((112 * rs + -94 * gs + -18 * bs) >> 8) + 128;
			}
		}
	}
}

void _CC_RGB24_NV12_2x2(
	uint32_t width,
	uint32_t height,
	const uint32_t *sstride,
	const uint32_t *dstride,
	const uint8_t *const *sdata,
	uint8_t *const *ddata
){
	for(uint32_t iheight = 0; EXPECT(iheight < height, 1); iheight += 2){
		{
			const uint8_t *sdata00 = sdata[0] + (iheight + 0) * sstride[0];
			const uint8_t *sdata01 = sdata[0] + (iheight + 1) * sstride[0];
			uint8_t *ddata00 = ddata[0] + (iheight + 0) * dstride[0];
			uint8_t *ddata01 = ddata[0] + (iheight + 1) * dstride[0];
			uint8_t *ddata01_limit = &ddata01[width];
			while(EXPECT(ddata01 != ddata01_limit, 1)){
				*ddata00++ = ((66 * sdata00[0] + 129 * sdata00[1] + 25 * sdata00[2]) >> 8) + 16;
				*ddata01++ = ((66 * sdata01[0] + 129 * sdata01[1] + 25 * sdata01[2]) >> 8) + 16;
				sdata01 += 3;
				sdata00 += 3;
			}
		}
		{
			const uint8_t *sdata00 = sdata[0] + (iheight + 0) * sstride[0];
			const uint8_t *sdata01 = sdata[0] + (iheight + 1) * sstride[0];
			uint8_t *ddata10 = ddata[1] + (iheight / 2) * dstride[1];
			uint8_t *ddata10_limit = &ddata10[width];
			while(EXPECT(ddata10 != ddata10_limit, 1)){
				uint16_t rs = 0;
				uint16_t gs = 0;
				uint16_t bs = 0;
				for(uint8_t i2 = 0; i2 < 2; i2++){
					rs += *sdata00++;
					gs += *sdata00++;
					bs += *sdata00++;
					rs += *sdata01++;
					gs += *sdata01++;
					bs += *sdata01++;
				}
				rs >>= 2;
				gs >>= 2;
				bs >>= 2;
				*ddata10++ = ((-38 * rs + -74 * gs + 112 * bs) >> 8) + 128;
				*ddata10++ = ((112 * rs + -94 * gs + -18 * bs) >> 8) + 128;
			}
		}
	}
}
