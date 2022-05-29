#pragma once

#include <opus/opusfile.h>

#include _FAN_PATH(types/types.h)

#ifdef fan_compiler_visual_studio

#endif

#ifndef fan_audio_set_backend
	#if defined(fan_platform_unix)
		#if defined(fan_platform_linux)
			#define fan_audio_set_backend 1
		#else
			#define fan_audio_set_backend 0
		#endif
	#elif defined(fan_platform_windows)
		#define fan_audio_set_backend 0
	#else
		#error ?
	#endif
#endif

namespace fan {
	namespace audio {
		namespace { namespace constants {
			const uint32_t opus_decode_sample_rate = 48000;

			const f32_t OneSampleTime = (f32_t)1 / opus_decode_sample_rate;

			const uint32_t CallFrameCount = 480;
			const uint32_t ChannelAmount = 2;
			const uint32_t FrameCacheAmount = 4800;
			const uint64_t FrameCacheTime = opus_decode_sample_rate / CallFrameCount * 1; // 1 second
		}}

		#if fan_audio_set_backend == 0
			#include "backend/uni/miniaudio/a.h"
		#elif fan_audio_set_backend == 1
			#include "backend/unix/linux/alsa/a.h"
		#else
			#error ?
		#endif
	}
}
