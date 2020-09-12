#include <AL/al.h>
#include <AL/alc.h>

#include <FAN/Vectors.hpp>

class audio {
	constexpr audio() : device(alcOpenDevice(NULL)), context(alcCreateContext(device, NULL)) {
		if (!device) {
			LOG("no sound device");
			exit(1);
		}
		if (!context) {
			LOG("no sound context");
			exit(1);
		}
	}
	ALCdevice* device;
	ALCcontext* context;
};