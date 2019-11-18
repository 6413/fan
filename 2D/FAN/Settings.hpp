#pragma once
#include "Vectors.hpp"

void GetFps();

namespace Settings {
	extern float deltaTime;
}

namespace FanKeys {
	constexpr auto CONSOLE_ENTRY = 268;
	constexpr auto CONSOLE_EXIT = 269;
	constexpr auto UNDERSCORE = '|';
}

namespace FanColors {
	extern Vec3 White;
	extern Vec3 Red;
	extern Vec3 Green;
	extern Vec3 Blue;
}

namespace entityProperties {
	extern int amount;
}

typedef int arrayIndex;