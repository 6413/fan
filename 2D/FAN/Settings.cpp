#include "Settings.hpp"
#include <ctime>
#include <cstdio>

namespace Settings {
	float deltaTime = 0;
}

namespace FanColors {
	Vec3 White = Vec3(1.0, 1.0, 1.0);
	Vec3 Red(1.0, 0.0, 0.0);
	Vec3 Green(0.0, 1.0, 0.0);
	Vec3 Blue(0.0, 0.0, 1.0);
}

namespace entityProperties {
	int amount = 1;
}

void GetFps(){
	static int fps = 0;
	static double start = clock();
	if ((clock() - start) / CLOCKS_PER_SEC > 1.0) {
		printf("%d\n", fps);

		fps = 0;
		start = clock();
	}
	fps++;
}