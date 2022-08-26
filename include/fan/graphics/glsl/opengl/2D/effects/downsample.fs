R"(
	#version 330 core

	layout (location = 0) out vec3 o_color;

	in vec2 texture_coordinate;

	uniform sampler2D _t00;

	uniform vec2 resolution;

	vec3 PowVec3(vec3 v, float p)
	{
			return vec3(pow(v.x, p), pow(v.y, p), pow(v.z, p));
	}

	const float invGamma = 1.0 / 2.2;
	vec3 ToSRGB(vec3 v)   { return PowVec3(v, invGamma); }

	float sRGBToLuma(vec3 col)
	{
			//return dot(col, vec3(0.2126f, 0.7152f, 0.0722f));
		return dot(col, vec3(0.299f, 0.587f, 0.114f));
	}

	float KarisAverage(vec3 col)
	{
		// Formula is 1 / (1 + luma)
		float luma = sRGBToLuma(ToSRGB(col)) * 0.25f;
		return 1.0f / (1.0f + luma);
	}

	void main() {

		float x = 1.0 / resolution.x;
		float y = 1.0 / resolution.y;
		
		// Take 13 samples around current texel:
		// a - b - c
		// - j - k -
		// d - e - f
		// - l - m -
		// g - h - i
		// === ('e' is the current texel) ===
		vec3 a = texture(_t00, vec2(texture_coordinate.x - 2*x, texture_coordinate.y + 2*y)).rgb;
		vec3 b = texture(_t00, vec2(texture_coordinate.x,       texture_coordinate.y + 2*y)).rgb;
		vec3 c = texture(_t00, vec2(texture_coordinate.x + 2*x, texture_coordinate.y + 2*y)).rgb;

		vec3 d = texture(_t00, vec2(texture_coordinate.x - 2*x, texture_coordinate.y)).rgb;
		vec3 e = texture(_t00, vec2(texture_coordinate.x,       texture_coordinate.y)).rgb;
		vec3 f = texture(_t00, vec2(texture_coordinate.x + 2*x, texture_coordinate.y)).rgb;

		vec3 g = texture(_t00, vec2(texture_coordinate.x - 2*x, texture_coordinate.y - 2*y)).rgb;
		vec3 h = texture(_t00, vec2(texture_coordinate.x,       texture_coordinate.y - 2*y)).rgb;
		vec3 i = texture(_t00, vec2(texture_coordinate.x + 2*x, texture_coordinate.y - 2*y)).rgb;

		vec3 j = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
		vec3 k = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;
		vec3 l = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
		vec3 m = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

		// Apply weighted distribution:
		// 0.5 + 0.125 + 0.125 + 0.125 + 0.125 = 1
		// a,b,d,e * 0.125
		// b,c,e,f * 0.125
		// d,e,g,h * 0.125
		// e,f,h,i * 0.125
		// j,k,l,m * 0.5
		// This shows 5 square areas that are being sampled. But some of them overlap,
		// so to have an energy preserving downsample we need to make some adjustments.
		// The weights are the distributed, so that the sum of j,k,l,m (e.g.)
		// contribute 0.5 to the final color output. The code below is written
		// to effectively yield this sum. We get:
		// 0.125*5 + 0.03125*4 + 0.0625*4 = 1

		// Check if we need to perform Karis average on each block of 4 samples
		vec3 groups[5];
		//switch (mipLevel)
		//{
		//case 0:
		//	// We are writing to mip 0, so we need to apply Karis average to each block
		//	// of 4 samples to prevent fireflies (very bright subpixels, leads to pulsating
		//	// artifacts).
			groups[0] = (a+b+d+e) * (0.125f/4.0f);
			groups[1] = (b+c+e+f) * (0.125f/4.0f);
			groups[2] = (d+e+g+h) * (0.125f/4.0f);
			groups[3] = (e+f+h+i) * (0.125f/4.0f);
			groups[4] = (j+k+l+m) * (0.5f/4.0f);
			groups[0] *= KarisAverage(groups[0]);
			groups[1] *= KarisAverage(groups[1]);
			groups[2] *= KarisAverage(groups[2]);
			groups[3] *= KarisAverage(groups[3]);
			groups[4] *= KarisAverage(groups[4]);
			o_color = groups[0]+groups[1]+groups[2]+groups[3]+groups[4];
			o_color = max(o_color, 0.0001f);
			
		//	break;
		//default:
			//o_color = e*0.125;                // ok
			//o_color += (a+c+g+i)*0.03125;     // ok
			//o_color += (b+d+f+h)*0.0625;      // ok
			//o_color += (j+k+l+m)*0.125;       // ok
			////o_color.rgb = vec3(1, 0, 0);
		////}
	}
)"