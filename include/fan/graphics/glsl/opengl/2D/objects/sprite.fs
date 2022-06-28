R"(
#version 330

in vec2 texture_coordinate;

in vec4 instance_color;

out vec4 o_color;

uniform sampler2D texture_sampler;

uniform float input;

vec2 hash( vec2 p ) // replace this by something better
{
	p = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

	vec2  i = floor( p + (p.x+p.y)*K1 );
    vec2  a = p - i + (i.x+i.y)*K2;
    float m = step(a.y,a.x); 
    vec2  o = vec2(m,1.0-m);
    vec2  b = a - o + K2;
	vec2  c = a - 1.0 + 2.0*K2;
    vec3  h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );
	vec3  n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3(70.0) );
}

float rand(vec2 co){
return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {

vec2 iResolution = vec2(800, 600);

vec2 p = gl_FragCoord.xy / iResolution.xy;

	vec2 uv = p*vec2(iResolution.x/iResolution.y,1.0) + input;
	
	float f = 0.0;

	uv *= 5.0;
      mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
	f  = 0.5000*noise( uv ); uv = m*uv;
	f += 0.2500*noise( uv ); uv = m*uv;
	f += 0.1250*noise( uv ); uv = m*uv;
	f += 0.0625*noise( uv ); uv = m*uv;

	f = 0.5 + 0.5*f;

  o_color = texture(texture_sampler, texture_coordinate) * instance_color;
	float b = o_color.b;
	if (o_color.b > 0) {
		if (f > 0.5) {
			o_color.g = 0.501960784313725;
			o_color.b = 0;
		}
	f += 0.3500*noise( uv * (f * input)); uv = m*uv;
		float s = f * (f / 1.1);
		if (s > 0.5) {
			o_color.g -= 1;
			o_color.b = b;
		}
		
	}

}
)"