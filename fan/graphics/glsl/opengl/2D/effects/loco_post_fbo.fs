#version 330
in vec2 texture_coordinate;

out vec4 output_color;

uniform sampler2D _t00;
uniform usampler2D _t01;
uniform sampler2D _t02;

uniform float m_time;

vec2 CRTCurveUV( vec2 uv )
{
    uv = uv * 2.0 - 1.0;
    vec2 offset = abs( uv.yx ) / vec2( 1.5, 4.0 );
    uv = uv + uv * offset * offset;
    uv = uv * 0.5 + 0.5;
    return uv;
}

void main() {
	vec2 tc = texture_coordinate;
	//tc = CRTCurveUV(tc);
	vec3 scene_texture = texture(_t00, tc).rgb;
	output_color = vec4(scene_texture, 1);
}