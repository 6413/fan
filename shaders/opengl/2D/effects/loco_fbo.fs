#version 330

in vec2 texture_coordinate;

layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00; // HDR color texture
uniform sampler2D _t01; // Bloom texture
uniform sampler2D _t02; // Bloom texture
uniform float bloom_strength = 0;
uniform float gamma = 1.00;
uniform float gamma2 = 1.00;
uniform float bloom_gamma = 1.03;
uniform float exposure = 1.0;
uniform float contrast = 1.0;

uniform float framebuffer_alpha = 1.0;

uniform float bloom_intensity = 1.0;
uniform vec2 window_size;

vec3 reinhard_tone_mapping(vec3 color) {
    return color / (color + vec3(1.0));
}

vec3 uncharted2_tone_mapping(vec3 color) {
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
}

vec3 apply_bloom(vec3 hdrColor, vec3 bloomColor) {
  float brightness = dot(bloomColor, vec3(0.2126, 0.7152, 0.0722));
  
  // Wide smoothstep and power for natural falloff
  float bloomFactor = pow(smoothstep(0.6, 1.5, brightness), 1.2);

  vec3 result = hdrColor + bloomColor * bloomFactor * bloom_strength;
  return result;
}

float vignette(vec2 uv) {
    float dist = distance(uv, vec2(0.5));
    return smoothstep(0.8, 0.3, dist);
}

vec3 chromatic_aberration(vec3 color, vec2 uv) {
    float amount = 0.005;
    float r = texture2D(_t00, uv + vec2(amount, 0.0)).r;
    float g = texture2D(_t00, uv).g;
    float b = texture2D(_t00, uv - vec2(amount, 0.0)).b;
    return vec3(r, g, b);
}

float film_grain(vec2 uv) {
    float t = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
    return t * 0.05;
}

vec3 lens_distortion(vec2 uv, vec3 color) {
    float amount = 0.3;
    float dist = distance(uv, vec2(0.5));
    uv = uv + (uv - vec2(0.5)) * dist * amount;
    return texture2D(_t00, uv).rgb;
}

vec3 sepia_tone(vec3 color) {
    vec3 sepiaColor = vec3(1.2, 1.0, 0.8);
    return color * sepiaColor;
}

vec3 edge_detection(vec2 uv) {
    vec3 color = texture2D(_t00, uv).rgb;
    vec3 colorU = texture2D(_t00, uv + vec2(0.0, 0.01)).rgb;
    vec3 colorD = texture2D(_t00, uv - vec2(0.0, 0.01)).rgb;
    vec3 colorL = texture2D(_t00, uv - vec2(0.01, 0.0)).rgb;
    vec3 colorR = texture2D(_t00, uv + vec2(0.01, 0.0)).rgb;
    vec3 edgeColor = abs(colorU - colorD) + abs(colorL - colorR);
    return edgeColor;
}

vec3 pixelation(vec2 uv, vec3 color) {
    float pixelSize = 0.01;
    vec2 pixelatedUV = vec2(floor(uv.x / pixelSize) * pixelSize, floor(uv.y / pixelSize) * pixelSize);
    return texture2D(_t00, pixelatedUV).rgb;
}

vec3 invert_colors(vec3 color) {
    return vec3(1.0) - color;
}

vec3 rgb_split(vec2 uv, vec3 color) {
    float amount = 0.005;
    float r = texture2D(_t00, uv + vec2(amount, 0.0)).r;
    float g = texture2D(_t00, uv).g;
    float b = texture2D(_t00, uv - vec2(amount, 0.0)).b;
    return vec3(r, g, b);
}

vec3 apply_contrast(vec3 color, float contrast_value) {
  return clamp((color - 0.5) * contrast_value + 0.5, 0.0, 1.0);
}

vec3 apply_gamma(vec3 color, float gamma_value) {
  return pow(max(color, vec3(0.0)), vec3(1.0 / gamma_value));
}

vec3 apply_exposure(vec3 color, float exposure_value) {
  return color * exposure_value;
}


void main() {
vec2 uv = gl_FragCoord.xy / window_size;
    vec3 hdrColor = texture(_t00, texture_coordinate).rgb;
    vec3 bloomColor = texture(_t01, texture_coordinate).rgb;

    // Apply bloom effect
    vec3 color = apply_bloom(hdrColor, bloomColor);
    // Apply vignette
 //   color *= vignette(texture_coordinate);

    // Apply chromatic aberration
  //  color = chromatic_aberration(color, texture_coordinate);

    // Apply film grain
  //  color += film_grain(texture_coordinate);

    // Apply lens distortion
  //  color = lens_distortion(texture_coordinate, color);

    // Apply sepia tone
  //  color = sepia_tone(color);

    // Apply edge detection
  //  color += edge_detection(texture_coordinate);

    // Apply pixelation
  //  color = pixelation(texture_coordinate, color);

    // Apply invert colors
   // color = invert_colors(color);

    // Apply RGB split
  //  color = rgb_split(texture_coordinate, color);
    color = apply_exposure(color, exposure);
    color = apply_contrast(color, contrast);
    color = apply_gamma(color, gamma);


    o_attachment0 = vec4(color, framebuffer_alpha);
}