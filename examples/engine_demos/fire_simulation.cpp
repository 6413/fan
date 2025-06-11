#include <fan/pch.h>

f32_t g_off = 0;
f32_t b_off = 0.2;
f32_t base = 20.f;

fan::color temperature_to_color(f32_t temp) {
  //f32_t base = 80.0f;
  f32_t diff = 255.0f - base;
  f32_t t = temp / 1000.0f; // 1000c

  f32_t r = base + std::min(t * diff, diff);
  f32_t g = base + std::max(std::min((t - 1.0f - g_off) * diff, diff), 0.0f);
  f32_t b = base + std::max(std::min((t - 1.0f - b_off) * diff, diff), 0.0f);

  return fan::color::rgb(r, g, b);
}

f32_t depth_to_temperature(f32_t depth) {
  f32_t bottom = gloco->window.get_size().y;
  f32_t temp = depth / (bottom / 2.f) * 1000.f; // 2000c
  return temp;
}

int main() {
  using namespace fan::graphics;
  engine_t engine;

  fan::graphics::image_t noise_image;
  fan::vec2 image_size = 128;
  int seed = fan::random::value((uint32_t)0, (uint32_t)-1);

  FastNoiseLite noise;
  noise.SetFractalType(FastNoiseLite::FractalType_FBm);
  noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
  float noise_frequency = 0.047;
  float noise_fractal_gain = 0.81;
  float noise_fractal_lacunarity = 2.26;
  int noise_fractal_octaves = 5;

  noise.SetFractalPingPongStrength(2.0);
  f32_t noise_tex_min = -1.64;
  f32_t noise_tex_max = 0.320;


  std::vector<uint8_t> noise_data_rgb(image_size.multiply() * 3);


  fan::graphics::sprite_t s{ {
    .position = 600,
    .size = 128
  } };

  f32_t u_offset = 0;

  f32_t width = 4;

  fan_window_loop{
    ImGui::Begin("A");
  fan_imgui_dragfloat1(base, 0.5);
  fan_imgui_dragfloat1(g_off, 0.01);
  fan_imgui_dragfloat1(b_off, 0.01);
  fan_imgui_dragfloat1(width, 0.01);
     ImGui::DragInt("seed", &seed, 0.1) |
      ImGui::DragInt("octaves", &noise_fractal_octaves, 1) |
      ImGui::DragFloat("frequency", &noise_frequency, 0.001) |
      ImGui::DragFloat("fractal gain", &noise_fractal_gain, 0.01) |
      ImGui::DragFloat("fractal lacunarity", &noise_fractal_lacunarity, 0.01) |
      ImGui::DragFloat("texture min", &noise_tex_min, 0.01) |
      ImGui::DragFloat("texture max", &noise_tex_max, 0.01);

      noise.SetFrequency(noise_frequency);
      noise.SetFractalGain(noise_fractal_gain);
      noise.SetFractalLacunarity(noise_fractal_lacunarity);
      noise.SetFractalOctaves(noise_fractal_octaves);
      noise.SetSeed(seed);
    ImGui::End();

    if (noise_image.iic() == false) {
      engine.image_unload(noise_image);
    }

    int index = 0;
    for (int y = 0; y < image_size.y; y++) {
      for (int x = 0; x < image_size.x; x++) {
        f32_t noise_value = noise.GetNoise((f32_t)x, (f32_t)y - u_offset);
        float centerX = image_size.x / 2.0f;
        fan::color c = temperature_to_color(depth_to_temperature((f32_t)y / 3));
        f32_t scale = 255.f / (noise_tex_max - noise_tex_min);
        int noise =255.0 - (uint8_t)std::max(0.0f, std::min(255.0f, (noise_value - noise_tex_min) * scale));

        noise = std::clamp((f32_t)noise, 0.0f, 255.0f);
        float edgeFactor = std::max(0.0f, 1.0f - std::abs(x - centerX) / centerX);

        noise *= edgeFactor;

        if (noise < 30.0f) {
          noise = 0.0f;
        }


        noise -= (image_size.y - y) / image_size.y * 255.f;

        float dx = (x - centerX) / centerX;
        float distance = dx * dx;

        float peak = 255.0f * 0.5f;
        float intensity = std::exp(-distance * width);

        noise += intensity * peak;
        noise = std::clamp((f32_t)noise, 0.0f, 255.0f);

        noise = std::max(noise, 0);
        noise = std::clamp((f32_t)noise, 0.0f, 85.0f);
        float amplificationFactor = std::max(0.0f, 1.0f - distance); 
        noise *= amplificationFactor * 3;
        noise = pow(noise / 255.f, 2.f) * 255.f;

        if (y < 5) {
          noise = 0;
        }

        noise_data_rgb[index * 3 + 0] = c.r * 255.f * (noise / 255.f);
        noise_data_rgb[index * 3 + 1] = c.g * 255.f * (noise / 255.f);
        noise_data_rgb[index * 3 + 2] = c.b * 255.f * (noise / 255.f);
        index++;
      }
    }

    fan::image::info_t ii;
    ii.data = noise_data_rgb.data();
    ii.size = image_size;
    ii.channels = 3;

    loco_t::image_load_properties_t lp;
    lp.format = fan::graphics::image_format::rgb_unorm;
    lp.internal_format = fan::graphics::image_format::rgb_unorm;
    lp.min_filter = fan::graphics::image_filter::nearest;
    lp.mag_filter = fan::graphics::image_filter::nearest;
    lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;

    noise_image = engine.image_load(ii, lp);
    u_offset -= engine.delta_time * 50;
    s.set_image(noise_image);
  };
}