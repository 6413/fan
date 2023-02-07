// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)


#include _FAN_PATH(io/file.h)

#include <shaderc/shaderc.hpp>

struct shader_t {

  static std::string preprocess_shader(const std::string& source_name,
    shaderc_shader_kind kind,
    const fan::string& source) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    //options.AddMacroDefinition("MY_DEFINE", "1");

    shaderc::PreprocessedSourceCompilationResult result =
      compiler.PreprocessGlsl(source.c_str(), kind, source_name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
      fan::throw_error(result.GetErrorMessage().c_str());
    }

    return { result.cbegin(), result.cend() };
  }

  static std::vector<uint32_t> compile_file(const fan::string& source_name,
    shaderc_shader_kind kind,
    const fan::string& source) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    //options.AddMacroDefinition("MY_DEFINE", "1");
    #if fan_debug > 1
    options.SetOptimizationLevel(shaderc_optimization_level_zero);
    #else
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    #endif

    shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source.c_str(), kind, source_name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
      fan::throw_error(module.GetErrorMessage().c_str());
    }

    return { module.cbegin(), module.cend() };
  }

  void set_vertex(const fan::string& shader_name, const fan::string& shader_code) {
    // fan::print(
    //   "processed vertex shader:", path, "resulted in:",
    // preprocess_shader(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
    // );

    auto spirv =
      compile_file(shader_name.c_str(), shaderc_glsl_vertex_shader, shader_code);
  }
  void set_fragment(const fan::string& shader_name, const fan::string& shader_code) {

    //fan::print(
     // "processed vertex shader:", path, "resulted in:",
    //preprocess_shader(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
    //);

    auto spirv =
      compile_file(shader_name.c_str(), shaderc_glsl_fragment_shader, shader_code);
  }
};

int main() {

  shader_t shader;
  fan::string str;
  fan::io::file::read("include/fan/graphics/glsl/opengl/2D/objects/light_sun.fs", &str);
  
  shader.set_fragment("light_sun.fs", str);


  return 0;
}