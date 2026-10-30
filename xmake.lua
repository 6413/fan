set_project("fan")
set_languages("cxx23")

option("compiler")
  set_default("clang")
  set_values("clang", "gcc", "g++")
option_end()

local compiler = get_config("compiler")
local is_gcc = compiler == "gcc" or compiler == "g++"

if is_plat("wasm") then
  add_cxxflags("-pthread", "-DFAN_WASM", {force = true})
  add_ldflags("-s USE_GLFW=3", "-s ASYNCIFY=1", "-pthread", "-s PTHREAD_POOL_SIZE=4", {force = true})
else
  set_toolchains(is_gcc and "gcc" or "clang")
end

rule("mode.mode_none") rule_end()
rule("mode.release-minsize") rule_end()
add_rules("mode.mode_none", "mode.debug", "mode.release", "mode.release-minsize")

rule("mode.asan")
  on_load(function (target)
    target:set("optimize", "none")
    target:set("symbols", "debug")
    target:add("cxxflags", "-fsanitize=address", "-fno-omit-frame-pointer")
    target:add("ldflags", "-fsanitize=address")
    target:add("defines", "_DEBUG=3")
  end)
rule_end()

if is_mode("asan") then add_rules("mode.asan") end
set_defaultmode("mode_none")

if is_mode("mode_none") or is_mode("debug") then
  add_defines("_DEBUG=3")
end
if is_mode("release") then
  set_optimize("fastest")
  add_cxflags("-O3", "-ffunction-sections", "-fdata-sections", {force = true})
  add_ldflags("-s", "-fuse-ld=gold", "-Wl,--gc-sections", "-Wl,--icf=all", {force = true})
  add_defines("NDEBUG", "_DEBUG=0")
elseif is_mode("release-minsize") then
  set_optimize("smallest")
  add_cxflags("-Oz", "-ffunction-sections", "-fdata-sections", {force = true})
  add_ldflags("-s", "-Wl,--gc-sections", {force = true})
  add_defines("NDEBUG", "_DEBUG=0")
elseif is_mode("debug") then
  set_optimize("none")
  set_symbols("debug")
  add_cxflags("-g", "-gdwarf-4", "-fno-inline", "-fno-inline-functions", {force = true})
end

local fan_features = {
  FAN_WINDOW = true,
  FAN_2D = true,
  FAN_GUI = true,
  FAN_PHYSICS_2D = true,
  FAN_JSON = true,
  FAN_3D = false,

  FAN_FMT = true,
  FAN_WAYLAND_SCREEN = false,
  FAN_NETWORK = false,
  FAN_AUDIO = false,
  FAN_VIDEO = false,
  FAN_REFLECTION = false,
  FAN_SHADERC = true
}

local fan_feature_names = {}
for name in pairs(fan_features) do table.insert(fan_feature_names, name) end
table.sort(fan_feature_names)
for _, name in ipairs(fan_feature_names) do
  local enabled = fan_features[name]
  option(name) set_default(enabled) option_end()
  if has_config(name) then add_defines(name) end
end

option("static_runtime") set_default(false) option_end()

if not has_config("FAN_WINDOW") then
  for _, f in ipairs({"FAN_GUI", "FAN_2D", "FAN_3D", "FAN_VIDEO"}) do
    if has_config(f) then os.raise(f .. " requires FAN_WINDOW") end
  end
end

option("FAN_REFLECTION_MAIN") set_default(false) option_end()

if has_config("FAN_REFLECTION") or has_config("FAN_REFLECTION_MAIN") then
  set_languages("cxx26")
end
if has_config("FAN_REFLECTION") then
  add_cxxflags("-freflection", {force = true})
end

option("FAN_USE_STD_MODULE") set_default(false) set_showmenu(true) add_defines("FAN_USE_STD_MODULE") option_end()
option("main") set_default("examples/engine_demos/engine_demo.cpp") option_end()
option("FAN_PREBUILT_SHADER_LIBS")
  set_default(true)
  set_showmenu(true)
  set_description("Use system-installed shaderc/glslang/spirv-tools instead of building them from source")
option_end()

local static_req = {system = false, configs = {shared = false}}
-- python is only a build-time tool for glslang/spirv-tools; use the system one instead of building from source
add_requireconfs("**.python", {system = true, override = true})
if has_config("FAN_FMT") then add_requires("fmt 10.2.1", static_req) end
if has_config("FAN_2D") then
  add_defines("VK_ENABLE_BETA_EXTENSIONS")
  add_requires("vulkan-headers v1.4.335", {system = false})
  package("glslang")
    set_homepage("https://github.com/KhronosGroup/glslang/")
    set_description("Khronos-reference front end for GLSL/ESSL, partial front end for HLSL, and a SPIR-V generator.")
    set_license("Apache-2.0")

    add_urls("https://github.com/KhronosGroup/glslang.git")
    add_versions("1.4.335+0", "b5782e52ee2f7b3e40bb9c80d15b47016e008bc9")

    add_configs("binaryonly",  {description = "Only use binary program.", default = false, type = "boolean"})
    add_configs("exceptions",  {description = "Build with exception support.", default = false, type = "boolean"})
    add_configs("spirv_tools", {description = "Enable SPIRV-Tools integration (Optimizer).", default = false, type = "boolean"})
    add_configs("hlsl",        {description = "Enable HLSL support.", default = false, type = "boolean"})
    add_configs("rtti",        {description = "Build with RTTI support.", default = false, type = "boolean"})
    add_configs("tools",       {description = "Build the glslangValidator tool.", default = false, type = "boolean"})
    add_configs("default_resource_limits", {description = "Build with default resource limits.", default = false, type = "boolean"})
    if is_plat("wasm") then
        add_configs("shared", {description = "Build shared library.", default = false, type = "boolean", readonly = true})
    end

    add_deps("cmake")
    if is_plat("linux", "bsd") then
        add_syslinks("pthread")
    end

    on_load(function (package)
        if package:is_binary() or package:config("binaryonly") then
            package:config_set("tools", true)
            package:set("kind", "binary")
        end
        if package:config("spirv_tools") or package:config("tools") then
            package:add("deps", "python 3.x", {kind = "binary"})
        end
        if package:config("spirv_tools") then
            package:add("deps", "spirv-tools")
        end

        if package:config("tools") then
            package:addenv("PATH", "bin")
        end
        package:add("links", "glslang", "MachineIndependent", "GenericCodeGen", "OGLCompiler", "OSDependent", "SPIRV", "SPVRemapper")
        if package:config("hlsl") then
            package:add("links", "HLSL")
        end
        if package:config("default_resource_limits") then
            package:add("links", "glslang-default-resource-limits")
        end
    end)

    on_install(function (package)
        if package:config("spirv_tools") then
            io.replace("StandAlone/CMakeLists.txt", "target_link_libraries(glslangValidator ${LIBRARIES})", [[
                target_link_libraries(glslangValidator ${LIBRARIES} SPIRV-Tools-opt SPIRV-Tools-link SPIRV-Tools-reduce SPIRV-Tools)
            ]], {plain = true})
            io.replace("SPIRV/CMakeLists.txt", "target_link_libraries(SPIRV PRIVATE MachineIndependent SPIRV-Tools-opt)", [[
                target_link_libraries(SPIRV PRIVATE MachineIndependent SPIRV-Tools-opt SPIRV-Tools-link SPIRV-Tools-reduce SPIRV-Tools)
            ]], {plain = true})
        end

        io.replace("CMakeLists.txt", 'set(CMAKE_DEBUG_POSTFIX "d")', [[
            message(WARNING "Disabled CMake Debug Postfix for xmake package generation")
        ]], {plain = true})

        if package:is_plat("wasm") then
            io.replace("CMakeLists.txt", [[add_link_options("-Wl,--no-undefined")]], "", {plain = true})
        end

        local configs = {"-DENABLE_CTEST=OFF", "-DGLSLANG_TESTS=OFF", "-DBUILD_EXTERNAL=OFF", "-DENABLE_PCH=OFF"}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and not package:config("binaryonly") and "ON" or "OFF"))
        table.insert(configs, "-DENABLE_EXCEPTIONS=" .. (package:config("exceptions") and "ON" or "OFF"))
        table.insert(configs, "-DENABLE_HLSL=" .. (package:config("hlsl") and "ON" or "OFF"))
        table.insert(configs, "-DENABLE_RTTI=" .. (package:config("rtti") and "ON" or "OFF"))
        table.insert(configs, "-DENABLE_GLSLANG_BINARIES=" .. (package:config("tools") and "ON" or "OFF"))

        local packagedeps = {}
        if package:config("spirv_tools") then
            table.insert(configs, "-DENABLE_OPT=ON")
            table.insert(configs, "-DALLOW_EXTERNAL_SPIRV_TOOLS=ON")
            table.insert(packagedeps, "spirv-tools")
        else
            table.insert(configs, "-DENABLE_OPT=OFF")
        end
        import("package.tools.cmake").install(package, configs, {packagedeps = packagedeps})

        if package:is_binary() then
            os.rm(package:installdir("*|bin"))
        else
            os.cp("glslang/MachineIndependent/**.h", package:installdir("include", "glslang", "MachineIndependent"))
            os.cp("glslang/Include/**.h", package:installdir("include", "glslang", "Include"))

            if package:config("tools") then
                local bindir = package:installdir("bin")
                local glslangValidator = path.join(bindir, "glslangValidator" .. (is_host("windows") and ".exe" or ""))
                if not os.isfile(glslangValidator) then
                    local glslang = path.join(bindir, "glslang" .. (is_host("windows") and ".exe" or ""))
                    os.trycp(glslang, glslangValidator)
                end
            end
        end
    end)

    on_test(function (package)
        if not package:is_cross() and package:config("tools") then
            os.vrun("glslangValidator --version")
        end

        if not package:is_binary() then
            assert(package:has_cxxfuncs("ShInitialize", {configs = {languages = "c++11"}, includes = "glslang/Public/ShaderLang.h"}))
        end
    end)
  package_end()

  package("spirv-tools")
    set_homepage("https://github.com/KhronosGroup/SPIRV-Tools/")
    set_description("SPIR-V Tools")
    set_license("Apache-2.0")

    add_urls("https://github.com/KhronosGroup/SPIRV-Tools.git")
    add_versions("1.4.335+0", "vulkan-sdk-1.4.335.0")

    add_deps("cmake >=3.17.2")
    add_deps("python 3.x", {kind = "binary"})
    if is_plat("linux") then
        add_extsources("apt::spirv-tools", "pacman::spirv-tools")
    end

    on_load(function (package)
        local sdkver = package:version():split("%+")[1]
        package:add("deps", "spirv-headers " .. sdkver)
    end)

    on_fetch("macosx", function (package, opt)
        if opt.system then
            local result = package:find_package("spirv-tools")
            if result and not result.includedirs then
                for _, linkdir in ipairs(result.linkdirs) do
                    if linkdir:startswith("/usr") then
                        local includedir = path.join(path.directory(linkdir), "include", "spirv-tools")
                        if os.isdir(includedir) then
                            includedir = path.directory(includedir)
                            result.includedirs = result.includedirs or {}
                            table.insert(result.includedirs, includedir)
                        end
                    end
                end
            end
            return result
        end
    end)

    on_install(function (package)
        package:addenv("PATH", "bin")
        local configs = {"-DSPIRV_SKIP_TESTS=ON", "-DSPIRV_WERROR=OFF"}
        if package:debug() then
            table.insert(configs, "-DCMAKE_COMPILE_PDB_OUTPUT_DIRECTORY=''")
        end
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        local spirv = package:dep("spirv-headers")
        table.insert(configs, "-DSPIRV-Headers_SOURCE_DIR=" .. spirv:installdir():gsub("\\", "/"))
        io.replace("source/libspirv.cpp", [[bool SpirvTools::Validate(const std::vector<uint32_t>& binary) const {
  return Validate(binary.data(), binary.size());
}

bool SpirvTools::Validate(const uint32_t* binary,
                          const size_t binary_size) const {
  return spvValidateBinary(impl_->context, binary, binary_size, nullptr) ==
         SPV_SUCCESS;
}

bool SpirvTools::Validate(const uint32_t* binary, const size_t binary_size,
                          spv_validator_options options) const {
  spv_const_binary_t the_binary{binary, binary_size};
  spv_diagnostic diagnostic = nullptr;
  bool valid = spvValidateWithOptions(impl_->context, options, &the_binary,
                                      &diagnostic) == SPV_SUCCESS;
  if (!valid && impl_->context->consumer) {
    impl_->context->consumer.operator()(
        SPV_MSG_ERROR, nullptr, diagnostic->position, diagnostic->error);
  }
  spvDiagnosticDestroy(diagnostic);
  return valid;
}]], [[bool SpirvTools::Validate(const std::vector<uint32_t>& binary) const {
  return true;
}

bool SpirvTools::Validate(const uint32_t* binary,
                          const size_t binary_size) const {
  return true;
}

bool SpirvTools::Validate(const uint32_t* binary, const size_t binary_size,
                          spv_validator_options options) const {
  return true;
}]], {plain = true})
        if package:is_plat("windows") then
            import("package.tools.cmake").install(package, configs, {buildir = os.tmpfile() .. ".dir"})
        else
            import("package.tools.cmake").install(package, configs)
        end
        package:add("links", "SPIRV-Tools-link", "SPIRV-Tools-reduce", "SPIRV-Tools-opt")
        if package:config("shared") then
            package:add("links", "SPIRV-Tools-shared")
        else
            package:add("links", "SPIRV-Tools")
        end
    end)

    on_test(function (package)
        if not package:is_cross() then
            os.runv("spirv-as --help")
            os.runv("spirv-opt --help")
        end
        assert(package:has_cxxfuncs("spvContextCreate", {configs = {languages = "c++17"}, includes = "spirv-tools/libspirv.hpp"}))
    end)
  package_end()

  package("shaderc")
    set_homepage("https://github.com/google/shaderc")
    set_description("A collection of tools, libraries, and tests for Vulkan shader compilation.")
    set_license("Apache-2.0")

    add_urls("https://github.com/google/shaderc/archive/refs/tags/$(version).tar.gz",
             "https://github.com/google/shaderc.git")

    add_versions("v2024.1", "eb3b5f0c16313d34f208d90c2fa1e588a23283eed63b101edd5422be6165d528")

    add_configs("binaryonly", {description = "Only use binary program.", default = false, type = "boolean"})
    add_configs("exceptions", {description = "Enable exception handling", default = true, type = "boolean"})
    if is_plat("windows", "wasm") then
        add_configs("shared", {description = "Build shared library.", default = false, type = "boolean", readonly = true})
    end

    add_deps("cmake")

    if is_plat("linux") then
        add_syslinks("pthread")
    end

    on_load(function (package)
        if package:config("binaryonly") then
            package:set("kind", "binary")
        end

        -- prebuilt shaderc comes from the system, so don't pull in its build deps
        if not import("core.project.config").get("FAN_PREBUILT_SHADER_LIBS") then
            if package:version():ge("2022.3") then
                package:add("deps", "glslang")
            else
                package:add("deps", "glslang <=1.3.211")
            end
            package:add("deps", "spirv-tools", "spirv-headers")
        end

        if package:config("shared") then
            package:add("links", "shaderc_shared")
        else
            package:add("links", "shaderc_combined")
        end
    end)

    on_fetch(function (package, opt)
        if opt.system and package:is_binary() then
            return package:find_tool("glslc")
        end
    end)

    on_install(function (package)
        local opt = {}
        opt.packagedeps = {"glslang", "spirv-tools", "spirv-headers"}
        io.replace("CMakeLists.txt", "add_subdirectory(third_party)", "", {plain = true})
        io.replace("libshaderc_util/src/compiler.cc", "SPIRV/GlslangToSpv.h", "glslang/SPIRV/GlslangToSpv.h", {plain = true})
        io.replace("libshaderc_util/src/compiler.cc", "shader.setHlslIoMapping(hlsl_iomap_);", "", {plain = true})
        io.replace("libshaderc_util/src/spirv_tools_wrapper.cc", [[  spvtools::ValidatorOptions val_opts;
  // This allows flexible memory layout for HLSL.
  val_opts.SetSkipBlockLayout(true);
  // This allows HLSL legalization regarding resources.
  val_opts.SetRelaxLogicalPointer(true);
  // This uses relaxed rules for pre-legalized HLSL.
  val_opts.SetBeforeHlslLegalization(true);

  // Set additional optimizer options.
  optimizer_options.set_validator_options(val_opts);
  optimizer_options.set_run_validator(true);]], "", {plain = true})

        if not package:has_tool("sh", "link") then
            local links = {}
            for _, dep in ipairs({"glslang", "spirv-tools"}) do
                local fetchinfo = package:dep(dep):fetch()
                if fetchinfo then
                    for _, link in ipairs(fetchinfo.links) do
                        table.insert(links, link)
                    end
                end
            end
            if package:version():ge("2023.8") then
                io.replace("libshaderc_util/CMakeLists.txt", "glslang SPIRV", table.concat(links, " "), {plain = true})
            else
                io.replace("glslc/CMakeLists.txt", "glslang OSDependent OGLCompiler HLSL glslang SPIRV", "", {plain = true})
                io.replace("libshaderc_util/CMakeLists.txt", "glslang OSDependent OGLCompiler HLSL glslang SPIRV", table.concat(links, " "), {plain = true})
            end
            links = table.join({"shaderc", "shaderc_util"}, links)
            io.replace("glslc/CMakeLists.txt", "shaderc_util shaderc", table.concat(links, " "), {plain = true})
        end

        local version_str = format("shaderc %s\nspirv-tools %s\nglslang %s\0", 
            package:version(),
            package:dep("spirv-tools"):version(),
            package:dep("glslang"):version()
        )

        local version_c_array = "{"
        for i = 1, #version_str do
            local char = version_str:sub(i, i)
            if char == "\n" then
                char = [[\n]]
            end
            if char == "\0" then
                char = [[\0]]
            end
            version_c_array = format([[%s'%s',]], version_c_array, char)
        end
        version_c_array = version_c_array .. "}"

        io.writefile("glslc/src/build-version.inc", version_c_array)
        io.replace("glslc/CMakeLists.txt", "add_dependencies(glslc_exe build-version)", "", {plain = true})

        local configs = {
            "-DSHADERC_SKIP_EXAMPLES=ON",
            "-DSHADERC_SKIP_TESTS=ON",
            "-DSHADERC_ENABLE_COPYRIGHT_CHECK=OFF"
        }
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        if package:is_plat("windows") then
            io.replace("CMakeLists.txt", [[set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")]], "", {plain = true})
            io.replace("CMakeLists.txt", [[set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")]], "", {plain = true})
            table.insert(configs, "-DSHADERC_ENABLE_SHARED_CRT=" .. (package:config("vs_runtime"):startswith("MT") and "OFF" or "ON"))
        end

        if package:config("exceptions") then
            table.insert(configs, "-DDISABLE_EXCEPTIONS=OFF")
            if package:is_plat("windows") and package:has_tool("cxx", "cl", "clang_cl") then
                opt.cxflags = "/EHsc"
            end
        else
            table.insert(configs, "-DDISABLE_EXCEPTIONS=ON")
        end
        import("package.tools.cmake").install(package, configs, opt)
        package:addenv("PATH", "bin")
    end)

    on_test(function (package)
        if not package:is_cross() then
            os.vrun("glslc --version")
        end

        if not package:is_binary() then
            assert(package:has_cfuncs("shaderc_compiler_initialize", {includes = "shaderc/shaderc.h"}))
        end
    end)
  package_end()

  if has_config("FAN_SHADERC") then
    if has_config("FAN_PREBUILT_SHADER_LIBS") then
      -- glslang/spirv-tools are only build deps of shaderc, so the system shaderc is all we need
      add_requires("shaderc", {system = true})
    else
      add_requires("glslang 1.4.335+0", static_req)
      add_requires("spirv-tools 1.4.335+0", static_req)
      add_requires("spirv-headers 1.4.335+0", {system = false})
      add_requires("shaderc", static_req)
    end
  end
end

if not is_plat("wasm") then
  add_requires("libuv 1.48.0", static_req)
  if has_config("FAN_WINDOW") then
    add_requires("zlib 1.3.1", static_req)
    add_requires("libpng 1.6.43", static_req)
    add_requires("libwebp 1.3.2", static_req)
    add_requires("glfw 3.4", static_req)
  end

  if has_config("FAN_GUI") then
    package("freetype")
      set_base("freetype")
      add_urls("https://github.com/freetype/freetype.git")
      add_versions("2.13.2", "VER-2-13-2")
      add_deps("cmake")
      on_load(function (package)
        package:add("includedirs", "include/freetype2")
      end)
      on_install(function (package)
        local configs = {}
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"))
        table.insert(configs, "-DFT_DISABLE_HARFBUZZ=TRUE")
        table.insert(configs, "-DFT_DISABLE_BROTLI=TRUE")
        table.insert(configs, "-DFT_DISABLE_BZIP2=TRUE")
        import("package.tools.cmake").install(package, configs)
      end)
    package_end()
    add_requires("freetype 2.13.2", static_req)
    add_requires("lunasvg 2.4.1", static_req)
  end
  if has_config("FAN_PHYSICS_2D") then
    add_requires("box2d 3.1.1", static_req)
  end
end

add_includedirs(".", {public = true})
add_sysincludedirs("third_party/fan/include", {public = true})
if has_config("FAN_2D") then add_sysincludedirs("third_party/fan/include/VulkanMemoryAllocator/include", {public = true}) end

if not is_gcc and not is_plat("wasm") then add_cxxflags("-stdlib=libstdc++", {force = true}) end

set_policy("build.c++.modules.std", true)
set_policy("build.c++.modules.reuse", true)
add_cxxflags("-pthread", {force = true})

local common_flags = {
  "-Wall", "-Wextra", "-Wno-unused-variable", "-Wno-unused-parameter", "-Wno-unused-function",
  "-Wno-missing-field-initializers", "-Wno-sign-compare",
  "-Wno-unused-but-set-parameter", "-Wno-unused-value", "-Wno-padded", "-Wno-parentheses",
  "-fsized-deallocation"
}
add_cxxflags(common_flags, {force = true})

if is_gcc then
  add_cxxflags(
    "-Wno-unused-static-function",
    "-fmax-errors=20",
    "-fmodules-ts",
    "-fno-module-lazy",
    {force = true}
  )
else
  add_cxxflags(
    "-ferror-limit=20",
    "-Wno-include-angled-in-module-purview",
    "-fmacro-backtrace-limit=0",
    "-Wno-shift-op-parentheses",
    "-Wno-int-to-void-pointer-cast",
    "-Wno-bitwise-op-parentheses",
    {force = true}
  )
end

if has_config("FAN_GUI") then
  add_defines("IMGUI_DEFINE_MATH_OPERATORS", "IMGUI_DISABLE_SSE", "STBI_NO_SIMD")
end

local module_files = {
  "fan/types/types.ixx", "fan/types/color.ixx", "fan/types/vector.ixx", "fan/types/quaternion.ixx",
  "fan/types/matrix.ixx", "fan/types/fstring.ixx", "fan/types/compile_time_string.ixx",
  "fan/types/flat_hash_map.ixx", "fan/types/bitset.ixx", "fan/memory/memory.ixx",
  "fan/math/math.ixx", "fan/math/intersection.ixx", "fan/system/time.ixx", "fan/utils/mpl.ixx",
  "fan/utils/utility.ixx", "fan/debug/formatter.ixx", "fan/debug/print_error.ixx", "fan/debug/print.ixx",
  "fan/utils/random.ixx", "fan/debug/log_dispatcher.ixx", "fan/utils/crypto.ixx", "fan/network/network_crypto.ixx", "fan/system/process.ixx",
  "fan/system/dynamic_library.ixx", "fan/system/hot_reload.ixx",
  "fan/io/io_types.ixx", "fan/io/directory.ixx", "fan/io/file.ixx", "fan/io/io_prompt.ixx",
  "fan/event/event_types.ixx", "fan/event/event.ixx", "fan/event/uv_raw.ixx", "fan/utils/compression.ixx"
}

local feature_modules = {
  FAN_WINDOW = {
    "fan/window/window.ixx", "fan/window/input_common.ixx", "fan/window/input.ixx", "fan/window/input_action.ixx",
    "fan/graphics/common_types.ixx", "fan/graphics/material.ixx", "fan/graphics/camera.ixx", "fan/graphics/image_load.ixx",
    "fan/graphics/webp.ixx", "fan/graphics/stb.ixx", "fan/graphics/common_context.ixx",
    "fan/graphics/audio_subsystem.ixx", "fan/graphics/physics_subsystem.ixx", "fan/graphics/input_subsystem.ixx",
    "fan/graphics/loco.ixx", "fan/graphics/graphics.ixx", "fan/graphics/file_dialog.ixx",
    "fan/graphics/2D/algorithm/raycast_grid.ixx", "fan/graphics/2D/algorithm/raymarch.ixx", "fan/graphics/2D/algorithm/chunk_renderer.ixx", "fan/graphics/gameplay/gameplay_types.ixx",
    "fan/graphics/gameplay/gameplay.ixx", "fan/graphics/graphics_event.ixx", "fan/texture_pack/tp0.ixx",
    "fan/physics/physics_types.ixx", "fan/algorithm/noise.ixx", "fan/algorithm/pathfind.ixx", "fan/gameplay/spatial.ixx", "fan/gameplay/ecs.ixx",
    "fan/graphics/gui/console.ixx",
    "fan/graphics/gui/tilemap_editor/loader.ixx", "fan/graphics/gui/tilemap_editor/renderer0.ixx"
  },

  FAN_2D = { "fan/graphics/2D/shapes_types.ixx", "fan/graphics/2D/grid_placer.ixx", "fan/graphics/2D/culling.ixx", "fan/graphics/2D/shapes.ixx",
              "fan/graphics/hot_shape_store.ixx",
              "fan/graphics/vulkan/vk_core_types.ixx", "fan/graphics/vulkan/vk_core_vai.ixx", "fan/graphics/vulkan/vk_core_image.ixx", "fan/graphics/vulkan/vk_core_compute.ixx", "fan/graphics/vulkan/vk_core_pipeline.ixx", "fan/graphics/vulkan/vk_core_camera_subsystem.ixx", "fan/graphics/vulkan/vk_core_uniform_block.ixx", "fan/graphics/vulkan/vk_core_shader_subsystem.ixx", "fan/graphics/vulkan/vk_core.ixx", "fan/algorithm/tween.ixx"
  },
  FAN_JSON = { "fan/types/json.ixx" },
  FAN_FMT = { "fan/utils/fmt.ixx" },
  FAN_NETWORK = { "fan/network/network.ixx", "fan/network/network_socket.ixx", "fan/graphics/2D/graphics_network.ixx" },
  FAN_AUDIO = { "fan/audio/audio.ixx" },
  FAN_PHYSICS_2D = { "fan/physics/b2_integration.ixx", "fan/physics/physics_common_context.ixx", "fan/graphics/physics_shapes.ixx", "fan/physics/vehicle_controller.ixx" },
  FAN_GUI = {
    "fan/graphics/gui/gui_base.ixx", "fan/graphics/gui/gui_input.ixx",
    "fan/graphics/gui/gui_types.ixx", "fan/graphics/gui/gui.ixx", "fan/graphics/gui/text_logger.ixx",
    "fan/graphics/gui/settings_menu.ixx", "fan/graphics/gui/keybinds_menu.ixx", "fan/graphics/gameplay/items.ixx",
    "fan/graphics/gui/gameplay/equipment.ixx", "fan/graphics/gui/gameplay/hotbar.ixx",
    "fan/graphics/gui/gameplay/inventory.ixx", "fan/graphics/gui/gameplay/inventory_hotbar.ixx",
    "fan/graphics/gui/gameplay/drag_drop.ixx", "fan/graphics/gui/gameplay/slot_renderer.ixx",
    "fan/graphics/gui/tilemap_editor/editor_core.ixx",
    "fan/graphics/gui/tilemap_editor/editor_ui.ixx",
    "fan/graphics/gui/fgm/viewport.ixx", "fan/graphics/gui/fgm/selection.ixx",
    "fan/graphics/gui/fgm/properties_ui.ixx", "fan/graphics/gui/fgm/animation_system.ixx",
    "fan/graphics/gui/fgm/scene_serializer.ixx", "fan/graphics/gui/fgm/fgm.ixx",
    "fan/graphics/gui/fgm/fgm_types.ixx",
    "fan/graphics/scene.ixx"
  },
  FAN_3D = { "fan/graphics/3D/objects/fms.ixx", "fan/graphics/voxel.ixx" },
  FAN_VIDEO = { "fan/video/codec.ixx", "fan/video/screen.ixx", "fan/video/renderer.ixx", "fan/video/video.ixx" },
  FAN_WAYLAND_SCREEN = {}
}

local feature_names = {}
for feat in pairs(feature_modules) do table.insert(feature_names, feat) end
table.sort(feature_names)
for _, feat in ipairs(feature_names) do
  if has_config(feat) then
    for _, f in ipairs(feature_modules[feat]) do table.insert(module_files, f) end
  end
end
table.insert(module_files, "fan/fan.ixx")

local impl_files = {}
for _, m in ipairs(module_files) do
  local p = path.join(path.directory(m), path.basename(m) .. "_impl.cpp")
  if os.isfile(p) then table.insert(impl_files, p) end
end
if os.isfile("fan/memory/memory_profile.cpp") then
  table.insert(impl_files, "fan/memory/memory_profile.cpp")
end
if has_config("FAN_2D") then
  for _, f in ipairs({"vk_core_device", "vk_core_shader", "vk_core_image", "vk_mem_alloc"}) do
    table.insert(impl_files, "fan/graphics/vulkan/" .. f .. "_impl.cpp")
  end
end
if has_config("FAN_WINDOW") and os.isfile("fan/graphics/2D/algorithm/AStar.cpp") then
  table.insert(impl_files, "fan/graphics/2D/algorithm/AStar.cpp")
end

if has_config("FAN_GUI") then
  target("imgui")
    set_kind("static")
    set_warnings("none")
    ---add_rules("c++.unity_build", {batchsize = 16})
    if has_config("FAN_WINDOW") then
      add_packages("glfw")
    end
    if has_config("FAN_2D") then
      add_packages("vulkan-headers")
      if is_plat("linux") then add_syslinks("vulkan") end
    end
    if not is_gcc and not is_plat("wasm") then
      add_cxxflags("-stdlib=libstdc++", {force = true})
      add_ldflags("-stdlib=libstdc++", "-lstdc++", {force = true})
    end
    if is_plat("linux") then
      add_packages("freetype", "lunasvg", "zlib", "libpng")
    else
      add_linkdirs("third_party/fan/lib")
      add_links("freetype", "lunasvg")
      add_syslinks("png16", "z")
    end
    add_includedirs("fan/imgui", "fan/imgui/misc/freetype", "third_party/fan/include")
    on_load(function (target)
      if target:is_plat("linux") then
        import("lib.detect.find_tool")
        if find_tool("pkg-config") then
          local res = os.iorunv("pkg-config", {"--cflags-only-I", "glib-2.0"})
          if res then
            local flags = res:split("%s+")
            table.sort(flags)
            for _, p in ipairs(flags) do
              if p:startswith("-I") then target:add("sysincludedirs", p:sub(3)) end
            end
          end
        end
      end
    end)
    add_files(
      "fan/imgui/imgui.cpp", "fan/imgui/imgui_draw.cpp", "fan/imgui/imgui_widgets.cpp",
      "fan/imgui/imgui_tables.cpp", "fan/imgui/imgui_impl_glfw.cpp", "fan/imgui/implot_items.cpp",
      "fan/imgui/implot.cpp", "fan/imgui/text_editor.cpp", "fan/imgui/misc/freetype/imgui_freetype.cpp",
      "fan/imgui/ImGuizmo.cpp"
    )
    if has_config("FAN_2D") then add_files("fan/imgui/imgui_impl_vulkan.cpp") end
  target_end()
end

if not is_plat("wasm") and has_config("FAN_WINDOW") then
  target("nfd")
    set_kind("static")
    set_warnings("none")
    add_rules("c++.unity_build", {batchsize = 8})
    add_files("fan/nativefiledialog/nfd_common.c")
    if is_plat("linux") then add_files("fan/nativefiledialog/nfd_gtk.c")
    elseif is_plat("windows") then add_files("fan/nativefiledialog/nfd_win.cpp") end
    on_load(function (target)
      if target:is_plat("linux") then
        import("lib.detect.find_tool")
        if find_tool("pkg-config") then
          local cflags = os.iorunv("pkg-config", {"--cflags-only-I", "gtk+-3.0"})
          if cflags then
            local flags = cflags:split("%s+")
            table.sort(flags)
            for _, p in ipairs(flags) do
              if p:startswith("-I") then target:add("sysincludedirs", p:sub(3)) end
            end
          end
          local libs = os.iorunv("pkg-config", {"--libs", "gtk+-3.0"})
          if libs then
            local flags = libs:split("%s+")
            table.sort(flags)
            for _, l in ipairs(flags) do
              if l:startswith("-l") then target:add("links", l:sub(3)) end
            end
          end
        end
      end
    end)
  target_end()
end

option("buildlib") set_default(false) option_end()

-- Hot-reloadable game module: Game.dll (Windows) / libGame.so (Linux).
-- Plain C++ only, no `import fan` -> rebuilds in ~1s.
-- Engine (a.exe with examples/hot_reload/main.cpp) owns GameState
-- and reloads this library live via dynamic_library_t + GetGameAPI().
target("Game")
  set_kind("shared")
  set_languages("cxx23")
  set_targetdir("$(projectdir)")
  set_policy("build.c++.modules", true)
  set_policy("build.c++.modules.std", true)
  add_files("examples/hot_reload/Game.cpp")
  add_includedirs(".", {public = true})
  set_warnings("none")
target_end()

target("a.exe")
  set_kind(has_config("buildlib") and "static" or "binary")

  if is_plat("wasm") then
    set_extension(".html")
    add_ldflags("--preload-file shaders -s MAX_WEBGL_VERSION=2", "-s MIN_WEBGL_VERSION=2", {force = true})
    add_linkdirs("third_party/fan/lib/wasm")
    add_links("uv_wasm", "webp_wasm")
  end

  if has_config("FAN_GUI") then add_deps("imgui") end
  if not is_plat("wasm") and has_config("FAN_WINDOW") then add_deps("nfd") end
  if has_config("FAN_FMT") then add_packages("fmt") end

if has_config("FAN_2D") then
    add_packages("vulkan-headers")
    if has_config("FAN_SHADERC") then add_packages("shaderc") end
    if is_plat("linux") then add_syslinks("vulkan") end
  end

  if has_config("FAN_2D") and not has_config("FAN_SHADERC") then
    before_build(function (target)
      import("lib.detect.find_tool")
      local glslc = find_tool("glslc")
      if not glslc then raise("FAN_SHADERC is disabled: glslc is required to precompile shaders/vulkan") end
      local failures = {}
      local compiled, up_to_date = 0, 0
      cprint("${cyan}[shaders]${clear} started building shaders (shaders/vulkan)")
      for _, ext in ipairs({"vert", "frag", "comp", "rgen", "rmiss", "rchit", "rahit"}) do
        for _, src in ipairs(os.files("shaders/vulkan/**." .. ext)) do
          local dst = src .. ".spv"
          if not os.isfile(dst) or os.mtime(src) > os.mtime(dst) then
            compiled = compiled + 1
            try {
              function () os.runv(glslc.program, {"--target-env=vulkan1.2", "-O", src, "-o", dst}) end,
              catch { function (e) table.insert(failures, {src, tostring(e)}) end }
            }
          else
            up_to_date = up_to_date + 1
          end
        end
      end
      if #failures > 0 then
        for _, f in ipairs(failures) do
          cprint("${red}shader precompile failed: %s${clear}\n%s", f[1], f[2])
        end
        raise("shader precompilation failed for " .. #failures .. " file(s)")
      end
      cprint("${green}[shaders]${clear} finished building shaders: %d compiled, %d up to date", compiled, up_to_date)
    end)
  end

  for _, f in ipairs(module_files) do add_files(f) end

  if has_config("FAN_REFLECTION") then
    local reflection_files = os.files("fan/reflection/*.ixx")
    table.sort(reflection_files)
    for _, f in ipairs(reflection_files) do
      add_files(f, {cxxflags = "-freflection"})
      local impl = path.join(path.directory(f), path.basename(f) .. "_impl.cpp")
      if os.isfile(impl) then add_files(impl, {cxxflags = "-freflection"}) end
    end
  end

  for _, f in ipairs(impl_files) do add_files(f) end


  set_policy("check.auto_ignore_flags", false)
  if not has_config("buildlib") then
    if has_config("FAN_REFLECTION") or has_config("FAN_REFLECTION_MAIN") then
      add_files(get_config("main"), {cxxflags = "-freflection"})
    else
      add_files(get_config("main"))
    end
  end

  add_includedirs(".", {public = true})
  add_sysincludedirs("third_party/fan/include", {public = true})
  if has_config("FAN_2D") then
    add_includedirs(
      "third_party/fan/include/VulkanMemoryAllocator/include",
      {public = true}
    )
  end
  add_linkdirs("third_party/fan/lib")

  if is_plat("linux") then
    add_links("stdc++exp", "crypto")
    if has_config("static_runtime") then add_ldflags("-static-libstdc++", "-static-libgcc", {force = true}) end
    add_packages("libuv")
    if has_config("FAN_WINDOW") then
      add_packages("glfw", "zlib", "libpng", "libwebp")
    end

    if has_config("FAN_AUDIO") then add_links("opus", "pulse-simple", "pulse") end
    if has_config("FAN_NETWORK") then add_links("ssl", "curl") end
    if has_config("FAN_GUI") then add_packages("freetype", "lunasvg") end
    if has_config("FAN_PHYSICS_2D") then add_packages("box2d") end
    if has_config("FAN_3D") then add_links("assimp") end
    if has_config("FAN_VIDEO") then add_links("avformat", "avcodec", "swscale", "avutil", "xcb", "xcb-shm", "X11") end
    if has_config("FAN_2D") then add_packages("vulkansdk") end
    if has_config("FAN_WAYLAND_SCREEN") then
      -- WITCH MD/SCR captures through the xdg-desktop-portal ScreenCast + PipeWire instead of X11
      add_defines("__HAVE_WAYLAND")
      add_files("fan/video/wlr-screencopy-unstable-v1-protocol.c", "fan/video/xdg-output-unstable-v1-client-protocol.c")
      add_links("wayland-client", "pipewire-0.3", "dbus-1", "avcodec", "avutil", "swscale")
    end
  elseif is_plat("windows") then
    add_linkdirs("lib/GLFW", "lib/GLEW", "lib/libuv", "lib/libwebp", "lib/opus", "lib/openssl")
    add_links("glfw3_mt", "glew32s", "uv_a", "libwebp", "opus", "libssl", "libcrypto")
    if has_config("FAN_GUI") then add_linkdirs("lib/freetype", "lib/lunasvg") add_links("freetype", "lunasvg") end
    if has_config("FAN_PHYSICS_2D") then add_linkdirs("lib/box2d") add_links("box2d") end
    if has_config("FAN_3D") then add_linkdirs("C:/Program Files/Assimp/lib/x64") add_links("assimp-vc143-mt") end
    if has_config("FAN_2D") then add_packages("vulkansdk") end
    if has_config("FAN_WAYLAND_SCREEN") then
      add_linkdirs("lib/libx264", "lib/openh264")
      add_links("DXGI", "D3D11", "libx264", "welsdcore", "welsecore", "WelsDecPlus", "WelsEncPlus", "WelsVP")
    end
  end

  on_load(function (target)
    local is_gfx = has_config("FAN_WINDOW") or has_config("FAN_2D")
    local missing_base = not os.isfile("third_party/fan/.core.stamp")
    local missing_gfx  = is_gfx and not os.isfile("third_party/fan/.gfx.stamp")

    if missing_base or missing_gfx then
      print("Missing third_party headers. Running install.sh...")
      local args = {}
      if target:is_plat("wasm") then table.insert(args, "--wasm") end
      if not is_gfx then table.insert(args, "--core") end
      if os.execv("./install.sh", args) ~= 0 then raise("./install.sh failed") end
    end

    if target:is_plat("linux") then
      import("lib.detect.find_tool")
      if has_config("FAN_GUI") then
        if find_tool("pkg-config") then
          local libs = os.iorunv("pkg-config", {"--libs", "gtk+-3.0"})
          if libs then
            local flags = libs:split("%s+")
            table.sort(flags)
            for _, l in ipairs(flags) do
              if l:startswith("-l") then target:add("links", l:sub(3)) end
            end
          end
        end
      end
      if has_config("FAN_WAYLAND_SCREEN") then
        if not find_tool("pkg-config") then raise("FAN_WAYLAND_SCREEN needs pkg-config for dbus-1 and libpipewire-0.3") end
        local cflags = os.iorunv("pkg-config", {"--cflags", "dbus-1", "libpipewire-0.3"})
        for _, f in ipairs(cflags:split("%s+")) do
          if f:startswith("-I") then target:add("sysincludedirs", f:sub(3)) end
        end
      end
      if find_tool("mold") then target:add("ldflags", "-fuse-ld=mold", {force = true})
      elseif find_tool("gold") then target:add("ldflags", "-fuse-ld=gold", {force = true}) end
    end
  end)

  after_build(function (target)
    if has_config("buildlib") then
      os.mkdir("gcm.cache")
      for _, f in ipairs(os.files("build/**.gcm")) do os.cp(f, "gcm.cache") end
      os.cp(target:targetfile(), ".")
      print("Library artifacts copied to current directory.")
    end
  end)
target_end()

local marker = "fan_modules_info_printed.flag"
after_load(function (target)
  if target:name() ~= "a.exe" or os.isfile(marker) then return end
  io.writefile(marker, "1")
  print("Module files: " .. #module_files)
  print("Implementation files: " .. #impl_files)
  if #impl_files > 0 then
    print("Found implementations:")
    for i = 1, math.min(5, #impl_files) do print("  - " .. impl_files[i]) end
    if #impl_files > 5 then print("  ... and " .. (#impl_files - 5) .. " more") end
  end
end)
