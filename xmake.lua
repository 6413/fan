set_project("fan")
set_languages("cxx23")

if is_plat("wasm") then
  add_cxxflags("-pthread", "-DFAN_WASM", {force = true})
  add_ldflags("-s USE_GLFW=3", "-s ASYNCIFY=1", "-pthread", "-s PTHREAD_POOL_SIZE=4", {force = true})
else
  option("compiler") set_default("clang") option_end()
  set_toolchains(get_config("compiler") == "gcc" and "gcc" or "clang")
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
  add_cxflags("-O3", {force = true})
  add_ldflags("-s", {force = true})
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

  FAN_VULKAN = true,
  FAN_FMT = false,
  FAN_WAYLAND_SCREEN = false,
  FAN_NETWORK = false,
  FAN_AUDIO = false,
  FAN_VIDEO = false,
  FAN_REFLECTION = false
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

if has_config("FAN_REFLECTION") then add_cxxflags("-freflection", {force = true}) end

option("FAN_USE_STD_MODULE") set_default(false) set_showmenu(true) add_defines("FAN_USE_STD_MODULE") option_end()
option("main") set_default("examples/engine_demos/engine_demo.cpp") option_end()

local static_req = {system = false, configs = {shared = false}}
if has_config("FAN_FMT") then add_requires("fmt 10.2.1", static_req) end
if has_config("FAN_VULKAN") then
  add_defines("VK_ENABLE_BETA_EXTENSIONS")
  add_requires("vulkan-headers v1.4.335", {system = false})
  add_requires("shaderc", static_req)
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
if has_config("FAN_VULKAN") then add_sysincludedirs("third_party/fan/include/VulkanMemoryAllocator/include", {public = true}) end

local is_gcc = get_config("compiler") == "gcc"
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

if is_gcc then
  add_cxxflags("-fmax-errors=20", "-fmodules-ts", "-fno-module-lazy", {force = true})
else
  add_cxxflags("-ferror-limit=20", "-Wno-include-angled-in-module-purview", "-fmacro-backtrace-limit=0", "-Wno-shift-op-parentheses", "-Wno-int-to-void-pointer-cast", "-Wno-bitwise-op-parentheses", {force = true})
end

if has_config("FAN_GUI") then
  add_defines("IMGUI_DEFINE_MATH_OPERATORS", "IMGUI_DISABLE_SSE", "STBI_NO_SIMD")
end

local module_files = {
  "fan/types/types.ixx", "fan/types/color.ixx", "fan/types/vector.ixx", "fan/types/quaternion.ixx",
  "fan/types/matrix.ixx", "fan/types/fstring.ixx", "fan/types/compile_time_string.ixx",
  "fan/types/flat_hash_map.ixx", "fan/types/bitset.ixx", "fan/memory/memory.ixx",
  "fan/math/math.ixx", "fan/math/intersection.ixx", "fan/time.ixx", "fan/mpl.ixx",
  "fan/utility.ixx", "fan/formatter.ixx", "fan/print_error.ixx", "fan/print.ixx",
  "fan/random.ixx", "fan/log_dispatcher.ixx", "fan/crypto.ixx", "fan/process.ixx",
  "fan/io/io_types.ixx", "fan/io/directory.ixx", "fan/io/file.ixx", "fan/io/io_prompt.ixx",
  "fan/event/event_types.ixx", "fan/event/event.ixx", "fan/event/uv_raw.ixx", "fan/compression.ixx",
  "fan/tween.ixx"
}

local feature_modules = {
  FAN_WINDOW = {
    "fan/window/window.ixx", "fan/window/input_common.ixx", "fan/window/input.ixx", "fan/window/input_action.ixx",
    "fan/graphics/common_types.ixx", "fan/graphics/material.ixx", "fan/graphics/camera.ixx", "fan/graphics/image_load.ixx",
    "fan/graphics/webp.ixx", "fan/graphics/stb.ixx", "fan/graphics/common_context.ixx",
    "fan/graphics/audio_subsystem.ixx", "fan/graphics/physics_subsystem.ixx", "fan/graphics/input_subsystem.ixx",
    "fan/graphics/loco.ixx", "fan/graphics/graphics.ixx", "fan/graphics/file_dialog.ixx",
    "fan/graphics/2D/algorithm/raycast_grid.ixx", "fan/graphics/gameplay/gameplay_types.ixx",
    "fan/graphics/gameplay/gameplay.ixx", "fan/graphics/graphics_event.ixx", "fan/texture_pack/tp0.ixx",
    "fan/physics/physics_types.ixx", "fan/noise.ixx", "fan/pathfind.ixx", "fan/spatial.ixx", "fan/ecs.ixx",
    "fan/graphics/gui/console.ixx",
    "fan/graphics/gui/tilemap_editor/loader.ixx", "fan/graphics/gui/tilemap_editor/renderer0.ixx"
  },

  FAN_VULKAN = { "fan/graphics/vulkan/vk_core_types.ixx", "fan/graphics/vulkan/vk_core_vai.ixx", "fan/graphics/vulkan/vk_core_image.ixx", "fan/graphics/vulkan/vk_core_compute.ixx", "fan/graphics/vulkan/vk_core_pipeline.ixx", "fan/graphics/vulkan/vk_core_camera_subsystem.ixx", "fan/graphics/vulkan/vk_core_uniform_block.ixx", "fan/graphics/vulkan/vk_core_shader_subsystem.ixx", "fan/graphics/vulkan/vk_core.ixx" },
  FAN_2D = { "fan/graphics/2D/shapes_types.ixx", "fan/graphics/2D/grid_placer.ixx", "fan/graphics/2D/culling.ixx", "fan/graphics/2D/shapes.ixx" },
  FAN_JSON = { "fan/types/json.ixx" },
  FAN_FMT = { "fan/fmt.ixx" },
  FAN_NETWORK = { "fan/network/network.ixx", "fan/network/network_socket.ixx", "fan/graphics/2D/graphics_network.ixx" },
  FAN_AUDIO = { "fan/audio/audio.ixx" },
  FAN_PHYSICS_2D = { "fan/physics/b2_integration.ixx", "fan/physics/physics_common_context.ixx", "fan/graphics/physics_shapes.ixx" },
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
    "fan/graphics/scene.ixx"
  },
  FAN_3D = { "fan/graphics/3D/objects/fms.ixx", "fan/graphics/voxel.ixx" },
  FAN_VIDEO = { "fan/video/codec.ixx", "fan/video/screen.ixx", "fan/video/renderer.ixx", "fan/video/video.ixx" },
  FAN_WAYLAND_SCREEN = { "fan/video/screen_codec.ixx" }
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

if has_config("FAN_VULKAN") then
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
    if has_config("FAN_VULKAN") then
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
    if has_config("FAN_VULKAN") then add_files("fan/imgui/imgui_impl_vulkan.cpp") end
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

  if has_config("FAN_VULKAN") then
    add_packages("vulkan-headers", "shaderc")
    if is_plat("linux") then add_syslinks("vulkan") end
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
  if not has_config("buildlib") then add_files(get_config("main")) end

  add_includedirs(".", {public = true})
  add_sysincludedirs("third_party/fan/include", {public = true})
  if has_config("FAN_VULKAN") then
    add_includedirs(
      "third_party/fan/include/VulkanMemoryAllocator/include",
      {public = true}
    )
  end
  add_linkdirs("third_party/fan/lib")

  if is_plat("linux") then
    add_links("stdc++exp")
    if has_config("static_runtime") then add_ldflags("-static-libstdc++", "-static-libgcc", {force = true}) end
    add_packages("libuv")
    if has_config("FAN_WINDOW") then
      add_packages("glfw", "zlib", "libpng", "libwebp")
    end

    if has_config("FAN_AUDIO") then add_links("opus", "pulse-simple", "pulse") end
    if has_config("FAN_NETWORK") then add_links("ssl", "crypto", "curl") end
    if has_config("FAN_GUI") then add_packages("freetype", "lunasvg") end
    if has_config("FAN_PHYSICS_2D") then add_packages("box2d") end
    if has_config("FAN_3D") then add_links("assimp") end
    if has_config("FAN_VULKAN") then add_packages("vulkansdk") end
    if has_config("FAN_WAYLAND_SCREEN") then add_links("wayland-client", "pipewire-0.3", "dbus-1", "avcodec", "avutil", "swscale") end
  elseif is_plat("windows") then
    add_linkdirs("lib/GLFW", "lib/GLEW", "lib/libuv", "lib/libwebp", "lib/opus", "lib/openssl")
    add_links("glfw3_mt", "glew32s", "uv_a", "libwebp", "opus", "libssl", "libcrypto")
    if has_config("FAN_GUI") then add_linkdirs("lib/freetype", "lib/lunasvg") add_links("freetype", "lunasvg") end
    if has_config("FAN_PHYSICS_2D") then add_linkdirs("lib/box2d") add_links("box2d") end
    if has_config("FAN_3D") then add_linkdirs("C:/Program Files/Assimp/lib/x64") add_links("assimp-vc143-mt") end
    if has_config("FAN_VULKAN") then add_packages("vulkansdk") end
    if has_config("FAN_WAYLAND_SCREEN") then
      add_linkdirs("lib/libx264", "lib/openh264")
      add_links("DXGI", "D3D11", "libx264", "welsdcore", "welsecore", "WelsDecPlus", "WelsEncPlus", "WelsVP")
    end
  end

  on_load(function (target)
    local is_gfx = has_config("FAN_WINDOW") or has_config("FAN_VULKAN")
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