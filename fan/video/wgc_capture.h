#pragma once

// Windows Graphics Capture for one window: the frames as GPU textures at the window's own rate, also when other
// windows cover it. PrintWindow, used before, made the window paint itself into memory every frame: a DirectX game
// (Unreal) came out at about 5 frames a second. Windows 10 1903 and later; open() returns null where it can't be used

#if defined(_WIN32)

#include <cstdint>

struct ID3D11Device;
struct ID3D11Texture2D;

namespace fan::graphics::wgc {
  struct session_t;

  // capture of hwnd into textures of device, null when it can't be (old Windows, a window it refuses)
  session_t* open(ID3D11Device* device, void* hwnd);
  void close(session_t* session);
  // the newest frame since the last call: its texture (a reference the caller releases) and the size of the
  // window's picture in it (the texture can be bigger). False when no new frame came
  bool take(session_t* session, ID3D11Texture2D** texture, std::uint32_t& width, std::uint32_t& height);
  // the top left of what the capture shows of hwnd, on the desktop: the visible frame, without the invisible resize
  // borders GetWindowRect counts (about 7 pixels at the sides and bottom), so the pointer drawn over it lines up
  bool visible_origin(void* hwnd, std::int32_t& x, std::int32_t& y);
}

#endif
