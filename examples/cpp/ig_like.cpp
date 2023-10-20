#include fan_pch

#include _FAN_PATH(system.h)

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

Mat img; Mat templ, templ2; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";

int match_method = TM_SQDIFF;
int max_Trackbar = 5;

Mat hwnd2mat(HWND hwnd)
{
  HDC hwindowDC, hwindowCompatibleDC;

  int height, width, srcheight, srcwidth;
  HBITMAP hbwindow;
  Mat src;
  BITMAPINFOHEADER  bi;

  hwindowDC = GetDC(hwnd);
  hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
  SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

  RECT windowsize;
  GetClientRect(hwnd, &windowsize);

  srcheight = windowsize.bottom;
  srcwidth = windowsize.right;
  height = windowsize.bottom / 1;
  width = windowsize.right / 1;

  src.create(height, width, CV_8UC4);

  // create a bitmap
  hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
  bi.biSize = sizeof(BITMAPINFOHEADER);
  bi.biWidth = width;
  bi.biHeight = -height;
  bi.biPlanes = 1;
  bi.biBitCount = 32;
  bi.biCompression = BI_RGB;
  bi.biSizeImage = 0;
  bi.biXPelsPerMeter = 0;
  bi.biYPelsPerMeter = 0;
  bi.biClrUsed = 0;
  bi.biClrImportant = 0;

  SelectObject(hwindowCompatibleDC, hbwindow);
  StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY);
  GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

  DeleteObject(hbwindow);
  DeleteDC(hwindowCompatibleDC);
  ReleaseDC(hwnd, hwindowDC);

  return src;
}

fan::vec2i get_result(cv::Mat& tmp) {
  cv::Mat img_gray, templ_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_BGRA2RGB);
  matchTemplate(img_gray, tmp, result, match_method);

  normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

  if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
  {
    matchLoc = minLoc;
  }
  else
  {
    matchLoc = maxLoc;
  }
  return *(fan::vec2i*)&matchLoc;
}

void MatchingMethod() {
  Mat img_display;
  img.copyTo(img_display);

  int result_cols = img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;

  result.create(result_rows, result_cols, CV_32FC1);

  bool do_skip = false;

  fan::vec2i click_src = get_result(templ);

  if (click_src != 0) {
    fan::sys::input::set_mouse_position(click_src + templ.cols / 2);
    uint64_t random_delay = fan::random::value_i64(1e+9, 5e+9);
    fan::delay(fan::time::nanoseconds(random_delay));
    fan::sys::input::send_mouse_event(fan::mouse_left, fan::mouse_state::press);
    fan::delay(fan::time::nanoseconds(0.05e+9));
    fan::sys::input::send_mouse_event(fan::mouse_left, fan::mouse_state::release);
    do_skip = true;
  }

  fan::sys::input::set_mouse_position(fan::vec2i(1872, 570));
  uint64_t random_delay = fan::random::value_i64(1e+9, 5e+9);
  fan::delay(fan::time::nanoseconds(random_delay));
  fan::sys::input::send_mouse_event(fan::mouse_left, fan::mouse_state::press);
  fan::delay(fan::time::nanoseconds(0.05e+9));
  fan::sys::input::send_mouse_event(fan::mouse_left, fan::mouse_state::release);

  return;
}

int main(int argc, char** argv) {
  templ = imread("search.jpg", 1);
  templ2 = imread("search1.jpg", 1);
  fan::sys::input input;

  HWND hwndDesktop = GetDesktopWindow();

  fan::print("press f2 to start");
  fan::print("press f9 force quit");

  input.listen_keyboard([&](uint16_t key, fan::keyboard_state state, bool action) {

    if (state != fan::keyboard_state::press) {
      return;
    }

    if (!action) {
      return;
    }

    switch (key) {
      case fan::key_f2: {
        auto f = [&] {
          while (1) {
            img = hwnd2mat(hwndDesktop);
            MatchingMethod();
            fan::delay(fan::time::nanoseconds(.5e+9));
          }
          };

        std::thread t(f);

        t.detach();

        break;
      }
      case fan::key_f9: {
        exit(0);
      }
    }

    });

  input.thread_loop([] {});


  return 0;
}