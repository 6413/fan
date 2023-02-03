// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include _FAN_PATH(types/vector.h)

//#define loco_vulkan

#define loco_window
#define loco_context

#include <future>
#include <thread>
#include <chrono>
#include <array>
#include <iostream>
#include <random>

size_t doHeavyWork(size_t arg)
{
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<unsigned int> rnd(333, 777);
  //simulate heavy work...
  std::this_thread::sleep_for(std::chrono::milliseconds(rnd(rng)));
  return arg * 33;
}

//wrapper function for checking whether a task has finished and
//the result can be retrieved by a std::future
template<typename R>
bool isReady(const std::future<R>& f)
{
  return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

int main()
{
  constexpr size_t numTasks = 5;

  std::array<std::future<size_t>, numTasks> futures;

  size_t index = 1;
  for (auto& f : futures)
  {
    f = std::async(std::launch::async, doHeavyWork, index);
    index++;
  }

  std::array<bool, numTasks> finishedTasks;
  for (auto& i : finishedTasks)
    i = false;

  size_t numFinishedTasks = 0;

  do
  {
    for (size_t i = 0; i < numTasks; ++i)
    {
      if (!finishedTasks[i] && isReady(futures[i]))
      {
        finishedTasks[i] = true;
        numFinishedTasks++;
        std::cout << "task " << i << " ended with result " << futures[i].get() << '\n';
      }
    }
  } while (numFinishedTasks < numTasks);

  std::cin.get();
}