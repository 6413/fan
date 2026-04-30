#include <cstdint>
#include <coroutine>
#include <cstdio>
#include <string>
import fan;
import fan.ipc;

struct message_t {
  uint32_t id;
  char text[124];
};
using fast_queue_t = fan::ipc::ring_buffer_t<message_t, 1024>;

fan::event::task_t receiver_run(fast_queue_t* queue, fan::ipc::async_consumer_t& consumer) {
  while (1) {
    co_await consumer.wait();
    message_t msg;
    while (queue->try_pop(msg)) {
      fan::print("received:", msg.id, msg.text);
    }
  }
}

int main() {
  fan::ipc::shared_memory_t shm("my_fast_ipc", sizeof(fast_queue_t), true);
  auto* queue = new (shm.ptr) fast_queue_t{};
  fan::ipc::async_consumer_t consumer("my_fast_evt");
  fan::ipc::async_producer_t producer("my_fast_evt");

  auto task = receiver_run(queue, consumer);

  auto after_task = fan::event::after(500, [&] {
    for (uint32_t i = 0; i < 10; ++i) {
      message_t msg;
      msg.id = i;
      snprintf(msg.text, sizeof(msg.text), "hello %u", i);
      queue->push(msg);
      producer.signal();
    }
  });

  fan::event::loop();

}