import fan;
import std;

using namespace fan::graphics;

fan::event::task_t task() {
    while(1) {
        fan::print(fan::random::string(5));
        co_await fan::co_sleep(1000);
    }
}

int main() {
    engine_t engine;
    auto t = task();
    rectangle_t rect({500, 500}, 32, fan::colors::red);
    engine.loop([]{
        
    });
}