#include fan_pch

void f(const fan::vec2& v) {

}
void f(const fan::vec3& v) {

}

int main() {
  f(fan::vec3() + fan::vec3());
}