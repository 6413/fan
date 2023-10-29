#include fan_pch

void common_uintptr_Balance(uintptr_t* v, uintptr_t size, uintptr_t desired) {
  sintptr_t total = 0;
  while (true) {
    total = 0;
    sintptr_t idx = -1;
    sintptr_t m = -10000000;
    for (sintptr_t i = 0; i < size; ++i) {
      if (m < (int64_t)v[i]) {
        m = v[i];
        idx = i;
      }
      total += v[i];
    }
    if (total <= desired) {
      break;
    }
    --v[idx];
  }
}

int main() {
  std::vector<uintptr_t> v{35, 2, 88};
  common_uintptr_Balance(v.data(), v.size(), 100);
  for (auto i : v) {
    fan::print(i);
  }
}