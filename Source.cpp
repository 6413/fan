#include fan_pch

void common_uintptr_Balance(uintptr_t* v, uintptr_t size, uintptr_t desired) {
  int64_t total = 0;
  while (true) {
    total = 0;
    int64_t idx = -1;
    int64_t m = -10000000;
    for (int64_t i = 0; i < size; ++i) {
      if (m < (int64_t)v[i]) {
        m = v[i];
        idx = i;
      }
      total += v[i];
    }
    int64_t m2 = -10000000;
    int64_t idx2 = -1;
    int64_t total_minus = 0;
    for (int64_t i = 0; i < size; ++i) {
      if (i != idx && m2 < (int64_t)v[i]) {
        total_minus = m - m2;
        m2 = v[i];
        idx2 = i;
      }
    }
    if (total - (m - m2) < desired) {
      total_minus = desired - ( - (m ;
    }
    else {
      total_minus = m - m2;
    }
    total -= total_minus;
    v[idx] -= total_minus;
    if (total <= desired) {
      break;
    }
  }
}

int main() {
  std::vector<uintptr_t> v{35, 2, 88};
  common_uintptr_Balance(v.data(), v.size(), 100);
  for (auto i : v) {
    fan::print(i);
  }
}