#include fan_pch

#include <algorithm>

std::vector<int> f(std::vector<int>& v, int desired) {
  std::vector<int> v2 = v;
  int total = std::accumulate(v.begin(), v.end(), 0);
  int num_elements = v.size();

  if (total == desired) {
    return v;
  }

  int average = desired / num_elements;
  int remainder = desired % num_elements;

  for (int& num : v) {
    if (num > average) {
      num = average;
    }
    else if (remainder > 0 && num == average) {
      num++;
      remainder--;
    }
  }

  int sum = 0;
  int i = 0;
  while (sum != desired) {
    sum = 0;
    for (int i = 0; i < v.size(); ++i) {
      if (v2[i] > v[i]) {
        v[i]++;
      }
      sum += v[i];
    }
  }

  return v;
}


int main() {
  int desired = 100;
  std::vector<int> input{45, 2, 88};
  fan::print("desired:", desired);
  fan::print_no_endline("input:");
  for (auto i : input) {
    fan::print_no_endline(i);
  }
  auto o = f(input, desired);
  fan::print("");
  fan::print_no_endline("output:");
  for (auto i : o) {
    fan::print_no_endline(i);
  }
}