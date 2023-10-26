#include fan_pch

#include <algorithm>

int compare(const void* a, const void* b) {
  return (*(int*)b - *(int*)a);
}

int* f(int* v, int size, int desired) {
  int total = 0;

  // Create a max heap
  qsort(v, size, sizeof(int), compare);

  while (total != desired) {
    // Remove the current maximum value from the heap
    int max_val = v[0];
    total -= max_val;
    v[0]--;

    // Restore the heap property
    int i = 0;
    int left, right, largest;
    while (1) {
      largest = i;
      left = 2 * i + 1;
      right = 2 * i + 2;

      if (left < size && v[left] > v[largest]) {
        largest = left;
      }

      if (right < size && v[right] > v[largest]) {
        largest = right;
      }

      if (largest != i) {
        // Swap v[i] and v[largest]
        int temp = v[i];
        v[i] = v[largest];
        v[largest] = temp;
        i = largest;
      }
      else {
        break;
      }
    }

    // Add the updated maximum value back to the heap
    total += max_val;
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
  auto o = f(input.data(), input.size(), desired);
  fan::print("");
  fan::print_no_endline("output:");
  for (auto i : std::vector<int>(o, o + input.size())) {
    fan::print_no_endline(i);
  }
}