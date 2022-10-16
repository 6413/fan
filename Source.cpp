#include <iostream>
#include <intrin.h>
using namespace std;

#pragma intrinsic(_BitScanReverse)

int main()
{
  unsigned long mask = 4;
  unsigned long index;
  unsigned char isNonzero;

  cout << "Enter a positive integer as the mask: " << flush;
  isNonzero = _BitScanReverse(&index, mask);
  if (isNonzero)
  {
    cout << "Mask: " << mask << " Index: " << 31 - index << endl;
  }
  else
  {
    cout << "No set bits found.  Mask is zero." << endl;
  }
}