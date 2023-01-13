// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#include <string>

template <typename ...Args>
void print(const Args&... args) {
  ((std::cout << args), ...) << '\n';
}

class Asiakastili {
public:
  void LisaaAsiakas(int tilinumero, int saldo, const std::string& nimi) {
    t = tilinumero;
    s = saldo;
    n = nimi;
  }
  void NaytaTiedot() const {
    print("Tilinumero: ", t);
    print("Asiakkaan nimi: ", n);
    print("Tilin saldo: ", s);
  }
  void MuutaSaldoa(int saldo) {
    s = saldo;
  }
  bool VertaaTiliNro(int tilinro) const {
    return t == tilinro;
  }
private:
  int t;
  int s;
  std::string n;
};

int main() {
  return 0;
}