struct st0_t {

};

struct st1_t {

};


struct empty {

};

template <typename T>
struct protc_t_ {

	using type = T;

	protc_t_() = default;
	protc_t_(int y) :x(y) {}
	int x;
};

using protc_t = protc_t_<empty>;

template <typename in>
struct common {

};

struct protc2s_t : common<protc2s_t>{
	protc_t_<st1_t> a;
	protc_t b = 1;
	protc_t_<st0_t> c;
};

template <typename T>
void f(protc_t_<T> p, T x) {

}

int main() {
	protc2s_t x;
	f(x.c, st0_t());
	f(x.a, st0_t());
}