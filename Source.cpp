#include <iostream>
#include <fan/types/types.h>

template<typename Callable>
using _return_type_of_t = typename decltype(std::function{ std::declval<Callable>() })::result_type;
template <typename type = long double>
struct ProtocolC_t {
	/*
	using m_type = _return_type_of_t<decltype([] {
		type v;
		return v;
	})>;
	*/
	struct m_type : _return_type_of_t<decltype([] {
		type v;
		return v;
	}) > {};
		/* data struct size */
		uint32_t m_DSS = fan::conditional_value_t<std::is_same<type, long double>::value, 0, sizeof(m_type)>::value;
};
#define _ProtocolC_t(p0) \
  ProtocolC_t<_return_type_of_t<decltype([] { \
    p0 v; \
    return v;\
  })>>
int main() {
	_ProtocolC_t(struct {
		uint64_t ClientIdentify;
		uint64_t ServerIdentify;
	}) InformInvalidIdentify;
	InformInvalidIdentify.m_DSS = 5;
	//ProtocolC_t < > p;

	//using type = decltype(p)::type;

	//struct s_t : fan::return_type_of_t<decltype([] { \
	//	type v;
	//return v;
	//	}) > {

	//};

	//decltype(p)::type x;
	//
	//return x.a;

	//make_prot(struct {
	//	uint64_t a;
	//	uint64_t b;
	//})x;

	//make_prot(struct {
	//	uint64_t a;
	//	uint64_t b;
	//})y;

	//auto x = make_protocol((struct a{};) {
	//	uint64_t a;
	//	uint64_t b;
	//});


	//protc2s_t x;

}