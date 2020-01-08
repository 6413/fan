#pragma once
#include <FAN/Alloc.hpp>

template<typename T1, typename T2>
constexpr auto separate_bits(T1 px, T2  py) {
	return (((px) >> (py)) & 1);
}

constexpr void D_magic(unsigned char*& in, size_t& insize, size_t& seek) {
	in += seek == 7;
	insize--;
	seek = seek == 7 ? 0 : seek + 1;
}

template <typename type>
struct keytype {
	constexpr keytype() : nextnode{ 0 }, output(0), init(0) {}
	size_t nextnode[2];
	type output;
	bool init;
};

template <typename type>
struct dbt {
	Alloc<keytype<type>> nodes;
	constexpr dbt() : nodes(1) { }
	constexpr dbt(size_t reserve) : nodes(reserve) {}

	constexpr void find_road(size_t& node, unsigned char* in, size_t& inlen, size_t& seek) {
		while (inlen) {
			size_t tnode = nodes[node].nextnode[separate_bits(*(unsigned char*)in, seek)];
			if (!tnode)
				return;
			node = tnode;
			D_magic(in, inlen, seek);
		}
	}
	constexpr void push(unsigned char* in, size_t inlen, type output) {
		size_t node = 0;
		size_t seek = 0;
		find_road(node, in, inlen, seek);
		while (inlen) {
			node = nodes[node].nextnode[separate_bits(*(unsigned char*)in, seek)] = nodes.current();
			nodes.push_back(keytype<type>());
			D_magic(in, inlen, seek);
		}
		nodes[node].output = output;
		nodes[node].init = true;
	}
	constexpr auto search(unsigned char* in, size_t inlen) {
		size_t node = 0;
		size_t seek = 0;
		find_road(node, in, inlen, seek);
		return inlen ? 0 : nodes[node].init ? nodes[node].output : 0;
	}
};