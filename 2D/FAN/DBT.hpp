#pragma once
#include <FAN/Alloc.hpp>

/*
////////////

DEFINE DBT BEFORE USING - #define _DBT

////////////
*/

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
	type* output;
	bool init;
};

template <typename type>
class DBT {
public:
	vector<keytype<type>> nodes;
	constexpr DBT() : nodes(1), _Size(0) { }
	constexpr DBT(size_t _Reserve) : nodes(_Reserve), _Size(_Reserve) {}
	~DBT() {
		/*for (int i = 0; i < nodes.size(); i++) {
			if (nodes.data() + i) {
				if (nodes[i].output) {
					nodes[i].output = 0;
					delete nodes[i].output;
					nodes[i].output = NULL;
				}
			}
		}*/
	}
	constexpr void find_road(size_t& node, unsigned char*& in, size_t& inlen, size_t& seek) {
		while (inlen) {
			size_t tnode = nodes[node].nextnode[separate_bits(*(unsigned char*)in, seek)];
			if (!tnode)
				return;
			node = tnode;
			D_magic(in, inlen, seek);
		}
	}
	constexpr void dbt_in(unsigned char* in, size_t inlen, type output) {
		size_t node = 0;
		size_t seek = 0;
		find_road(node, in, inlen, seek);
		while (inlen) {
			node = nodes[node].nextnode[separate_bits(*(unsigned char*)in, seek)] = nodes.current();
			nodes.push_back(keytype<type>());
			D_magic(in, inlen, seek);
		}
		nodes[node].output = new type;
		*nodes[node].output = output;
		_Size++;
	}
	constexpr auto dbt_out(unsigned char* in, size_t inlen) {
		size_t node = 0;
		size_t seek = 0;
		find_road(node, in, inlen, seek);
		return inlen ? (type*)(0) : (nodes.data() + node)->output ? (nodes.data() + node)->output : (type*)(0);
	}
	constexpr bool empty() const {
		return !_Size;
	}
	constexpr size_t size() const {
		return _Size;
	}
private:
	size_t _Size;
};