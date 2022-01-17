#pragma once

#include <vector>
#include <cstdint>

#define BLL_set_debug_InvalidAction 1

#ifndef BLL_set_SafeNext
	#define BLL_set_SafeNext 1
#endif

// node type amount of nodes allowed
template <typename type_t, typename node_type_t = uint32_t>
struct bll_t {

	bll_t() {
		open();
	}

	void open() {

		nodes.resize(2);

		src = 0;
		dst = 1;

		e.c = 0;
		e.p = 0;

		get_node_by_reference(src)->next = dst;
		get_node_by_reference(dst)->prev = src;

		#if BLL_set_SafeNext
			SafeNext = (node_type_t)-1;
		#endif

	}

	const constexpr type_t operator[](node_type_t node_reference) const {
		return get_node_by_reference(node_reference)->data;
	}

	constexpr type_t& operator[](node_type_t node_reference) {
		return get_node_by_reference(node_reference)->data;
	}

	constexpr node_type_t new_node_empty() {
		node_type_t node_reference = e.c;
		e.c = get_node_by_reference(node_reference)->next;
		e.p--;
		return node_reference;
	}

	constexpr void new_node() {
		if(e.p){
			return new_node_empty();
		}
		else{
			return new_node_alloc();
		}
	}

	// push front
	constexpr node_type_t new_node_first_empty() {
		node_type_t node_reference = new_node_empty();
		node_type_t src_node_reference = src;
		get_node_by_reference(node_reference)->next = src_node_reference;
		get_node_by_reference(src_node_reference)->prev = node_reference;

		src = node_reference;
		return src_node_reference;
	}

	constexpr node_type_t push_front(const type_t& value) {

		node_type_t reference = new_node_first();
		(*this)[reference] = value;

		return reference;
	}

	constexpr node_type_t new_node_first_alloc() {
		node_type_t node_reference = new_node_alloc();
		node_type_t src_node_reference = src;
		get_node_by_reference(node_reference)->next = src_node_reference;
		get_node_by_reference(src_node_reference)->prev = node_reference;

		src = node_reference;
		return src_node_reference;
	}

	constexpr node_type_t new_node_first() {
		if(e.p){
			return new_node_first_empty();
		}
		else{
			return new_node_first_alloc();
		}
	}

	constexpr node_type_t new_node_last_empty() {

		node_type_t node_reference = new_node_empty();
		node_type_t dst_node_reference = dst;
		get_node_by_reference(node_reference)->prev = dst_node_reference;
		get_node_by_reference(dst_node_reference)->next = node_reference;

		dst = node_reference;
		return dst_node_reference;
	}

	constexpr node_type_t new_node_last_alloc() {

		node_type_t node_reference = new_node_alloc();
		node_type_t dst_node_reference = dst;
		get_node_by_reference(node_reference)->prev = dst_node_reference;
		get_node_by_reference(dst_node_reference)->next = node_reference;

		dst = node_reference;
		return dst_node_reference;
	}

	constexpr node_type_t new_node_last() {
		if(e.p){
			return new_node_last_empty();
		}
		else{
			return new_node_last_alloc();
		}
	}

	constexpr node_type_t push_back(const type_t& value) {
		node_type_t reference = new_node_last();
		(*this)[reference] = value;

		return reference;
	}

	// src->dst
	constexpr void link_next(node_type_t src, node_type_t dst) {
		node_t *src_node = get_node_by_reference(src);
		node_t *dst_node = get_node_by_reference(dst);

		node_type_t next_node_reference = src_node->next;
		node_t *next_node = get_node_by_reference(next_node_reference);
		src_node->next = dst;
		dst_node->prev = src;
		dst_node->next = next_node_reference;
		next_node->prev = dst;
	}

	constexpr void link_prev(node_type_t src, node_type_t dst) {
		node_t *src_node = get_node_by_reference(src);

		node_type_t prev_node_reference = src_node->prev;

		node_t *prev_node = get_node_by_reference(prev_node_reference);
		prev_node->next = dst;
		node_t* dst_node = get_node_by_reference(dst);
		dst_node->prev = prev_node_reference;
		dst_node->next = src;
		src_node->prev = dst;
	}

	constexpr void unlink(node_type_t node_reference) {
		#if BLL_set_debug_InvalidAction == 1
				assert(node_reference != src);
				assert(node_reference != dst);
		#endif

		node_t *node = get_node_by_reference(node_reference);

		#if BLL_set_SafeNext
				if(this->SafeNext == node_reference){
					this->SafeNext = node->prev;
				}
		#endif

		node_type_t next_node_reference = node->next;
		node_type_t prev_node_reference = node->prev;
		get_node_by_reference(prev_node_reference)->next = next_node_reference;
		get_node_by_reference(next_node_reference)->prev = prev_node_reference;

		node->next = e.c;
		node->prev = -1;
		e.c = node_reference;
		e.p++;

	}

	constexpr void erase(node_type_t node_reference) {
		unlink(node_reference);
	}

	constexpr node_type_t get_node_first() {
		return get_node_by_reference(src)->next;
	}

	constexpr node_type_t get_node_last() {
		return get_node_by_reference(dst)->prev;
	}

	constexpr bool is_node_reference_fronter(node_type_t src, node_type_t dst) {
		do{
			node_t* src_node = get_node_by_reference(src);
			src = src_node->next;
			if(src == dst){
				return 0;
			}
		}while(src != dst);
		return 1;
	}

	constexpr node_type_t new_node_alloc() {

		nodes.push_back({});
		return nodes.size() - 1;
	}

	struct{
		node_type_t c;
		node_type_t p;
	}e;
	
	constexpr auto size() const {
		return nodes.size() - e.p - 2;
	}

	const constexpr node_type_t begin() {
		return this->get_node_by_reference(this->src)->next;
	}

	const constexpr node_type_t end() {
		return this->dst;
	}

	/*void iterator::operator++(iterator it) {

	}*/

	constexpr node_type_t prev(node_type_t reference) {

#if BLL_set_debug_InvalidAction == 1

		if (reference == src) {
			assert(0);
		}

		if (reference == dst) {
			assert(0);
		}

		if (reference >= nodes.size()) {
			assert(0);
		}

		auto x = this->get_node_by_reference(reference)->next;

		if (x == src) {
			assert(0);
		}

#endif

		return this->get_node_by_reference(reference)->prev;
	}

	constexpr node_type_t next(node_type_t reference) {

		#if BLL_set_debug_InvalidAction == 1

			if (reference == dst) {
				assert(0);
			}

			if (reference >= nodes.size()) {
				assert(0);
			}

			auto x = this->get_node_by_reference(reference)->next;

			if (x == src) {
				assert(0);
			}

		#endif

		return this->get_node_by_reference(reference)->next;
	}

	constexpr void clear() {
		nodes.clear();

		this->open();
	}

	struct node_t {

		node_type_t next;
		node_type_t prev;

		type_t data;
	};

	constexpr node_t* get_node_by_reference(node_type_t node_reference) {

	#if BLL_set_debug_InvalidAction == 1
		if (node_reference >= nodes.size()) {
			assert(0);
		}
	#endif

		return (node_t*)(&nodes[node_reference]);
	}
	const constexpr node_t* get_node_by_reference(node_type_t node_reference) const {
		return (node_t*)(&nodes[node_reference]);
	}

	constexpr bool node_exists(node_t *node){
		if(node->prev == -1){
			return 1;
		}
		return 0;
	}

	constexpr bool node_reference_exists(node_type_t node_reference) {
		return node_exists(nodes[node_reference]);
	}

#if BLL_set_SafeNext
	void start_safe_next(node_type_t NodeReference){
#if BLL_set_debug_InvalidAction == 1
		if(this->SafeNext != (node_type_t)-1) {
			assert(0);
		}
#endif
		this->SafeNext = NodeReference;
	}
	node_type_t end_safe_next(){
		#if BLL_set_debug_InvalidAction == 1

			if(this->SafeNext == (node_type_t)-1) {
				assert(0);
			}

		#endif

		node_t *Node = get_node_by_reference(this->SafeNext);
		this->SafeNext = (node_type_t)-1;
		return Node->next;
	}
#endif

//protected:											

#if BLL_set_SafeNext

	node_type_t SafeNext;

#endif

	node_type_t src;
	node_type_t dst;

	std::vector<node_t> nodes;
};

#undef BLL_set_SafeNext

#undef BLL_set_debug_InvalidAction
