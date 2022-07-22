#include <set>
#include <stdexcept>

using node_value_type = uint32_t;

typedef std::multiset<node_value_type>::iterator LHB_node_t;

struct lhb_t{

	using type_t = std::multiset<node_value_type>;

	void open() {
		set = new type_t;
	}
	void close() {
		set = new type_t;
	}
	void push(node_value_type value) {
		set->insert(value);
	}
	node_value_type find_lowest(node_value_type value) const {
		auto found = set->lower_bound(value);
		if (found == set->end()) {
			fan::throw_error("invalid node");
		}
		return *found;
	}

	//void erase(node_value_type value) {
	//	set->insert(value);
	//}

	type_t* set;
};

//bool LHB_is_node_invalid(LHB_t *lhb, LHB_node_t node){
//	return node == lhb->set.end();
//}
//
//void LHB_open(LHB_t *lhb, node_value_type size){
//
//}
//void LHB_close(LHB_t *lhb){
//	lhb->set.clear();
//}
//
//LHB_node_t LHB_lo(LHB_t *lhb, node_value_type v){
//	auto found = lhb->set.lower_bound(v);
//	if(!LHB_is_node_invalid(lhb, found)){
//		return lhb->set.insert(found, v);
//	}
//	else{
//		return lhb->set.insert(v);
//	}
//}
//
//LHB_node_t LHB_hi(LHB_t *lhb, node_value_type v){
//	auto found = lhb->set.upper_bound(v);
//	if(!LHB_is_node_invalid(lhb, found)){
//		return lhb->set.insert(found, v);
//	}
//	else{
//		return lhb->set.insert(v);
//	}
//}
//
//LHB_node_t LHB_link_next(LHB_t *lhb, LHB_node_t node, node_value_type v){
//	throw std::runtime_error("not implemented");
//}
//LHB_node_t LHB_link_prev(LHB_t *lhb, LHB_node_t node, node_value_type v){
//	throw std::runtime_error("not implemented");
//}
//
//LHB_node_t LHB_link_src(LHB_t *lhb, node_value_type v){
//	return lhb->set.begin();
//}
//LHB_node_t LHB_link_dst(LHB_t *lhb, node_value_type v){
//	return lhb->set.end();
//}
//
//LHB_node_t LHB_link(LHB_t *lhb, node_value_type v){
//	return lhb->set.insert(v);
//}
//
//node_value_type LHB_out_num(LHB_t *lhb, LHB_node_t node){
//	auto found = lhb->set.find(*node);
//	if (!LHB_is_node_invalid(lhb, found)) {
//		return *found;
//	}
//	throw std::runtime_error("value not found");
//}
//uint8_t *LHB_out(LHB_t *lhb, LHB_node_t node){
//	throw std::runtime_error("not implemented");
//}
//
//LHB_node_t LHB_node_begin(LHB_t *lhb){
//	return lhb->set.begin();
//}
//LHB_node_t LHB_node_end(LHB_t *lhb){
//	return lhb->set.end();
//}
//LHB_node_t LHB_node_iterate(LHB_t *lhb, LHB_node_t node){
//	return std::next(node);
//}
//LHB_node_t LHB_node_riterate(LHB_t *lhb, LHB_node_t node){
//	return std::prev(node);
//}
