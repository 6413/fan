typedef BLL_set_type_node CONCAT2(BLL_set_prefix, _NodeReference_t);

typedef struct{
	fan::vector_t nodes;
	CONCAT2(BLL_set_prefix, _NodeReference_t) src;
	CONCAT2(BLL_set_prefix, _NodeReference_t) dst;
	struct{
		CONCAT2(BLL_set_prefix, _NodeReference_t) c;
		CONCAT2(BLL_set_prefix, _NodeReference_t) p;
	}e;
}CONCAT2(BLL_set_prefix, _t);
