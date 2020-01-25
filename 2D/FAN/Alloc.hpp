#pragma once
#include <iostream>
#include <memory>

#define ALLOC_BUFFER 0xfff

template <typename _Type>
class Alloc;

template <typename _Type>
class iterator {
public:
	iterator() : _Ptr(0) {}
	iterator(_Type* _Temp) : _Ptr(_Temp) {}
	//iterator(const Alloc<_Type>& _InIt) : _Ptr(_InIt._Data) {}
	constexpr bool operator!=(const iterator& _It) {
		return _It._Ptr != _Ptr;
	}
	constexpr iterator operator++() {
		++_Ptr;
		return *this;
	}
	constexpr iterator operator--() {
		--_Ptr;
		return *this;
	}
	constexpr iterator operator-(iterator<_Type> _Index) {
		_Ptr = (_Type*)(_Ptr - _Index._Ptr);
		return *this;
	}
	constexpr iterator operator-(size_t _Index) {
		_Ptr = (_Type*)(_Ptr - _Index);
		return *this;
	}
	constexpr _Type& operator*() const {
		return *_Ptr;
	}
	constexpr iterator operator+(_Type _Value) const {
		return _Ptr + _Value;
	}
protected:
	_Type* _Ptr;
};

template <typename _Type>
class Alloc : public iterator<_Type> {
public:
	Alloc(bool DBT = false) : _Size(0), _Data(NULL), _Current(DBT ? 1 : 0) {}
	template <typename type>
	Alloc(type _Reserve, bool DBT = false) : _Data(NULL), _Current(DBT ? 1 : 0) {
		resize(_Reserve);
	}
	~Alloc() {}
	_Type& operator[](size_t _Index) const {
		return _Data[_Index];
	}
	constexpr _Type operator[](const iterator<_Type>& _Index) const {
		return _Data + _Index;
	}
	constexpr _Type operator*() const {
		return _Data[0];
	}
	constexpr void insert(size_t _Index, _Type _Value) {
		//if (!_Size) {
		//	resize(1);
		//	id[_Index] = _Current;
		//	_Data[_Current] = _Value;
		//	_Current++;
		//	return;
		//}
		//else if (_Size <= _Current){
		//	std::unique_ptr<_Type[]> _Temp = std::make_unique<_Type[]>(_Size);
		//	copy(_Data, _Temp, _Size);
		//	_Size += ALLOC_BUFFER;
		//	_Data = std::make_unique<_Type[]>(_Size);
		//	copy(_Temp, _Data, _Size - ALLOC_BUFFER);
		//}
		/*if (!_Size) {
			resize(1);
			_Data[_Index] = _Value;
			return;
		}
		if (_Size - 1 <= _Index) {
			copy_resize(_Index + 1);
			for (size_t _I = _Size; _I > _Index; _I--) {
				_Data[_I] = _Data[_I - 1];
			}
		}
		_Data[_Index] = _Value;
		_Current++;*/
	}
	constexpr void insert(size_t _From, size_t _To, _Type _Value) {
		if (_Size <= _To) {
			copy_resize(_To);
		}
		for (int _I = _From; _I < _To; _I++) {
			_Data[_I] = _Value;
		}
	}
	//constexpr void free_to_max() {
	//	if (_Size > _Current) {
	//		_Type* _Temp = new _Type[_Current];
	//		copy(_Data, _Temp, _Current);
	//		delete[] _Data;
	//		_Size = _Current;
	//		_Data = new _Type[_Current];
	//		copy(_Temp, _Data, _Current);
	//		delete[] _Temp;
	//	}
	//}
	void push_back(_Type _Value) {
		if (_Size > _Current) {
			_Data[_Current] = _Value;
			_Current++;
			return;
		}
		if (!_Size) {
			_Data = std::make_unique<_Type[]>(++_Size);
			_Data[_Size - 1] = _Value;
			_Current++;
			return;
		}
		if (_Size <= _Current) {
			std::unique_ptr<_Type[]> _Temp = std::make_unique<_Type[]>(_Size);
			copy(_Data, _Temp, _Size);
			_Size += ALLOC_BUFFER;
			_Data = std::make_unique<_Type[]>(_Size);
			copy(_Temp, _Data, _Size - ALLOC_BUFFER);
		}
		_Data[_Current] = _Value;
		_Current++;
	}
	template <typename _Alloc>
	constexpr void copy(const _Alloc& _Src, _Alloc& _Dest, const size_t _Buffer) {
		int _Index = 0;
		for (; _Index != _Buffer; ++_Index) {
			_Dest[_Index] = _Src[_Index];
		}
	}

	constexpr size_t size() const {
		return this->_Size;
	}
	constexpr size_t current() const {
		return this->_Current;
	}
	constexpr bool empty() const {
		return !this->_Size;
	}
	constexpr _Type* data() const {
		return _Data.get();
	}
	constexpr void resize(size_t _Reserve) {
		_Data = std::make_unique<_Type[]>(_Reserve);
		_Size = _Reserve;
	}
	constexpr void copy_resize(size_t _Reserve) {
		std::unique_ptr<_Type[]> _Temp = std::make_unique<_Type[]>(_Size);
		copy(_Data, _Temp, _Size);
		this->resize(_Reserve);
		copy(_Temp, _Data, _Reserve);
	}
	constexpr iterator<_Type> begin() const {
		return _Data.get();
	}
	constexpr iterator<_Type> end() const {
		return _Data.get() + _Current;
	}
	constexpr void erase(size_t _Index) {
		for (size_t _I = _Index; _I < current() - 1; _I++) {
			_Data[_I] = _Data[_I + 1];
		}
		copy_resize(current() - 1);
		_Current--;
	}
	constexpr void erase(iterator<_Type> _Index) {
		for (auto _I = _Index; _I != end() - 1; ++_I) {
			*_I = *(_I + 1);
		}
		copy_resize(current() - 1);
		_Current--;
	}
	constexpr void erase_all() {
		_Data.reset();
	}
private:
	std::unique_ptr<_Type[]> _Data;
	size_t _Size;
	size_t _Current;
};