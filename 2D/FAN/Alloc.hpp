#pragma once
#include <iostream>

template <typename _Type>
class Alloc {
public:
	Alloc() : _First(0), _Last(0), _Size(0), _Data(NULL) {}
	Alloc(_Type _Data) {
		this->_Data = _Data;
	}
	/*Alloc(size_t _Reserve) : _First(0), _Last(_Reserve - 1) {
		resize(_Reserve);
	}*/
	~Alloc() {
		if (_Data != NULL) {
			//delete[] _Data;
		}
	}
	constexpr _Type& operator[](size_t _Where) {
		return _Data[_Where];
	}
	constexpr _Type operator[](size_t _Where) const {
		return _Data[_Where];
	}
	void push_back(_Type _Value) {
		if (this->empty()) {
			_Data = new _Type[++_Size];
		}
		else {
			_Type* _Temp = new _Type[_Size];
			this->copy(_Data, _Temp, _Last);
			_Data = new _Type[++_Size];
			this->copy(_Temp, _Data, _Last);
			delete[] _Temp;
		}
		_Data[_Last] = _Value;
		_Last = _Size;
	}

	constexpr void copy(const _Type* _Src, _Type* _Dest, size_t _Buffer) {
		int _Index = 0;
		for (; _Index != _Buffer; ++_Index) {
			_Dest[_Index] = _Src[_Index];
		}
	}
	constexpr size_t size() const {
		return this->_Size;
	}
	constexpr bool empty() const {
		return !this->_Size;
	}
	constexpr size_t begin() const {
		return this->_First;
	}
	constexpr size_t end() const {
		return this->_Last;
	}
	constexpr _Type* data() const {
		return _Data;
	}
	constexpr void resize(size_t _Reserve) {
		_Data = new _Type[_Reserve];
		_Size = _Reserve;
	}
private:
	size_t _First;
	size_t _Last;
	size_t _Size;
	_Type* _Data;
};