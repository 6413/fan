#include "Alloc.hpp"

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

template < typename _Key, typename _Ty, typename _Cmp = std::less<_Key>, typename _Alloc = std::allocator< std::pair<const _Key, _Ty> > > class map
{
public:
    typedef map<_Key, _Ty, _Cmp, _Alloc> _Myt;
    typedef _Key key_type;
    typedef _Ty mapped_type;
    typedef _Cmp compare_type;
    typedef _Alloc allocator_type;
    typedef std::pair<const key_type, mapped_type> value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type* iterator;
    typedef const value_type* const_iterator;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    map()
        : size_(0), capacity_(20), data_(_Alloc().allocate(20))
    {
    }

    map(const _Myt& _Rhs)
        : size_(_Rhs.size_), capacity_(_Rhs.size_ + 20), data_(_Alloc().allocate(_Rhs.size_))
    {
        int count = 0;
        for (iterator i = &_Rhs.data_[0]; i != &_Rhs.data_[_Rhs.size_]; ++i, ++count)
        {
            _Alloc().construct(&data_[count], *i);
        }
    }

    bool empty() {
        return size_;
    }

    ~map()
    {
        if (!empty())
        {
            for (iterator i = begin(); i != end(); ++i)
            {
                _Alloc().destroy(i);
            }
            _Alloc().deallocate(data_, capacity_);
        }
    }

    _Myt& insert(const value_type& _Value)
    {
        if (++size_ >= capacity_)
        {
            reserve(capacity_ * 2);
        }
        _Alloc().construct(&data_[size_ - 1], _Value);
        return *this;
    }

    iterator begin() const
    {
        return &data_[0];
    }

    iterator end() const
    {
        return &data_[size_];
    }

    bool has_key(const key_type& _Key)
    {
        for (iterator i = begin(); i != end(); ++i)
        {
            if (i->first == _Key)
            {
                return true;
            }
        }
        return false;
    }

    mapped_type& operator[](const key_type& _Key)
    {
        if (has_key(_Key)) 
        {
            for (iterator i = begin(); i != end(); ++i)
            {
                if (i->first == _Key)
                {
                    return i->second;
                }
            }
        }
        
        size_type op = size_;
        insert(value_type(_Key, mapped_type()));
        return data_[op].second;
    }

    _Myt& reserve(size_type _Capacity)
    {
        int count = 0;
        if (_Capacity < capacity_)
        {
            return *this;
        }
        pointer buf = _Alloc().allocate(_Capacity);
        for (iterator i = begin(); i != end(); ++i, ++count)
        {
            _Alloc().construct(&buf[count], *i);
        }
        std::swap(data_, buf);
        for (iterator i = &buf[0]; i != &buf[size_]; ++i)
        {
            _Alloc().destroy(i);
        }
        _Alloc().deallocate(buf, capacity_);
        capacity_ = _Capacity;
    }
    private:
        pointer data_;
        size_type size_, capacity_;
};