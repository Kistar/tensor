#ifndef TENSOR_HPP_
#define TENSOR_HPP_
#include <assert.h>
#include <string.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

namespace Tensor {

constexpr size_t None = std::numeric_limits<size_t>::max();

/* ************************************************************************************
    Build a Tensor of basic types with arbitrary dimensions.
    Functions that got realized:
    1. indexing an element of the Tensor (it is also a Tensor)
    2. concatenate two tensors on certain dimension, regarding that the length of the other dimensions of these two
   Tensors are the same
    3. duplicating a tensor with copy()
    4. multipling a tensor with repeat()
    5. get a value (not a tensor) in tensor with at()
    6. insert / remove a dimension from Tensor with squeeze / unsqueeze
    7. build special Tensor with arange / zeros
    8. print the info or/and data of the Tensor with info(), operator<< and print()
    9. outputs the data in a tensor into vector with append_to()
   ************************************************************************************ */
template <typename T>
class Tensor;

template <typename T>
Tensor<T> cat(const Tensor<T> &t1, const Tensor<T> &t2, size_t i);

template <typename T = double>
class Tensor {
    friend Tensor cat<T>(const Tensor &t1, const Tensor &t2, size_t i);

private:
    // size_: sizes of a single dimension
    vector<size_t> size_;
    // accumulated_size_: example: a (3, 5, 7)-Tensor has accumulated_size_ (105, 35, 7)
    vector<size_t> accumulated_size_;
    // data_:memory management for original data,
    std::shared_ptr<T[]> data_;
    // total_size_: total size of Tensor, it equals the product of elements in size_, and the first element of
    // accumulated_size_
    size_t total_size_;
    // offset from start of memory tile;
    size_t offset_;

public:
    // special Tensor that starts with integer start and ends before integer end
    // this function can reshape the Tensor with size
    static Tensor<T> arange(int start, int end, vector<size_t> size) {
        assert(end > start);
        size_t total_size = 1;
            for (auto s : size) {
                total_size *= s;
            }
        assert(total_size = end - start);
        std::shared_ptr<T[]> _data{new T[end - start]{}};
        auto                 number = start;
            for (auto ptr = _data.get(); number < end; number++, ptr++) {
                *ptr = number;
            }

        return Tensor<T>(size, _data);
    }

    // special 1d Tensor that starts with integer start and ends before integer end
    static Tensor<T> arange(int start, int end) {
        assert(end > start);
        std::shared_ptr<T[]> _data{new T[end - start]{}};
        auto                 number = start;
            for (auto ptr = _data.get(); number < end; number++, ptr++) {
                *ptr = number;
            }

        return Tensor<T>(vector<size_t>{static_cast<size_t>(end - start)}, _data);
    }

    Tensor<T>() = default;

    // build a Tensor with special shape and without initial value. The initial values are given as default
    explicit Tensor<T>(vector<size_t> size) : size_(size), offset_(0) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
            for (size_t i = size_.size(); i > 0; i--) {
                total_size_ *= size_[i - 1];
                accumulated_size_[i - 1] = total_size_;
            }
            if (total_size_ == 0) {
                return;
                // throw std::range_error("total size of Tensor should not be 0, please check!");
            }
        data_ = std::shared_ptr<T[]>{new T[total_size_]{}};
    }

    Tensor<T>(vector<size_t> size, vector<T> v) : size_(size), offset_(0) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
            for (size_t i = size_.size(); i > 0; i--) {
                total_size_ *= size_[i - 1];
                accumulated_size_[i - 1] = total_size_;
            }
        assert(v.size() == total_size_);
            if (v.size() == 0) {
                return;
                // throw std::range_error("total size of Tensor should not be 0, please check!");
            }
        data_ = std::shared_ptr<T[]>{new T[total_size_]{}};
        memcpy(data_.get(), &(v[0]), total_size_ * sizeof(T));
    }

    Tensor<T>(vector<size_t> size, std::shared_ptr<T[]> _data, size_t _offset = 0)
        : size_(size), data_(_data), offset_(_offset) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
            for (size_t i = size_.size(); i > 0; i--) {
                total_size_ *= size_[i - 1];
                accumulated_size_[i - 1] = total_size_;
            }
            if (total_size_ == 0) {
                assert(data_ == nullptr);
                // throw std::range_error("total size of Tensor should not be 0, please check!");
            }
    }

    Tensor<T>(Tensor<T> &&t)
        : size_(t.size_), accumulated_size_(t.accumulated_size_), total_size_(t.total_size_), offset_(t.offset_) {
        data_ = t.data_;
    }
    Tensor<T> &operator=(Tensor<T> &&t) {
        size_             = t.size_;
        accumulated_size_ = t.accumulated_size_;
        offset_           = t.offset_;
        total_size_       = t.total_size_;
        data_             = t.data_;
        return *this;
    }

    // output: the pointer of the begin of the data, no matter who holds the data
    // the data on the pointer can be edited
    T *data() {
        return data_.get() + offset_;
    }

    // output: the pointer of the begin of the data, no matter who holds the data
    // the data on the pointer is constant
    const T *data() const {
        return data_.get() + offset_;
    }

    // copy always returns a Tensor with its own memory
    Tensor<T> copy() const {
        std::shared_ptr<T[]> _data{new T[total_size_]{}};
        memcpy(_data.get(), data(), total_size_);
        return Tensor<T>(size_, _data);
    }

    // destructor, it also destructed its children and erase the pointer to itself in its parent
    // as long as it has parent and/or children
    ~Tensor<T>() = default;

    // return a copy of its shape(size)
    inline vector<size_t> size() const {
        return size_;
    }
    // return a copy of its total size
    inline size_t total_size() const {
        return total_size_;
    }

    // get a slice of original tensor, the data pointer still points to the original data
    Tensor<T> operator()(vector<size_t> index) {
        assert(index.size() <= size_.size());
        vector<size_t> new_size;
        new_size.insert(new_size.end(), size_.begin() + index.size(), size_.end());
            if (new_size.size() == 0) {
                new_size.push_back(1);
            }
        size_t move = 0;
            for (size_t i = 0; i < index.size(); i++) {
                assert(index[i] < size_[i]);
                move += index[i] * accumulated_size_[i + 1];
            }
        return Tensor<T>(new_size, data_, move);
    }

    // output: get a reference of a value in the tensor
    T &at(vector<size_t> position) {
        assert(position.size() == size_.size());
        size_t pos = 0;
        size_t i   = 0;
            for (; i < position.size() - 1; i++) {
                assert(position[i] < size_[i]);
                pos += position[i] * accumulated_size_[i + 1];
            }
        assert(position[i] < size_[i]);
        pos += position[i];

        return *(data() + pos);
    }

    // repeat the tensor several times, it works directly on original tensor
    Tensor<T> &repeat(vector<size_t> times) {
        assert(times.size() == size_.size());
        size_t total_times = 1;
            for (size_t i = 0; i < times.size(); i++) {
                total_times *= times[i];
            }
            if (total_times == 0 || data_ == nullptr) {
                total_size_     = 0;
                data_           = nullptr;
                size_t tmp_size = 1;
                    for (size_t i = size_.size(); i > 0; i--) {
                        size_[i - 1] *= times[i - 1];
                        tmp_size *= size_[i - 1];
                        accumulated_size_[i - 1] = tmp_size;
                    }
                return *this;
            }
        std::shared_ptr<T[]> _data(new T[total_size_ * total_times]{});

        vector<size_t> index(size_.size() - 1);
        vector<size_t> new_size(size_.size(), 1);
        vector<size_t> new_accumulated_size(size_.size(), 1);
        size_t         new_total_size = 1;
            for (size_t j = new_size.size(); j > 0; j--) {
                new_size[j - 1] = size_[j - 1] * times[j - 1];
                new_total_size *= new_size[j - 1];
                new_accumulated_size[j - 1] = new_total_size;
            }

        int    i                 = index.size() - 1;
        T     *new_data_position = _data.get();
        T     *data_position     = data_.get();
        size_t move              = 1;
        size_t new_move          = 1;
            while (i >= 0) {
                    if (index[i] > size_[i] - 1) {
                        new_data_position = _data.get();
                            for (int j = 0; j < i; j++) {
                                new_data_position += index[j] * new_accumulated_size[j + 1];
                            }
                            for (size_t k = 1; k < times[i]; k++) {
                                memcpy(new_data_position + k * size_[i] * new_accumulated_size[i + 1],
                                                new_data_position, size_[i] * new_accumulated_size[i + 1] * sizeof(T));
                            }
                        new_data_position += new_accumulated_size[i];

                        index[i] = 0;
                        i--;
                            if (i < 0) {
                                break;
                            }
                        index[i]++;
                        continue;
                    }
                    else if (i < index.size() - 1) {
                        data_position = data_.get();
                            for (int j = 0; j <= i; j++) {
                                data_position += index[j] * accumulated_size_[j + 1];
                            }

                        // new_data_position += new_accumulated_size[i];
                        i = index.size() - 1;
                    }
                    for (int k = 0; k < times[i + 1]; k++) {
                        memcpy(new_data_position + k * accumulated_size_[i + 1] +
                                            times[i + 1] * index[i] * accumulated_size_[i + 1],
                                        data_position + index[i] * accumulated_size_[i + 1],
                                        accumulated_size_[i + 1] * sizeof(T));
                    }
                index[i]++;
            }

        data_             = _data;
        size_             = new_size;
        accumulated_size_ = new_accumulated_size;
        total_size_       = new_total_size;
        return *this;
    }

    // insert a dimension into original tensor
    Tensor<T> &unsqueeze(size_t i) {
        assert(i <= size_.size());
        size_.insert(size_.begin() + i, 1);
        size_t tmp = 1;
            if (i < accumulated_size_.size()) {
                tmp = accumulated_size_[i];
            }
        accumulated_size_.insert(accumulated_size_.begin() + i, tmp);
        return *this;
    }

    // erase a dimension (whose size should be 1) of original tensor
    Tensor<T> &squeeze(size_t i) {
        assert(i <= size_.size());
        assert(size_[i] == 1);
        size_.erase(size_.begin() + i);
        accumulated_size_.erase(accumulated_size_.begin() + i);
        return *this;
    }

    // build a Tensor from a 1d vector
    Tensor<T> &operator=(const vector<T> &t) {
        assert(t.size() == total_size_);
        memcpy(data(), &(t[0]), total_size_ * sizeof(T));
        return *this;
    }

    // build a 2d-Tensor from a 2d vector
    Tensor<T> &operator=(const vector<vector<T>> &t) {
        assert(t.size() * t[0].size() == total_size_);
        T *ptr = data();
            for (auto d : t) {
                assert(d.size() == t[0].size());
                memcpy(ptr, &(t[0]), t[0].size() * sizeof(T));
                ptr += t[0].size();
            }
        return *this;
    }

    // print the values of the tensor, every column of the tensor is bracketed with "[]"
    // and two elements are divided by ", "
    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
            if (t.total_size_ == 0) {
                    for (size_t i = 0; i < t.size_.size(); i++) {
                        os << "[";
                    }
                    for (size_t i = 0; i < t.size_.size(); i++) {
                        os << "]";
                    }
                return os;
            }
            for (size_t i = 0; i < t.size_.size(); i++) {
                os << "[";
            }
            for (size_t i = 0; i < t.total_size_; i++) {
                os << (t.data())[i];
                    for (size_t j = t.accumulated_size_.size(); j > 0; j--) {
                            if ((i + 1) % t.accumulated_size_[j - 1] == 0) {
                                os << "]\n";
                                continue;
                            }
                            else {
                                os << ",";
                                    while (j < t.accumulated_size_.size()) {
                                        os << "[";
                                        j++;
                                    }
                                break;
                            }
                    }
            }
        return os;
    }

    // print the shape infos of the tensor
    void info() {
        cout << "size: [";
            for (size_t i = 0; i < size_.size() - 1; i++) {
                cout << size_[i] << ", ";
            }
        cout << size_.back() << "]\n";
        cout << "accumulated size: [";
            for (size_t i = 0; i < accumulated_size_.size() - 1; i++) {
                cout << accumulated_size_[i] << ", ";
            }
        cout << accumulated_size_.back() << "]\n";
        cout << "total size: " << total_size_ << endl;
    }

    // print the shape info and the data values of the tensor
    void print() {
        info();
        cout << *this << endl;
    }

    // slice several incontinuous pieces of the tensor and form a new tensor.
    // the new tensor has its own memory and any changes on the new tensor would not influence the original one.
    Tensor<T> slice(vector<std::pair<size_t, size_t>> range) const {
        assert(range.size() == size_.size());
        vector<size_t> size(range.size());
        auto           accumulated_size = size;
        size_t         total_size       = 1;
        size_t         size_to_copy     = 1;
            for (size_t i = range.size(); i > 0; --i) {
                auto &p = range[i - 1];
                    if (p.first == None) {
                        p.first = 0;
                    }
                    if (p.second == None) {
                        p.second = size_[i - 1];
                    }
                assert(p.second >= p.first);
                assert(p.second <= size_[i - 1]);
                size[i - 1] = p.second - p.first;
                total_size *= size[i - 1];
                accumulated_size[i - 1] = total_size;
            }
            for (size_t i = range.size(); i > 0; --i) {
                    if (size[i - 1] == size_[i - 1]) {
                        range.pop_back();
                        size_to_copy = accumulated_size[i - 1];
                    }
                    else {
                        break;
                    }
            }

            if (range.size() == 0) {
                return copy();
            }

        size_to_copy *= size[range.size() - 1];
        std::shared_ptr<T[]> _data{new T[total_size]{}};
        vector<size_t>       index(range.size() - 1, 0);
        T                   *ptr = _data.get();
            while (ptr - _data.get() < total_size) {
                size_t move = range[index.size()].first * accumulated_size_[index.size() + 1];
                    for (size_t i = 0; i < index.size(); i++) {
                        move += accumulated_size_[i + 1] * (range[i].first + index[i]);
                    }
                memcpy(ptr, data() + move, size_to_copy * sizeof(T));
                ptr += size_to_copy;
                size_t i = index.size();
                    while (i > 0) {
                        index[i - 1]++;
                            if (index[i - 1] == size[i - 1]) {
                                index[i - 1] = 0;
                                i--;
                            }
                            else {
                                break;
                            }
                    }
                    if (i == 0) {
                        break;
                    }
            }
        return Tensor<T>(size, _data);
    }

    // put a given value into a continuous memory, in this function, the Tensor is considered as a 1-d vector
    inline void index_put(size_t begin, size_t end, T val) {
        assert(end > begin);
        assert(total_size_ >= end);
        std::fill(data() + begin, data() + end, val);
    }

    // outstreams the data of Tensor onto the end of a given vector
    inline void append_to(vector<T> &target) {
            if (!data()) {
                return;
            }
        target.insert(target.end(), data(), data() + total_size_);
    }
};

// concatenate two Tensors into one on a given dimension, except which sizes of the other dimensions should be the same
// the output is a new Tensor, who has independent memory and changes on it would not influence the original Tensors
template <typename T>
Tensor<T> cat(const Tensor<T> &t1, const Tensor<T> &t2, size_t i) {
    assert(i < t1.size_.size());
    assert(t1.size_.size() == t2.size_.size());

        for (size_t j = 0; j < t1.size_.size(); j++) {
                if (j == i) {
                    continue;
                }
            assert(t1.size_[j] == t2.size_[j]);
        }

    vector<size_t> size = t1.size_;
    size[i] += t2.size_[i];

        if (!t1.data()) {
            Tensor<T> output = t2.copy();
            output.size_     = size;
            return std::move(output);
        }
        else if (!t2.data()) {
            Tensor<T> output = t1.copy();
            output.size_     = size;
            return std::move(output);
        }

    std::shared_ptr<T[]> _data{new T[t1.total_size_ + t2.total_size_]{}};

    size_t package_size_1;
    size_t package_size_2;
        if (i > 0) {
            package_size_1 = t1.accumulated_size_[i];
            package_size_2 = t2.accumulated_size_[i];
        }
        else {
            package_size_1 = t1.total_size_;
            package_size_2 = t2.total_size_;
        }

    size_t process_1 = 0;
    size_t process_2 = 0;
    T     *position  = _data.get();

        while (process_1 < t1.total_size_) {
            memcpy(position, t1.data() + process_1, package_size_1 * sizeof(T));
            process_1 += package_size_1;
            position += package_size_1;

            memcpy(position, t2.data() + process_2, package_size_2 * sizeof(T));

            process_2 += package_size_2;
            position += package_size_2;
        }

    return Tensor<T>(size, _data);
}

// output: zero Tensor with given shape
inline Tensor<double> zeros(vector<size_t> size) {
    return Tensor<double>(size);
}
}  // namespace Tensor
#endif