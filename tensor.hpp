#ifndef TENSOR_HPP_
#define TENSOR_HPP_
#include <assert.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

namespace Tensor {

constexpr size_t None = std::numeric_limits<size_t>::max();

template <typename T>
class Tensor;

template <typename T>
Tensor<T> cat(const Tensor<T> &t1, const Tensor<T> &t2, size_t i);

template <typename T = double>
class Tensor {
    friend Tensor cat<T>(const Tensor &t1, const Tensor &t2, size_t i);

private:
    vector<size_t> size_;
    vector<size_t> accumulated_size_;
    std::unique_ptr<T[]> data_;
    T *source_;
    size_t total_size_;
    bool is_origin_;
    Tensor<T> *parent_;
    std::unordered_set<Tensor<T> *> children_;

public:
    static Tensor<T> arange(int start, int end, vector<size_t> size) {
        assert(end >= start);
        size_t total_size = 1;
        for (auto s : size) {
            total_size *= s;
        }
        assert(total_size = end - start);
        std::unique_ptr<T[]> _data{new T[end - start]{}};
        auto number = start;
        for (auto ptr = _data.get(); number < end; number++, ptr++) {
            *ptr = number;
        }

        return Tensor<T>(size, std::move(_data));
    }

    static Tensor<T> arange(int start, int end) {
        assert(end >= start);
        std::unique_ptr<T[]> _data{new T[end - start]{}};
        auto number = start;
        for (auto ptr = _data.get(); number < end; number++, ptr++) {
            *ptr = number;
        }

        return Tensor<T>(vector<size_t>{static_cast<size_t>(end - start)}, std::move(_data));
    }

    explicit Tensor<T>(vector<size_t> size) : size_(size), is_origin_(true), source_(nullptr) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
        for (size_t i = size_.size(); i > 0; i--) {
            total_size_ *= size_[i - 1];
            accumulated_size_[i - 1] = total_size_;
        }
        data_ = std::unique_ptr<T[]>{new T[total_size_]{}};
    }

    size_t get_total_size() const { return total_size_; }

    Tensor<T>(vector<size_t> size, vector<T> v) : size_(size), is_origin_(true), source_(nullptr) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
        for (size_t i = size_.size(); i > 0; i--) {
            total_size_ *= size_[i - 1];
            accumulated_size_[i - 1] = total_size_;
        }
        assert(v.size() == total_size_);
        parent_ = nullptr;
        data_ = std::unique_ptr<T[]>{new T[total_size_]{}};
        memcpy(data_.get(), &(v[0]), total_size_ * sizeof(T));
    }

    Tensor<T>(vector<size_t> size, std::unique_ptr<T[]> _data)
        : size_(size), is_origin_(true), data_(std::move(_data)), source_(nullptr), parent_(nullptr) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
        for (size_t i = size_.size(); i > 0; i--) {
            total_size_ *= size_[i - 1];
            accumulated_size_[i - 1] = total_size_;
        }
    }

    Tensor<T>(vector<size_t> size, T *source, Tensor<T> *parent = nullptr)
        : size_(size), is_origin_(false), data_(nullptr), source_(source), parent_(parent) {
        total_size_ = 1;
        accumulated_size_.resize(size_.size());
        for (size_t i = size_.size(); i > 0; i--) {
            total_size_ *= size_[i - 1];
            accumulated_size_[i - 1] = total_size_;
        }
        if (parent != nullptr) {
            parent->children_.insert(this);
        }
    }

    Tensor<T>(Tensor<T> &&t)
        : size_(t.size_),
          accumulated_size_(t.accumulated_size_),
          source_(t.source_),
          is_origin_(t.is_origin_),
          parent_(t.parent_) {
        data_ = std::move(t.data_);
        if (parent_ != nullptr) {
            parent_->children_.insert(this);
        }
    }
    Tensor<T> &operator=(Tensor<T> &&t) {
        size_ = t.size_;
        accumulated_size_ = t.accumulated_size_;
        source_ = t.source_;
        is_origin_ = t.is_origin_;
        parent_ = t.parent_;
        data_ = std::move(t.data_);
        if (parent_ != nullptr) {
            parent_->children_.insert(this);
        }
    }

    // data() returns the position of the data, no matter who holds the data
    T *data() { return (source_ == nullptr) ? data_.get() : source_; }

    const T *data() const { return (source_ == nullptr) ? data_.get() : source_; }

    // copy always returns a Tensor with its own memory
    Tensor<T> copy() const {
        std::unique_ptr<T[]> _data{new T[total_size_]{}};
        memcpy(_data.get(), data(), total_size_);
        return Tensor<T>(size_, std::move(_data));
    }

    ~Tensor<T>() {
        if (is_origin_) {
            for (auto tensor_ptr : children_) {
                tensor_ptr->~Tensor<T>();
            }
        } else {
            for (auto tensor_ptr : children_) {
                tensor_ptr->~Tensor<T>();
            }
            if (parent_ != nullptr) {
                parent_->children_.erase(parent_->children_.find(this));
            }
        }
    }

    inline vector<size_t> size() { return size_; }
    inline size_t total_size() { return total_size_; }

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
        T *source = data() + move;
        return Tensor<T>(new_size, source, this);
    }

    T &at(vector<size_t> position) {
        assert(position.size() == size_.size());
        size_t pos = 0;
        for (size_t i = 0; i < position.size() - 1; i++) {
            assert(position[i] < size_[i]);
            pos += position[i] * accumulated_size_[i + 1];
        }
        assert(position.back() < size_.back());
        pos += position.back();
        return *(data() + pos);
    }

    Tensor<T> &repeat(vector<size_t> times) {
        assert(times.size() == size_.size());
        assert(is_origin_);
        for (auto child : children_) {
            child->~Tensor<T>();
        }
        size_t total_times = 1;
        for (size_t i = 0; i < times.size(); i++) {
            total_times *= times[i];
        }
        std::unique_ptr<T[]> _data(new T[total_size_ * total_times]{});

        vector<size_t> index(size_.size());
        vector<size_t> new_size(size_.size(), 1);
        vector<size_t> new_accumulated_size(size_.size(), 1);
        size_t new_total_size = 1;
        for (size_t j = new_size.size(); j > 0; j--) {
            new_size[j - 1] = size_[j - 1] * times[j - 1];
            new_total_size *= new_size[j - 1];
            new_accumulated_size[j - 1] = new_total_size;
        }

        int i = index.size() - 1;
        T *new_data_position = _data.get();
        T *data_position = data_.get();
        size_t move = 1;
        size_t new_move = 1;
        while (i >= 0) {
            if (index[i] > size_[i] - 1) {
                // cout << "index[" << i << "]: " << index[i] << ", size_[" << i << "]: " << size_[i] << endl;
                index[i] = 0;
                i--;
                if (i < 0) {
                    break;
                }
                for (size_t k = 1; k < times[i]; k++) {
                    // cout << "new_accumulated_size[" << i + 1 << "]:" << new_accumulated_size[i + 1] << endl;
                    memcpy(new_data_position + k * new_accumulated_size[i + 1], new_data_position,
                           new_accumulated_size[i + 1] * sizeof(T));
                    // for (size_t l = 0; l < new_accumulated_size[i + 1]; l++) {
                    //     cout << "original data: " << *(new_data_position + l);
                    //     cout << ", data copied: " << *(new_data_position + k * new_accumulated_size[i + 1] + l) <<
                    //     endl;
                    // }
                }
                index[i]++;
                continue;
            } else if (i < index.size() - 1) {
                data_position += accumulated_size_[i + 1];
                new_data_position += times[i] * new_accumulated_size[i + 1];
                i = index.size() - 1;
            }
            for (int k = 0; k < times[i]; k++) {
                // cout << "accumulated_size_[" << i << "]:" << accumulated_size_[i] << endl;
                memcpy(new_data_position + k * accumulated_size_[i] + times[i] * index[i], data_position + index[i],
                       accumulated_size_[i] * sizeof(T));
                // for (size_t l = 0; l < accumulated_size_[i]; l++) {
                //     cout << "original data: " << *(data_position + l);
                //     cout << ", data copied: " << *(new_data_position + k * accumulated_size_[i] + l) << endl;
                // }
            }
            index[i]++;
            // cout << "index[" << i << "]: " << index[i] << ", size_[" << i << "]: " << size_[i] << ", times[" << i
            //      << "]: " << times[i] << endl;
        }
        data_ = std::move(_data);
        size_ = new_size;
        accumulated_size_ = new_accumulated_size;
        total_size_ = new_total_size;
        return *this;
    }

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

    Tensor<T> &squeeze(size_t i) {
        assert(i <= size_.size());
        assert(size_[i] == 1);
        size_.erase(size_.begin() + i);
        accumulated_size_.erase(accumulated_size_.begin() + i);
        return *this;
    }

    Tensor<T> &operator=(const vector<T> &t) {
        assert(t.size() == total_size_);
        memcpy(data(), &(t[0]), total_size_ * sizeof(T));
        return *this;
    }

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

    friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &t) {
        for (size_t i = 0; i < t.size_.size(); i++) {
            os << "[";
        }
        for (size_t i = 0; i < t.total_size_; i++) {
            os << (t.data())[i];
            for (size_t j = t.accumulated_size_.size(); j > 0; j--) {
                if ((i + 1) % t.accumulated_size_[j - 1] == 0) {
                    os << "]";
                    continue;
                } else {
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

    void print() {
        info();
        cout << *this << endl;
    }

    Tensor<T> slice(vector<std::pair<size_t, size_t>> range) const {
        assert(range.size() == size_.size());
        vector<size_t> size(range.size());
        auto accumulated_size = size;
        size_t total_size = 1;
        size_t size_to_copy = 1;
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
            } else {
                break;
            }
        }

        if (range.size() == 0) {
            return copy();
        }

        size_to_copy *= size[range.size() - 1];
        std::unique_ptr<T[]> _data{new T[total_size]{}};
        vector<size_t> index(range.size() - 1, 0);
        T *ptr = _data.get();
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
                } else {
                    break;
                }
            }
            if (i == 0) {
                break;
            }
        }
        return Tensor<T>(size, std::move(_data));
    }
    inline void index_put(size_t begin, size_t end, T val) {
        assert(end > begin);
        assert(total_size_ >= end);
        std::fill(data() + begin, data() + end, val);
        // new (data() + begin) T[end - begin]{val};
        // memset(data() + begin, val, (end - begin) * sizeof(T));
    }
    inline void append_to(vector<T> &target) { target.insert(target.end(), data(), data() + total_size_); }
};
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
    std::unique_ptr<T[]> _data{new T[t1.total_size_ + t2.total_size_]{}};
    size_t package_size_1;
    size_t package_size_2;
    if (i > 0) {
        package_size_1 = t1.accumulated_size_[i];
        package_size_2 = t2.accumulated_size_[i];
    } else {
        package_size_1 = t1.total_size_;
        package_size_2 = t2.total_size_;
    }
    size_t process_1 = 0;
    size_t process_2 = 0;
    T *position = _data.get();
    while (process_1 < t1.total_size_) {
        memcpy(position, t1.data() + process_1, package_size_1 * sizeof(T));
        process_1 += package_size_1;
        position += package_size_1;
        memcpy(position, t2.data() + process_2, package_size_2 * sizeof(T));
        process_2 += package_size_2;
        position += package_size_2;
    }
    return Tensor<T>(size, std::move(_data));
}
inline Tensor<double> zeros(vector<size_t> size) { return Tensor<double>(size); }
}  // namespace Tensor
#endif