#ifndef MUTEX_HPP_DEFINED
#define MUTEX_HPP_DEFINED

#include <mutex>

template <typename T>
class MutexGuard
{
public:

    MutexGuard() = default;
    MutexGuard( const T& value )
        : Mutex(),
          Value( value )
    {}
    ~MutexGuard() = default;

    T Get() const
    {
        std::lock_guard<std::mutex> lock(Mutex);
        T value = Value;
        return value;
    }

    void Set( const T& v )
    {
        std::lock_guard<std::mutex> lock(Mutex);
        Value = v;
    }

    mutable std::mutex Mutex;
    T Value;
};

#endif      // MUTEX_HPP_DEFINED