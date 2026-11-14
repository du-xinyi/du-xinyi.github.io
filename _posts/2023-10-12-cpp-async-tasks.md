---
title: C++ Asynchronous Tasks
description: 整理 C++ 中 std::async、std::future、std::promise 与 std::packaged_task 的用法，以及三种获取异步结果方式的区别
date: 2023-10-12 22:00:00 +0800
categories: [ Development, C++ ]
tags: [ C++, Multithreading ]
series: cpp
---

[C++ Multithreading](/posts/cpp-multithreading/)整理了线程的创建、互斥和同步。`std::thread`没有办法直接取回线程函数的返回值，`<future>`头文件提供了异步执行函数并获取结果的方式

## std::async

> 注：std::async 定义在 future 头文件中

`std::async`参数
不同于`std::thread`，`std::async`是一个函数，所以没有成员函数

| 重载版本 | 作用 |
| :--- | :--- |
| `template <class Fn, class… Args>`<br>`std::future<std::invoke_result_t<`<br>`std::decay_t<Fn>, std::decay_t<Args>…>>`<br>`async (Fn&& fn, Args&&… args)` | 异步或同步（由实现决定）<br>以`args`为参数执行`fn`<br>传递引用参数需要`std::ref`或`std::cref` |
| `template <class Fn, class… Args>`<br>`std::future<std::invoke_result_t<`<br>`std::decay_t<Fn>, std::decay_t<Args>…>>`<br>`async (launch policy, Fn&& fn, Args&&… args);` | 异步或同步（根据 policy 参数而定）<br>以`args`为参数执行`fn`<br>传递引用参数需要`std::ref`或`std::cref` |

`std::launch`强枚举类（enum class）  
`std::launch`有 2 个枚举值，不指定`policy`时相当于二者按位或（实际值由实现定义，下表为 libstdc++ 的值）

| 标识符                                                | 实际值   | 作用                                        |
| :---------------------------------------------------- | :------- | :------------------------------------------ |
| 枚举值`launch::async`                                 | `0x1(1)` | 异步启动                                    |
| 枚举值`launch::deferred`                              | `0x2(2)` | 调用`future::get`或`future::wait`时同步启动 |
| `launch::async`与`launch::deferred`按位或（默认策略） | `0x3(3)` | 同步或异步，由实现决定                      |

例一：

```c++
#include <iostream>
#include <thread>
#include <future>

int main()
{
    auto fut = std::async(
        std::launch::async, [](const char *message)
        { std::cout << message << std::flush; },
        "Hello, ");
    fut.wait(); // 等待异步任务完成
    std::cout << "World!" << std::endl;

    return 0;
}
```
{: file='std::async 的使用'}

输出结果：

```text
Hello, World!
```

## std::future

例二：

```c++
#include <iostream>
#include <future>

template <class... Args>
decltype(auto) sum(Args &&...args)
{
    return (0 + ... + args); // "0 +"避免空参数包错误
}

int main()
{
    std::future<int> val = std::async(std::launch::async, sum<int, int, int>, 1, 10, 100); // 必须带模板参数

    std::cout << val.get() << std::endl;

    return 0;
}
```
{: file='std::future 获取线程返回值'}

`std::future`常用成员函数

- 构造&析构函数

| 函数                             | 类型         | 作用 |
| :------------------------------- | :----------- | :--- |
| `future() noexcept`              | 默认构造函数 | 构造一个空的、无效的`future`对象<br>可以移动分配到另一个`future`对象 |
| `future(const future&) = delete` | 复制构造函数 | 删除`std::future`类型的拷贝构造函数 |
| `future (future&& x) noexcept`   | 移动构造函数 | 构造一个与`x`相同的对象并破坏`x` |
| `~future()`                      | 析构函数     | 析构对象 |

- 常用成员函数

| 函数 | 作用 |
| :--- | :--- |
| 一般：`T get()`<br>当类型为引用：`R& future<R&>::get()`<br>当类型为 void：`void future::get()` | 阻塞等待线程结束并获取返回值<br>若任务抛出异常则重新抛出<br>调用后`future`变为无效，只能调用一次 |
| `void wait() const` | 阻塞等待线程结束 |
| `template <class Rep, class Period>`<br>`future_status wait_for(const chrono::duration<Rep,Period>& rel_time) const;` | 阻塞等待`rel_time`<br>若在这段时间内线程结束则返回`future_status::ready`<br>若没结束则返回`future_status::timeout`<br>若`async`是以`launch::deferred`启动的<br>则不会阻塞并立即返回`future_status::deferred` |

`std::future_status`强枚举类  
见上文`std::future::wait_for`解释

例三：

```c++
#include <iostream>
#include <future>

void count_big_number()
{
    for (int i = 0; i <= 10'0000'0000; i++)
    {
        ;
    }
}

int main()
{
    std::future<void> fut = std::async(std::launch::async, count_big_number);
    std::cout << "Please wait" << std::flush;

    while (fut.wait_for(std::chrono::seconds(1)) != std::future_status::ready) // 每次等待1秒
    {
        std::cout << '.' << std::flush;
    }

    std::cout << std::endl
         << "Finished!" << std::endl;

    return 0;
}
```
{: file='void 特化 std::future'}

### std::promise

`std::thread`会忽略线程函数的返回值，不能像`std::async`那样直接得到`std::future`。一种做法是通过引用参数把结果带出来

例四：

```c++
#include <functional>
#include <iostream>
#include <thread>

constexpr long double PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628L;

// 求圆的直径、周长及面积
void get_circle_info(double r, double &d, double &c, double &s)
{
    d = r * 2;
    c = PI * d;
    s = PI * r * r;
}

int main()
{
    double r;
    std::cin >> r;
    double d, c, s;
    std::thread th(get_circle_info, r, std::ref(d), std::ref(c), std::ref(s));
    th.join(); // 必须等线程结束后才能读取结果
    std::cout << d << ' ' << c << ' ' << s << std::endl;

    return 0;
}
```
{: file='引用传递返回值'}

这种做法需要自己保证读取结果时线程已经结束，也无法传递异常。`std::promise`提供了更通用的方式

`std::promise`与`std::future`是同一共享状态的两端：`std::promise`是写入端，`std::future`是读取端  
`std::future`本身不能设置值，但可以通过`std::promise`设置值，再用`get_future()`得到对应的`std::future`

`std::promise`常用成员函数

- 构造&析构函数

| 函数 | 类型         | 作用                                                        |
| :--- | :----------- | :---------------------------------------------------------- |
| `promise()` | 默认构造函数 | 构造空的`std::promise`对象                                  |
| `template <class Alloc>`<br>`promise(allocator_arg_t aa, `<br>`const Alloc& alloc)` | 构造函数     | 与默认构造函数相同<br>但使用特定的内存分配器 alloc 构造对象 |
| `promise (const promise&) = delete` | 复制构造函数 | 删除`std::promise`类型的拷贝构造函数                        |
| `promise (promise&& x) noexcept` | 移动构造函数 | 构造一个与`x`相同的对象并破坏`x`                            |
| `~promise()` | 析构函数     | 析构对象                                                    |

- 常用成员函数

| 函数 | 作用 |
| :--- | :--- |
| 一般：<br>`void set_value (const T& val)`<br>`void set_value (T&& val)`<br>类型为引用：<br>`void promise<R&>::set_value (R& val)`<br>类型为 void：<br>`void promise::set_value (void)` | 设置`promise`的值并将共享状态设为`ready`<br>（将`future_status`设为`ready`）<br>`void`：特化：只将共享状态设为`ready` |
| `future get_future()` | 构造一个`future`对象，其值和`status`与 promise 相同 |

例五：
以例二中的代码为基础加以修改

```c++
#include <iostream>
#include <thread>
#include <future>

template <class... Args>
decltype(auto) sum(Args &&...args)
{
    return (0 + ... + args);
}

template <class... Args>
void sum_thread(std::promise<long long> &val, Args &&...args)
{
    val.set_value(sum(args...));
}

int main()
{
    std::promise<long long> sum_value;
    std::thread get_sum(sum_thread<int, int, int>, std::ref(sum_value), 1, 10, 100);
    std::cout << sum_value.get_future().get() << std::endl;
    get_sum.join();

    return 0;
}
```
{: file='std::promise 的使用'}

### std::packaged_task

`std::packaged_task`把一个可调用对象包装起来，调用它时会把返回值（或异常）存入共享状态，通过`get_future()`获取结果。与`std::async`相比，任务在哪个线程、什么时候执行由自己决定，适合用来实现线程池和任务队列

```c++
#include <future>
#include <iostream>
#include <thread>

int add(int a, int b)
{
    return a + b;
}

int main()
{
    std::packaged_task<int(int, int)> task(add);
    std::future<int> result = task.get_future();

    std::thread th(std::move(task), 1, 2); // packaged_task不可复制，只能移动

    std::cout << result.get() << std::endl; // 3
    th.join();

    return 0;
}
```
{: file='std::packaged_task 的使用'}

得到`std::future`的三种方式对比：

| 方式                 | 结果的设置            | 线程的管理                       |
| :------------------- | :-------------------- | :------------------------------- |
| `std::async`         | 函数返回值自动设置    | 由标准库决定                     |
| `std::packaged_task` | 函数返回值自动设置    | 自己决定在哪个线程执行           |
| `std::promise`       | 手动调用`set_value()` | 自己决定，可以在任意位置设置结果 |
