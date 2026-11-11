---
title: C++ Multithreading
description: 系统整理 C++ 多线程、同步原语、原子操作、异步任务与线程管理
date: 2023-10-12 21:30:00 +0800
categories: [ Development, C++ ]
tags: [ C++, Multithreading ]
---

在传统的 C++（C++11 之前）
中并没有引入线程这个概念，在 C++11 出来之前，如果想要在 C++ 中实现多线程，需要借助操作系统平台提供的 API，比如 Linux 的`<pthread.h>`
，或者 Windows 下的`<windows.h>`  
C++11 提供了语言层面上的多线程，包含在头文件`<thread>`
中。其解决了跨平台的问题，提供了管理线程、保护共享数据、线程间同步操作、原子操作等类。`C++11`引入了 5 个头文件来支持多线程编程

| 头文件                 | 作用 |
| :--------------------- | :--- |
| `<thread>`             | 提供多线程编程所需的类和函数，包括创建、启动、等待和管理线程 |
| `<mutex>`              | 提供互斥锁和其他同步原语的类和函数，用于保护共享资源，防止竞态条件 |
| `<atomic>`             | 提供原子操作库，用于执行线程安全的原子操作 |
| `<future>`             | 异步编程相关，用于异步执行函数和获取结果 |
| `<condition_variable>` | 提供条件变量类，用于线程之间的协调和通信 |

## std::thread

`std::thread`常用函数

- 构造&析构函数

| 函数 | 类别           | 作用                                           |
| :--- | :------------- | :--------------------------------------------- |
| `thread() noexcept` | 默认构造函数   | 创建一个不关联任何线程的空对象                 |
| `template <class Fn, class… Args>`<br>`explicit thread(Fn&& fn, Args&&… args)` | 初始化构造函数 | 创建一个线程<br>以`args`为参数<br>执行`fn`函数 |
| `thread(const thread&) = delete` | 复制构造函数   | 禁用`std::thread`类的拷贝构造函数              |
| `thread(thread&& x) noexcept` | 移动构造函数   | 构造一个与`x`相同的对象<br>会破坏`x`对象       |
| `~thread()` | 析构函数       | 析构对象                                       |

- 成员函数

| 函数                                   | 作用 |
| :------------------------------------- | :--- |
| `void join()`                          | 等待线程结束并清理资源（会阻塞） |
| `bool joinable()`                      | 返回线程是否可以执行`join`函数 |
| `void detach()`                        | 启动的线程自主在后台运行<br>仅当`joinable()`为`true`时可调用，且调用此函数会使其不能被`join` |
| `std::thread::id get_id()`             | 获取线程 id |
| `thread& operator=(std::thread &&rhs)` | 见移动构造函数<br>（如果对象是`joinable`，会调用`std::terminate()`结束程序） |

例一：

```c++
#include <iostream>
#include <thread>

void thread1() 
{ 
    std::cout << "Hello," << std::endl; 
}

void thread2() 
{ 
    std::cout << "World!" << std::endl; 
}

int main() 
{
    std::thread a(thread1);
    std::thread b(thread2);

    a.join();
    b.join();

    return 0;
}
```
{: file='基本使用'}

输出结果（两个线程间无同步，顺序不定，甚至可能交错）：

```text
Hello,
World!
```

或者

```text
World!
Hello,
```

例二：

```c++
#include <iostream>
#include <thread>

void countnumber(int id, unsigned int n)
{
    for (unsigned int i = 1; i <= n; i++)
    {
        ;
    }

    std::cout << "Thread " << id << " finished!" << std::endl;
}

int main()
{
    std::thread th[10];
    for (int i = 0; i < 10; i++)
    {
        th[i] = std::thread(countnumber, i, 100000000);
    }

    for (int i = 0; i < 10; i++)
    {
        th[i].join();
    }

    return 0;
}
```
{: file='有参数的函数'}

输出结果之一：

```text
Thread 4 finished!
Thread 8 finished!
Thread 6 finished!
Thread 3 finished!
Thread 1 finished!
Thread 7 finished!
Thread 9 finished!
Thread 5 finished!
Thread 2 finished!
Thread 0 finished!
```

例三：

```c++
#include <iostream>
#include <thread>

template <class T>
void changevalue(T &x, T val)
{
    x = val;
}

int main()
{
    std::thread th[100];
    int nums[100];
    for (int i = 0; i < 100; i++)
    {
        th[i] = std::thread(changevalue<int>, nums[i], i + 1);
    }

    for (int i = 0; i < 100; i++)
    {
        th[i].join();
        std::cout << nums[i] << std::endl;
    }

    return 0;
}
```
{: file='带引用参数的函数'}

正常编译这个程序，编译器一定会报错  
原因是`std::thread`会先把参数拷贝一份保存到新线程中，再把这份拷贝作为右值传给函数，而非`const`的左值引用`T &x`不能绑定到右值上

要让线程函数拿到原变量的引用，需要用`std::ref`或`std::cref`包装参数：

- `std::ref`传递可修改的引用
- `std::cref`传递`const`引用

修改代码为

```c++
#include <iostream>
#include <thread>

template <class T>
void changevalue(T &x, T val)
{
    x = val;
}

int main()
{
    std::thread th[100];
    int nums[100];
    for (int i = 0; i < 100; i++)
    {
        th[i] = std::thread(changevalue<int>, std::ref(nums[i]), i + 1);
    }

    for (int i = 0; i < 100; i++)
    {
        th[i].join();
        std::cout << nums[i] << std::endl;
    }

    return 0;
}
```
{: file='带引用参数的函数'}

输出结果：

```text
1
2
3
4
...
99
100
```

**注意事项：**

- 线程是在`thread`对象被定义的时候开始执行的，而不是在调用`join`函数时才执行的，调用`join`函数只是阻塞等待线程结束并回收资源
- 分离的线程（执行过`detach`的线程）会在自己结束时释放资源；但`main`返回、进程退出时，仍在运行的分离线程会被直接终止
- 线程会在函数运行完毕后自动释放，不推荐利用其他方法强制结束线程，可能会因资源未释放而导致内存泄漏
- **没有执行`join`或`detach`的`std::thread`对象在析构时（仍为`joinable`）会调用`std::terminate()`终止程序**

## std::this_thread

在`<thread>`头文件中，不仅有`std::thread`这个类，而且还有一个`std::this_thread`命名空间，其可以很方便地让线程对自己进行控制

`std::this_thread`常用函数

| 函数 | 作用                                     |
| :--- | :--------------------------------------- |
| `std::thread::id get_id() noexcept` | 获取当前线程 id                          |
| `template<class Rep, class Period>`<br>`void sleep_for( const std::chrono::duration<Rep,`<br>`Period>& sleep_duration )` | 等待`sleep_duration`                     |
| `void yield() noexcept` | 暂时放弃线程的执行，将主动权交给其他线程 |

例四：

```c++
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdint>

std::atomic_bool ready;

void sleep_ms(uintmax_t ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
void count()
{
    while (!ready)
        std::this_thread::yield();
    for (int i = 0; i <= 20'0000'0000; i++)
        ;
    std::cout << "Thread " << std::this_thread::get_id() << " finished!" << std::endl;

    return;
}
int main()
{
    ready = 0;
    std::thread th[10];
    for (int i = 0; i < 10; i++)
    {
        th[i] = std::thread(::count);
    }
    sleep_ms(5000);
    
    ready = true;
    std::cout << "Start!" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        th[i].join();
    }

    return 0;
}
```
{: file='std::this_thread 中常用函数的使用'}

输出结果：

```text
Start!
Thread 139621413664512 finished!
Thread 139621396879104 finished!
Thread 139621371700992 finished!
Thread 139621354915584 finished!
Thread 139621422057216 finished!
Thread 139621380093696 finished!
Thread 139621405271808 finished!
Thread 139621363308288 finished!
Thread 139621346522880 finished!
Thread 139621388486400 finished!
```

## std::mutex 和 std::atomic

### std::mutex

`std::mutex`是 C++ 中最基本的互斥量，一个线程将`mutex`锁住时，其它的线程就不能操作`mutex`，直到这个线程将`mutex`解锁

例五：

```c++
#include <iostream>
#include <thread>
#include <mutex>

int n = 0;
std::mutex mtx;

void count10000()
{
    for (int i = 1; i <= 10000; i++)
    {
        mtx.lock();
        n++;
        mtx.unlock();
    }
}

int main()
{
    std::thread th[100];
    for (std::thread &x : th)
    {
        x = std::thread(count10000);
    }

    for (std::thread &x : th)
    {
        x.join();
    }

    std::cout << n << std::endl;
    
    return 0;
}
```
{: file='std::mutex 的使用'}

输出结果为 1000000，这里如果不使用锁，实际输出结果比 1000000 小，并且输出结果不定

`n++`并不是一步完成的，而是读取、加一、写回三步。两个线程的这三步交替执行时，后写回的线程会覆盖掉另一个线程的结果，这种情况称为数据竞争

![两个线程同时执行 n++ 时丢失一次更新的过程](posts/2023-10-12-cpp-multithreading/data_race.png)

`std::mutex`互斥类型

| 类型                         | 说明              |
| :--------------------------- | :---------------- |
| `std::mutex`                 | 基本`Mutex`类     |
| `std::recursive_mutex`       | 递归`Mutex`类     |
| `std::timed_mutex`           | 定时`Mutex`类     |
| `std::recursive_timed_mutex` | 定时递归`Mutex`类 |

`std::mutex`常用成员函数（mutex 代指对象）

| 函数              | 作用 |
| :---------------- | :--- |
| `void lock()`     | 将`mutex`上锁<br>如果`mutex`已经被其它线程上锁，那么会阻塞，直到解锁；<br>如果`mutex`已经被同一个线程锁住，为未定义行为（通常表现为死锁），需要重复加锁应使用`std::recursive_mutex` |
| `void unlock()`   | 解锁`mutex`，释放其所有权<br>如果有线程因为调用`lock()`而被阻塞，其中一个线程会获得锁（不保证先后顺序）<br>如果当前线程未锁定`mutex`，为未定义行为 |
| `bool try_lock()` | 尝试将`mutex`上锁<br>如果`mutex`未被上锁，则将其上锁并返回`true`，如果`mutex`已被锁则返回`false` |

#### std::lock_guard

创建`std::lock_guard`对象时，它将尝试获取提供给它的互斥锁的所有权。当控制流离开`std::lock_guard`
对象的作用域时，`std::lock_guard`析构并释放互斥量

`std::lock_guard`特点：

- 创建即加锁，作用域结束自动析构并解锁，无需手工解锁
- 不能中途解锁，必须等作用域结束才解锁
- 不能复制

直接调用`lock()`和`unlock()`时，如果两者之间提前`return`或抛出异常，锁将永远不会被释放。使用`std::lock_guard`可以保证离开作用域时一定解锁，例五中的函数应写成

```c++
void count10000()
{
    for (int i = 1; i <= 10000; i++)
    {
        std::lock_guard<std::mutex> lock(mtx);
        n++;
    }
}
```
{: file='std::lock_guard 的使用'}

#### std::unique_lock

`std::unique_lock`具有`std::lock_guard`的所有功能，同时又具有其他很多方法，使用起来更加灵活方便，能够应对更复杂的锁定需要

`std::unique_lock`的特点：

- 创建时可以不锁定（通过指定第二个参数为`std::defer_lock`），而在需要时再锁定
- 可以随时加锁解锁
- 作用域规则同`std::lock_guard`，析构时自动释放锁
- 不可复制，可移动
- 条件变量需要该类型的锁作为参数（此时必须使用`std::unique_lock`）

```c++
std::unique_lock<std::mutex> lock(mtx, std::defer_lock); // 创建时不加锁

lock.lock();   // 需要时再加锁
n++;
lock.unlock(); // 可以提前解锁，缩短持有锁的时间

// 不需要锁的耗时操作
```
{: file='std::unique_lock 的使用'}

#### std::scoped_lock

一个线程按`A`、`B`的顺序加锁，另一个线程按`B`、`A`的顺序加锁，两者各持有一把锁并等待对方释放，就会产生死锁。C++17 提供的`std::scoped_lock`可以一次锁住多个互斥量，内部使用避免死锁的算法，与书写顺序无关

```c++
#include <iostream>
#include <mutex>
#include <thread>

struct Account
{
    std::mutex mtx;
    int balance = 100;
};

void transfer(Account &from, Account &to, int amount)
{
    std::scoped_lock lock(from.mtx, to.mtx); // 同时锁住两个互斥量
    from.balance -= amount;
    to.balance += amount;
}

int main()
{
    Account a, b;

    std::thread th1(transfer, std::ref(a), std::ref(b), 10);
    std::thread th2(transfer, std::ref(b), std::ref(a), 20);

    th1.join();
    th2.join();

    std::cout << a.balance << ' ' << b.balance << std::endl; // 110 90

    return 0;
}
```
{: file='std::scoped_lock 的使用'}

只锁一个互斥量时，`std::scoped_lock`与`std::lock_guard`等价

#### std::shared_mutex

读多写少的场景下，多个线程同时读取并不会产生冲突，用`std::mutex`会让读线程之间也互相等待。C++17 在`<shared_mutex>`中提供了读写锁`std::shared_mutex`：

- `std::shared_lock`获取共享锁，多个线程可以同时持有
- `std::unique_lock`或`std::lock_guard`获取独占锁，与其他所有读写线程互斥

```c++
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

std::shared_mutex config_mtx;
std::map<std::string, int> config{ {"timeout", 30} };

int read_config(const std::string &key)
{
    std::shared_lock lock(config_mtx); // 共享锁，多个读线程可以同时持有
    return config.at(key);
}

void write_config(const std::string &key, int value)
{
    std::unique_lock lock(config_mtx); // 独占锁
    config[key] = value;
}

int main()
{
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; i++)
    {
        readers.emplace_back([] { read_config("timeout"); });
    }
    std::thread writer(write_config, "timeout", 60);

    for (std::thread &th : readers)
    {
        th.join();
    }
    writer.join();

    std::cout << read_config("timeout") << std::endl; // 60

    return 0;
}
```
{: file='std::shared_mutex 的使用'}

### std::call_once

`std::call_once`配合`std::once_flag`，保证一个函数在多个线程中只被执行一次，常用于延迟初始化

```c++
#include <iostream>
#include <mutex>
#include <thread>

std::once_flag flag;

void init()
{
    std::cout << "init" << std::endl;
}

void worker()
{
    std::call_once(flag, init); // 无论多少个线程调用，init只执行一次
}

int main()
{
    std::thread th[4];
    for (std::thread &x : th)
    {
        x = std::thread(worker);
    }
    for (std::thread &x : th)
    {
        x.join();
    }

    return 0;
}
```
{: file='std::call_once 的使用'}

C++11 起，函数内`static`局部变量的初始化本身就是线程安全的，单例等场景直接使用`static`局部变量即可

### std::atomic

`std::mutex`很好地解决了多线程资源争抢的问题，但加锁和解锁都有开销，使用不当还会产生死锁。对于计数器、标志位这样的单个变量，可以改用`std::atomic`

例六：

```c++
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> n;

void count10000()
{
    for (int i = 1; i <= 10000; i++)
    {
        n++;
    }
}

int main()
{
    n = 0;
    std::thread th[100];
    for (std::thread &x : th)
    {
        x = std::thread(count10000);
    }
    for (std::thread &x : th)
    {
        x.join();
    }

    std::cout << n << std::endl;

    return 0;
}
```
{: file='std::atomic 的使用'}

输出结果同样为 1000000

> 原子操作是不可分割的操作，其他线程只能看到操作完成之前或完成之后的状态，看不到中间状态

`n++`对普通变量来说包含读取、加一、写回三步，多个线程交替执行就会丢失更新；对`std::atomic<int>`来说则是一个原子操作，不需要加锁

**注意事项：**

- `std::atomic`只保证单个操作是原子的，`n = n + 1`包含一次读取和一次写入，不是原子操作，应写成`n++`或`n += 1`
- 原子变量不能拷贝，也不能用另一个原子变量赋值
- 需要保护多个变量或一段代码时，仍应使用`std::mutex`

`std::atomic`常用成员函数

| 函数                                               | 作用 |
| :------------------------------------------------- | :--- |
| `T load()`                                         | 读取当前值 |
| `void store(T val)`                                | 写入新值 |
| `T exchange(T val)`                                | 写入新值并返回旧值 |
| `bool compare_exchange_strong(T& expected, T val)` | 当前值等于`expected`时写入`val`并返回`true`<br>否则把当前值写入`expected`并返回`false` |
| `T fetch_add(T val)`<br>`T fetch_sub(T val)`       | 加上、减去`val`并返回旧值（整数和指针类型） |

`std::atomic`构造函数

| 函数                               | 类型           | 作用 |
| :--------------------------------- | :------------- | :--- |
| `constexpr atomic() noexcept(…)`   | 默认构造函数   | 构造`std::atomic`对象<br>（C++20 起进行值初始化，如`std::atomic<int>`初值为 0；`T`的默认构造不抛异常时为`noexcept`） |
| `constexpr atomic(T val) noexcept` | 初始化构造函数 | 构造一个`std::atomic`对象<br>用`val`的值来初始化 |
| `atomic(const atomic&) = delete`   | 复制构造函数   | 删除`std::atomic`类型的拷贝构造函数 |

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

例七：

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

例八：

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

例九：

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

例十：

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

例十一：
以例八中的代码为基础加以修改

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

## condition_variable

`<condition_variable>`头文件主要包含了与条件变量相关的类和函数  
包括：

- 类
  - `std::condition_variable`
  - `std::condition_variable_any`
- 枚举类型
  - `std::cv_status`
- 函数
  - `std::notify_all_at_thread_exit()`

`std::condition_variable`必须结合`std::unique_lock`使用，`std::condition_variable_any`可以使用任何的锁

`std::condition_variable`常用成员函数

| 函数               | 作用 |
| :----------------- | :--- |
| `wait(lock)`       | 使当前线程等待，直到其他线程通知或唤醒<br>需要一个已经上锁的`std::unique_lock`对象作为参数 |
| `wait(lock, pred)` | 额外提供一个条件函数<br>当`pred`返回`false`时等待，被唤醒后重新检查<br>`pred`：返回`true`时才返回（可防止虚假唤醒） |
| `notify_one()`     | 通知等待在条件变量上的一个线程，使其从等待状态唤醒 |
| `notify_all()`     | 通知等待在条件变量上的所有线程，使它们从等待状态唤醒 |

条件变量用于让线程等待某个条件成立：等待的线程在条件不满足时释放锁并休眠，其他线程修改条件后通知它醒来。条件本身必须由互斥量保护，并且等待时要使用带`pred`的`wait`，否则可能因虚假唤醒或通知先于等待而出错

例十二：

```c++
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
std::queue<int> tasks;
bool finished = false;

void producer()
{
    for (int i = 1; i <= 5; i++)
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            tasks.push(i);
        }
        cv.notify_one(); // 先解锁再通知，被唤醒的线程可以立即拿到锁
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
    }
    cv.notify_all();
}

void consumer()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !tasks.empty() || finished; }); // 等待期间自动释放锁

        if (tasks.empty())
        {
            break; // 队列已空且生产结束
        }

        int task = tasks.front();
        tasks.pop();
        lock.unlock(); // 处理任务时不需要持有锁

        std::cout << "task " << task << std::endl;
    }
}

int main()
{
    std::thread th1(producer);
    std::thread th2(consumer);

    th1.join();
    th2.join();

    return 0;
}
```
{: file='生产者-消费者'}

输出结果：

```text
task 1
task 2
task 3
task 4
task 5
```

## thread_local

用`thread_local`声明的变量在每个线程中各有一份独立的副本，线程之间互不影响，不需要加锁。常用于每个线程各自的缓存、计数器、随机数生成器等

```c++
#include <iostream>
#include <mutex>
#include <thread>

thread_local int counter = 0; // 每个线程各有一份
std::mutex print_mtx;

void work(int id)
{
    for (int i = 0; i < 1000 * id; i++)
    {
        counter++; // 只修改当前线程的副本，不需要加锁
    }

    std::lock_guard<std::mutex> lock(print_mtx);
    std::cout << "thread " << id << ": " << counter << std::endl;
}

int main()
{
    std::thread th1(work, 1);
    std::thread th2(work, 2);

    th1.join();
    th2.join();

    std::cout << "main: " << counter << std::endl; // main: 0

    return 0;
}
```
{: file='thread_local 的使用'}

两个线程分别输出 1000 和 2000，主线程的`counter`仍为 0

## std::jthread

C++20 在`<thread>`中新增了`std::jthread`，与`std::thread`相比有两点改进：

- 析构时自动`join`，不会因为忘记`join`而调用`std::terminate()`
- 内置停止机制，线程函数的第一个参数为`std::stop_token`时，可以通过`request_stop()`请求线程退出

例十三：

```c++
#include <chrono>
#include <iostream>
#include <thread>

void worker(std::stop_token token)
{
    while (!token.stop_requested())
    {
        std::cout << "working" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "stopped" << std::endl;
}

int main()
{
    std::jthread th(worker);

    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    th.request_stop(); // 请求线程退出

    return 0; // 析构时自动join
}
```
{: file='std::jthread 的使用'}
