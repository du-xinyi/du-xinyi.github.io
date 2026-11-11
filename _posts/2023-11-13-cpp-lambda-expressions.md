---
title: C++ Lambda Expressions
description: 介绍现代 C++ Lambda 表达式的语法、捕获方式、泛型 Lambda 与使用中的注意事项
date: 2023-11-13 18:30:00 +0800
categories: [ Development, C++ ]
tags: [ C++, Lambda ]
---

Lambda 表达式是 C++11 引入的语法，用于在调用处或作为参数传递的位置直接定义一个匿名函数对象。通常，Lambda 用于封装传递给算法或异步方法的几行代码

## Lambda 表达式定义

```c++
#include <algorithm>
#include <cmath>

void abssort(float *x, unsigned n)
{
    std::sort(x, x + n,
        // Lambda expression begins
        [](float a, float b) {
            return (std::abs(a) < std::abs(b));
        } // end of lambda expression
    );
}
```
{: file='Lambda 表达式示例'}

在上面的示例中`std::sort`函数的第三个参数应该传递一个排序规则的函数，这里直接将排序规则的实现写在了传递参数的位置，省去了单独定义函数的过程。对于这种不需要复用且短小的函数，直接传递函数体可以增加代码的可读性

## Lambda 表达式语法定义

![Lambda 表达式的语法组成：捕获列表、参数列表、可变规则、异常说明、返回类型、函数体](posts/2023-11-13-cpp-lambda-expressions/语法定义.png)

1. 捕获列表。捕获列表总是出现在 Lambda 表达式的开始处，`[]`是 Lambda 引出符，编译器根据它判断接下来的代码是否是 Lambda 表达式。捕获列表能够捕获上下文中的变量以供 Lambda 使用
2. 参数列表。与普通函数的参数列表一致。如果不需要传递参数，可以连同括号`()`一起省略
3. 可变规则。即`mutable`修饰符，默认情况下 Lambda 的函数调用运算符是`const`的，不能修改按值捕获的变量，`mutable`可以取消其常量性
4. 异常说明。使用`noexcept`表明 Lambda 不会抛出异常（图中的`throw()`是 C++11 的旧写法，已在 C++20 中移除）
5. 返回类型。用尾置返回类型的形式声明，即`-> 类型`。大多数情况下可以省略，由编译器根据`return`语句推导
6. 函数体。内容与普通函数一样，除了可以使用参数之外，还可以使用所有捕获的变量

## Lambda 表达式参数详解

### Lambda 捕获列表

Lambda 表达式与普通函数最大的区别是，除了可以使用参数以外，还可以通过捕获列表访问上下文中的局部变量。捕获列表描述了哪些变量可以被 Lambda 使用，以及使用方式（按值或按引用）。捕获列表由多个捕获项组成，以逗号分隔，有以下几种形式：

- `[]`不捕获任何变量

```c++
auto function = [] {
    std::cout << "Hello World!" << std::endl;
};

function();
```
{: file='不捕获任何变量'}

- `[var]`按值捕获变量`var`，捕获的是 Lambda 创建时的副本，之后外部变量的修改不会影响 Lambda 内部

```c++
int num = 100;
auto function = [num] {
    std::cout << num << std::endl;
};

num = 200;
function(); // 100
```
{: file='按值捕获变量 var'}

- `[=]`按值捕获函数体中用到的所有局部变量

```c++
int index = 1;
int num = 100;
auto function = [=] {
    std::cout << "index: " << index << ", "
              << "num: " << num << std::endl;
};

function();
```
{: file='按值捕获所有用到的变量'}

- `[&var]`按引用捕获变量`var`

```c++
int num = 100;
auto function = [&num] {
    num = 1000;
};

function();
std::cout << "num: " << num << std::endl; // num: 1000
```
{: file='按引用捕获变量 var'}

- `[&]`按引用捕获函数体中用到的所有局部变量

```c++
int index = 1;
int num = 100;
auto function = [&] {
    num = 1000;
    index = 2;
};

function();
std::cout << "index: " << index << ", "
          << "num: " << num << std::endl; // index: 2, num: 1000
```
{: file='按引用捕获所有用到的变量'}

- `[this]`捕获当前对象的`this`指针，Lambda 内可以访问成员变量和成员函数
- `[*this]`按值捕获当前对象的副本（C++17），Lambda 的生命周期可以超出对象本身

```c++
#include <iostream>

class Lambda
{
public:
    void sayHello()
    {
        std::cout << "Hello" << std::endl;
    }

    void lambda()
    {
        auto function = [this] {
            sayHello();
        };

        function();
    }
};

int main()
{
    Lambda demo;
    demo.lambda();

    return 0;
}
```
{: file='捕获 this 指针'}

- 混合捕获
  - `[=, &a, &b]`按引用捕获变量`a`和`b`，按值捕获其它用到的变量
  - `[&, a]`按值捕获变量`a`，按引用捕获其它用到的变量

```c++
int index = 1;
int num = 100;
int factor = 2;
auto function = [=, &index, &num] {
    num = 1000 * factor;
    index = 2;
};

function();
std::cout << "index: " << index << ", "
          << "num: " << num << std::endl; // index: 2, num: 2000
```
{: file='按引用捕获 index 和 num，按值捕获其它变量'}

- `[name = 表达式]`初始化捕获（C++14），在捕获列表中定义一个新变量，可以用来移动只能移动的对象

```c++
#include <iostream>
#include <memory>
#include <utility>

int main()
{
    auto ptr = std::make_unique<int>(42);
    auto function = [p = std::move(ptr)] {
        std::cout << *p << std::endl;
    };

    function(); // 42

    return 0;
}
```
{: file='初始化捕获'}

捕获列表不允许重复捕获同一个变量，否则会导致编译错误。例如：

- `[=, a]`默认已经是按值捕获，再按值捕获`a`是重复
- `[&, &a]`默认已经是按引用捕获，再按引用捕获`a`是重复

如果 Lambda 函数体中按引用访问外部变量`total`，按值访问外部变量`factor`，以下几种捕获列表是等效的：

```c++
[&total, factor]
[factor, &total]
[&, factor]
[=, &total]
```
{: file='等效的捕获列表'}

**注意事项：**

- 按引用捕获时，要保证 Lambda 被调用时变量仍然存在。把按引用捕获了局部变量的 Lambda 返回到函数外，或交给另一个线程异步执行，变量销毁后引用就会悬空
- `[=]`在成员函数中会隐式捕获`this`指针（C++20 起不推荐，需要时显式写成`[=, this]`），此时访问的仍是原对象的成员，而不是副本
- 全局变量和静态变量不需要捕获，可以直接使用

### Lambda 参数列表

参数列表是可选的，在大多数方面与普通函数的参数列表相同

```c++
auto function = [](int first, int second) {
    return first + second;
};

function(100, 200);
```
{: file='Lambda 参数列表'}

C++14 起参数类型可以写成`auto`，称为泛型 Lambda，相当于函数调用运算符是一个模板

```c++
auto add = [](auto first, auto second) {
    return first + second;
};

add(1, 2);                                // 3
add(1.5, 2.5);                            // 4.0
add(std::string("a"), std::string("b"));  // "ab"
```
{: file='泛型 Lambda'}

### 可变规则 mutable

默认情况下不能在 Lambda 中修改按值捕获的变量，加上`mutable`后可以修改，但修改的是 Lambda 内部的副本，不会影响外部变量

```c++
#include <iostream>

int main()
{
    int m = 0;
    int n = 0;
    [&, n](int a) mutable { m = ++n + a; }(4);
    std::cout << m << std::endl << n << std::endl;

    return 0;
}
```
{: file='可变规则 mutable'}

输出结果：

```text
5
0
```

`m`按引用捕获，被修改为 5；`n`按值捕获，Lambda 内部的副本变为 1，外部的`n`仍为 0

### 异常说明

可以使用`noexcept`表明 Lambda 不会抛出任何异常。与普通函数一样，如果声明了`noexcept`的 Lambda 在函数体中抛出异常，程序会调用`std::terminate`终止

```c++
int main()
{
    []() noexcept { throw 5; }(); // 调用std::terminate

    return 0;
}
```
{: file='异常说明'}

### 返回类型

省略返回类型时，编译器根据`return`语句推导：有多个`return`语句时，它们推导出的类型必须一致；没有`return`语句（或只有`return;`）时，推导为`void`。需要指定返回类型时，使用尾置返回类型写在参数列表之后，此时参数列表的括号不能省略

```c++
auto x1 = [](int i) { return i; };                  // 推导为int
auto x2 = [](int i) -> double { return i; };        // 指定为double
auto x3 = [](bool flag) -> double {
    if (flag)
    {
        return 1; // 两个return的类型不同，必须指定返回类型
    }
    return 2.5;
};
```
{: file='返回类型'}

### Lambda 函数体

Lambda 的函数体可以包含普通函数体能包含的任何内容，可以访问以下几类变量：

- 捕获的变量
- 参数
- 函数体内声明的局部变量
- 类的数据成员（在成员函数中定义并捕获了`this`时）
- 具有静态存储期的变量，例如全局变量

## Lambda 的类型与存储

每个 Lambda 表达式的类型都是编译器生成的唯一的匿名类型，无法直接写出，只能用`auto`接收。需要把 Lambda 保存为成员变量、放进容器或作为函数参数时，有以下几种方式：

```c++
#include <functional>
#include <iostream>
#include <vector>

// 模板参数：没有额外开销，可以内联
template <class Fn>
void call(Fn fn)
{
    fn();
}

int main()
{
    int base = 10;

    // std::function：可以保存任何签名匹配的可调用对象，有少量运行时开销
    std::vector<std::function<int(int)>> functions;
    functions.push_back([](int x) { return x + 1; });
    functions.push_back([base](int x) { return x + base; });

    for (const auto &function : functions)
    {
        std::cout << function(1) << std::endl; // 2 11
    }

    // 不捕获任何变量的Lambda可以转换为函数指针
    int (*pointer)(int) = [](int x) { return x * 2; };
    std::cout << pointer(4) << std::endl; // 8

    call([] { std::cout << "called" << std::endl; });

    return 0;
}
```
{: file='Lambda 的存储'}

## 递归 Lambda

Lambda 没有名字，用`auto`接收时，在其初始化完成之前也不能引用自身，因此不能直接递归调用。常用的写法有两种

```c++
#include <functional>
#include <iostream>

int main()
{
    // 方式一：用std::function保存，并按引用捕获自身
    std::function<int(int)> factorial = [&factorial](int n) {
        return n <= 1 ? 1 : n * factorial(n - 1);
    };

    // 方式二：把自身作为参数传入，没有std::function的开销
    auto fibonacci = [](auto &&self, int n) -> int {
        return n < 2 ? n : self(self, n - 1) + self(self, n - 2);
    };

    std::cout << factorial(5) << std::endl;             // 120
    std::cout << fibonacci(fibonacci, 10) << std::endl; // 55

    return 0;
}
```
{: file='递归 Lambda'}

C++23 起可以用显式对象参数`this auto self`直接取得 Lambda 自身，调用时不需要再把自己传进去（需要 GCC 14 或更高版本）

```c++
auto fibonacci = [](this auto self, int n) -> int {
    return n < 2 ? n : self(n - 1) + self(n - 2);
};

fibonacci(10); // 55
```
{: file='C++23 的写法'}

## 编译期计算

C++17 起，Lambda 只要满足常量表达式的要求就可以在编译期求值

```c++
constexpr auto square = [](int x) { return x * x; };

static_assert(square(4) == 16); // 编译期完成计算
int values[square(3)];          // 可用于需要常量的位置
```
{: file='constexpr Lambda'}
