---
title: C++ Containers
description: 梳理 C++ STL 序列容器、关联容器、无序容器与容器适配器的特性、接口差异和使用方式
date: 2023-10-21 21:30:00 +0800
categories: [ Development, C++ ]
tags: [ C++, STL ]
---

容器是一些模板类的集合，其中封装的是组织数据的方法（也就是数据结构）。STL 提供有 3 类标准容器，分别是序列容器、关联容器和无序容器（哈希容器），另外还有在它们之上封装出的容器适配器

| 容器种类   | 包含的容器 | 特点                                           |
| :--------- | :--- | :--------------------------------------------- |
| 序列容器   | `std::array`<br>`std::vector`<br>`std::deque`<br>`std::list`<br>`std::forward_list` | 元素的位置与值无关，放在什么位置就位于什么位置 |
| 关联容器   | `std::set`<br>`std::multiset`<br>`std::map`<br>`std::multimap` | 元素按键排序，插入时自动放到合适的位置         |
| 无序容器   | `std::unordered_set`<br>`std::unordered_multiset`<br>`std::unordered_map`<br>`std::unordered_multimap` | 元素的位置由哈希函数确定，不保证顺序           |
| 容器适配器 | `std::stack`<br>`std::queue`<br>`std::priority_queue` | 在其他容器之上限制接口，只提供特定的访问方式   |

## 容器的选择

各容器的底层结构，以及访问和查找元素的复杂度

| 容器                | 底层结构     | 随机访问 | 查找                 |
| :------------------ | :----------- | :------- | :------------------- |
| `std::array`        | 固定长度数组 | O(1)     | O(n)                 |
| `std::vector`       | 动态数组     | O(1)     | O(n)                 |
| `std::deque`        | 分段连续数组 | O(1)     | O(n)                 |
| `std::list`         | 双向链表     | 不支持   | O(n)                 |
| `std::forward_list` | 单向链表     | 不支持   | O(n)                 |
| 关联容器            | 红黑树       | 不支持   | O(log n)             |
| 无序容器            | 哈希表       | 不支持   | 平均 O(1)，最坏 O(n) |

序列容器在不同位置插入和删除元素的复杂度

| 容器                | 头部增删 | 尾部增删  | 中间增删 |
| :------------------ | :------- | :-------- | :------- |
| `std::array`        | 不支持   | 不支持    | 不支持   |
| `std::vector`       | O(n)     | 均摊 O(1) | O(n)     |
| `std::deque`        | O(1)     | O(1)      | O(n)     |
| `std::list`         | O(1)     | O(1)      | O(1)     |
| `std::forward_list` | O(1)     | 不支持    | O(1)     |

关联容器和无序容器的元素位置由键决定，没有头部、尾部之分，插入和删除的复杂度与查找相同

- 链表在中间增删的 O(1) 指已经持有该位置的迭代器，找到该位置本身仍需 O(n)
- 没有特殊需求时优先使用`std::vector`，连续内存对缓存友好，实际性能通常优于链表
- 需要按键查找时，不要求顺序用无序容器，需要有序遍历或范围查询用关联容器

![vector、deque、list、forward_list 的内存布局](posts/2023-10-21-cpp-containers/memory_layout.png)

## 通用接口

以下成员函数为大多数容器所共有，后文不再重复

| 成员函数              | 功能 |
| :-------------------- | :--- |
| `begin()` / `end()`   | 返回指向第一个元素、最后一个元素之后一个位置的迭代器 |
| `cbegin()` / `cend()` | 同上，返回的是`const`迭代器，不能通过它修改元素 |
| `rbegin()` / `rend()` | 返回反向迭代器，用于逆序遍历（`std::forward_list`和无序容器没有） |
| `empty()`             | 判断容器是否为空 |
| `size()`              | 返回元素个数（`std::forward_list`没有） |
| `max_size()`          | 返回理论上可容纳的最大元素个数，与实现和元素类型有关 |
| `clear()`             | 移除所有元素（`std::array`没有） |
| `swap()`              | 交换两个同类型容器的内容 |

迭代器的能力由底层结构决定

| 迭代器类型     | 支持的操作                   | 对应容器                                  |
| :------------- | :--------------------------- | :---------------------------------------- |
| 随机访问迭代器 | `++`、`--`、`+n`、`-n`、`[]` | `std::array`、`std::vector`、`std::deque` |
| 双向迭代器     | `++`、`--`                   | `std::list`、关联容器                     |
| 前向迭代器     | `++`                         | `std::forward_list`、无序容器             |

`std::sort`要求随机访问迭代器，因此不能用于链表，链表需使用自带的`sort()`成员函数

## 序列容器

各序列容器提供的成员函数对比如下

| 成员函数 | array | vector | deque | list | forward_list | 功能                                   |
| :--- | :---: | :----: | :---: | :--: | :----------: | :------------------------------------- |
| `operator[]` |   ✓   |   ✓    |   ✓   |  —   |      —       | 按下标访问，不检查边界                 |
| `at()` |   ✓   |   ✓    |   ✓   |  —   |      —       | 按下标访问，越界时抛出异常             |
| `front()` |   ✓   |   ✓    |   ✓   |  ✓   |      ✓       | 返回第一个元素的引用                   |
| `back()` |   ✓   |   ✓    |   ✓   |  ✓   |      —       | 返回最后一个元素的引用                 |
| `data()` |   ✓   |   ✓    |   —   |  —   |      —       | 返回指向底层数组的指针                 |
| `fill()` |   ✓   |   —    |   —   |  —   |      —       | 把所有元素设为同一个值                 |
| `assign()` |   —   |   ✓    |   ✓   |  ✓   |      ✓       | 用新元素替换全部内容                   |
| `resize()` |   —   |   ✓    |   ✓   |  ✓   |      ✓       | 改变元素个数                           |
| `capacity()`<br>`reserve()` |   —   |   ✓    |   —   |  —   |      —       | 查询、预留容量                         |
| `shrink_to_fit()` |   —   |   ✓    |   ✓   |  —   |      —       | 请求释放多余的内存（非强制）           |
| `push_back()`<br>`emplace_back()`<br>`pop_back()` |   —   |   ✓    |   ✓   |  ✓   |      —       | 在尾部添加、就地构造、移除元素         |
| `push_front()`<br>`emplace_front()`<br>`pop_front()` |   —   |   —    |   ✓   |  ✓   |      ✓       | 在头部添加、就地构造、移除元素         |
| `insert()`<br>`emplace()`<br>`erase()` |   —   |   ✓    |   ✓   |  ✓   |      —       | 在指定位置插入、就地构造、移除元素     |
| `insert_after()`<br>`emplace_after()`<br>`erase_after()` |   —   |   —    |   —   |  —   |      ✓       | 在指定位置之后插入、就地构造、移除元素 |
| `splice()`<br>`merge()`<br>`sort()`<br>`reverse()`<br>`unique()`<br>`remove()`<br>`remove_if()` |   —   |   —    |   —   |  ✓   |      ✓       | 链表特有操作                           |

`front()`、`back()`、`pop_back()`、`pop_front()`和`operator[]`都不检查容器是否为空或下标是否越界，对空容器调用是未定义行为。`std::forward_list`中与`splice()`对应的是`splice_after()`

### std::array

定义在`<array>`头文件中。长度在编译期确定，不能增加或删除元素，只能修改元素的值，是对内置数组的封装，没有额外开销

```c++
#include <array>

std::array<double, 10> values1;                        // 10个元素，值未初始化
std::array<double, 10> values2{};                      // 所有元素初始化为0.0
std::array<double, 10> values3{0.5, 1.0, 1.5, 2.0};    // 初始化前4个元素，其余为0.0

values2.fill(3.14);                                    // 所有元素设为3.14
```
{: file='初始化'}

### std::vector

定义在`<vector>`头文件中。元素存放在一段连续内存中，空间不足时会申请一块更大的内存，并把原有元素移动过去

```c++
#include <vector>

std::vector<double> values1;                           // 空容器
std::vector<int> primes{2, 3, 5, 7, 11, 13, 17, 19};   // 含有8个元素
std::vector<double> values2(20);                       // 20个元素，初始值都为0.0
std::vector<double> values3(20, 1.0);                  // 20个元素，初始值都为1.0
std::vector<double> values4(values3);                  // 拷贝另一个容器
std::vector<int> values5(primes.begin(), primes.begin() + 3); // 拷贝一段区间：2,3,5
```
{: file='初始化'}

注意圆括号与花括号的区别：`std::vector<int> v(20, 1)`是 20 个 1，`std::vector<int> v{20, 1}`是 20 和 1 两个元素

`size()`是实际元素个数，`capacity()`是不重新分配内存时最多能容纳的元素个数。已知元素个数时先`reserve()`可以避免多次扩容

```c++
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> values;
    values.reserve(4);

    for (int i = 1; i <= 6; i++)
    {
        values.push_back(i);
        std::cout << "size: " << values.size()
                  << ", capacity: " << values.capacity() << std::endl;
    }

    values.insert(values.begin() + 1, 10); // 1 10 2 3 4 5 6
    values.erase(values.begin());          // 10 2 3 4 5 6

    for (int x : values)
    {
        std::cout << x << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='容量与增删'}

`push_back()`接收一个已有的对象，`emplace_back()`接收构造函数的参数并在容器内就地构造，可以省去一次临时对象的构造

```c++
#include <string>
#include <utility>
#include <vector>

int main()
{
    std::vector<std::pair<std::string, int>> items;

    items.push_back({"a", 1});  // 先构造pair，再移动到容器中
    items.emplace_back("b", 2); // 直接在容器中构造pair

    return 0;
}
```
{: file='push_back 与 emplace_back'}

**迭代器失效：**`std::vector`扩容后，原有的迭代器、指针和引用全部失效；`insert()`、`erase()`会使操作位置及其之后的迭代器失效。遍历过程中删除元素要使用`erase()`的返回值

```c++
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> values{1, 2, 3, 4, 5, 6};

    for (auto it = values.begin(); it != values.end();)
    {
        if (*it % 2 == 0)
        {
            it = values.erase(it); // erase返回被删除元素的下一个位置
        }
        else
        {
            ++it;
        }
    }

    // C++20起可以直接写成：std::erase_if(values, [](int x) { return x % 2 == 0; });

    for (int x : values)
    {
        std::cout << x << ' '; // 1 3 5
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='遍历时删除元素'}

### std::deque

定义在`<deque>`头文件中。由多段连续内存组成，头部和尾部的增删都是 O(1)，支持随机访问，但元素整体并不连续，因此没有`data()`、`capacity()`和`reserve()`

```c++
#include <deque>

std::deque<int> d1;                 // 空容器
std::deque<int> d2(10);             // 10个元素，初始值都为0
std::deque<int> d3(10, 5);          // 10个元素，初始值都为5
std::deque<int> d4{1, 2, 3};

d4.push_front(0);                   // 0 1 2 3
d4.push_back(4);                    // 0 1 2 3 4
d4.pop_front();                     // 1 2 3 4
```
{: file='初始化与头尾增删'}

在头尾插入元素会使所有迭代器失效，但指向已有元素的指针和引用仍然有效；在中间插入或删除则全部失效

### std::list

定义在`<list>`头文件中。以双向链表组织元素，在任何位置增删都不会移动其他元素，指向其他元素的迭代器始终有效；不支持随机访问，访问第`n`个元素需要从头遍历

```c++
#include <iostream>
#include <list>

int main()
{
    std::list<int> a{3, 1, 2};
    std::list<int> b{6, 5, 4};

    a.sort();
    b.sort();
    a.merge(b);                                 // 合并两个有序链表，a: 1 2 3 4 5 6，b变为空
    a.remove_if([](int x) { return x > 4; });   // 1 2 3 4
    a.reverse();                                // 4 3 2 1

    std::list<int> c{0, 0};
    a.splice(a.begin(), c);                     // 把c的元素移到a的开头：0 0 4 3 2 1
    a.unique();                                 // 相邻的重复元素只保留一个：0 4 3 2 1

    for (int x : a)
    {
        std::cout << x << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='链表特有操作'}

### std::forward_list

定义在`<forward_list>`头文件中。以单向链表组织元素，每个节点比`std::list`少一个指针，只能从前向后遍历。由于无法访问前一个节点，插入和删除都作用在指定位置**之后**，并提供`before_begin()`用于在头部操作；为节省空间不提供`size()`

```c++
#include <forward_list>
#include <iostream>

int main()
{
    std::forward_list<int> values{2, 3};

    values.push_front(1);                         // 1 2 3
    values.insert_after(values.before_begin(), 0); // 0 1 2 3
    values.erase_after(values.begin());           // 删除第一个元素之后的元素：0 2 3

    for (int x : values)
    {
        std::cout << x << ' ';
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='在指定位置之后增删'}

## 关联容器

关联容器按键的大小排序存储元素，默认使用`std::less`升序排列

| 容器            | 说明                                            |
| :-------------- | :---------------------------------------------- |
| `std::set`      | 定义在`<set>`头文件中，元素本身就是键，不能重复 |
| `std::multiset` | 定义在`<set>`头文件中，元素可以重复             |
| `std::map`      | 定义在`<map>`头文件中，存储键值对，键不能重复   |
| `std::multimap` | 定义在`<map>`头文件中，键可以重复               |

除通用接口外，4 种关联容器共有的成员函数如下

| 成员函数           | 功能 |
| :----------------- | :--- |
| `find(key)`        | 返回指向键等于`key`的元素的迭代器，没找到返回`end()` |
| `contains(key)`    | 判断是否存在键等于`key`的元素（C++20） |
| `count(key)`       | 返回键等于`key`的元素个数，`set`和`map`的结果只会是 0 或 1 |
| `lower_bound(key)` | 返回指向第一个键大于或等于`key`的元素的迭代器 |
| `upper_bound(key)` | 返回指向第一个键大于`key`的元素的迭代器 |
| `equal_range(key)` | 返回一对迭代器，分别等价于`lower_bound(key)`和`upper_bound(key)` |
| `insert()`         | 插入元素 |
| `emplace()`        | 就地构造并插入元素 |
| `emplace_hint()`   | 同`emplace()`，额外提供一个表示插入位置的迭代器作为提示 |
| `erase()`          | 按迭代器、键或区间删除元素 |
| `extract()`        | 取出一个节点而不销毁元素，可用于修改键后重新插入（C++17） |
| `merge()`          | 把另一个容器中的元素转移过来（C++17） |

元素的键是只读的，不能通过迭代器修改，否则会破坏排序。插入和删除不会使指向其他元素的迭代器失效

### std::map

```c++
#include <map>
#include <string>

std::map<std::string, int> map1;                                   // 空容器
std::map<std::string, int> map2{ {"C++", 10}, {"Container", 20} }; // 包含2个键值对
std::map<std::string, int> map3(map2);                             // 拷贝另一个容器
std::map<std::string, int> map4(++map2.begin(), map2.end());       // 拷贝一段区间：{"Container", 20}
```
{: file='初始化'}

`std::map`在共有接口之外，还提供按键访问值的成员函数

| 成员函数                       | 功能                                                       |
| :----------------------------- | :--------------------------------------------------------- |
| `operator[key]`                | 返回键对应的值的引用，键不存在时会插入一个值为默认值的元素 |
| `at(key)`                      | 返回键对应的值的引用，键不存在时抛出`std::out_of_range`    |
| `try_emplace(key, args...)`    | 键不存在时才构造并插入，键已存在时不做任何事（C++17）      |
| `insert_or_assign(key, value)` | 键不存在时插入，键已存在时覆盖其值（C++17）                |

```c++
#include <iostream>
#include <map>
#include <string>

int main()
{
    std::map<std::string, int> scores{ {"Alice", 90}, {"Bob", 80} };

    scores["Carol"] = 85;               // 键不存在，插入
    scores.insert({"Alice", 0});        // 键已存在，不插入也不修改
    scores.insert_or_assign("Bob", 95); // 键已存在，覆盖
    scores.try_emplace("Dave", 70);     // 键不存在，插入

    if (auto it = scores.find("Eve"); it == scores.end())
    {
        std::cout << "Eve not found" << std::endl;
    }

    scores.erase("Carol");

    for (const auto &[name, score] : scores) // 按键升序遍历
    {
        std::cout << name << ": " << score << std::endl;
    }

    return 0;
}
```
{: file='std::map 的使用'}

输出结果：

```text
Eve not found
Alice: 90
Bob: 95
Dave: 70
```

只是查询某个键是否存在时，应使用`find()`或`contains()`。`scores["Eve"]`会在键不存在时插入一个值为 0 的元素，并且不能用于`const`容器

### std::set

```c++
#include <iostream>
#include <set>

int main()
{
    std::set<int> values{3, 1, 2, 3}; // 重复的3只保留一个

    auto [it, inserted] = values.insert(2); // 元素已存在，inserted为false
    std::cout << std::boolalpha << inserted << std::endl;

    for (int x : values)
    {
        std::cout << x << ' '; // 1 2 3
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='std::set 的使用'}

### std::multimap 与 std::multiset

键可以重复，因此没有`operator[]`和`at()`，`count()`的结果可以大于 1。同一个键对应的所有元素通过`equal_range()`获取

```c++
#include <iostream>
#include <map>
#include <string>

int main()
{
    std::multimap<std::string, int> records{ {"a", 1}, {"b", 2}, {"a", 3} };

    std::cout << records.count("a") << std::endl; // 2

    auto [first, last] = records.equal_range("a");
    for (auto it = first; it != last; ++it)
    {
        std::cout << it->second << ' '; // 1 3
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='std::multimap 的使用'}

### 自定义排序规则

排序规则通过模板参数指定，可以是标准库的函数对象，也可以是自定义的比较类型。两个元素互相都不"小于"对方时被视为等价，在`set`和`map`中只会保留一个

```c++
#include <functional>
#include <iostream>
#include <set>
#include <string>

struct ByLength
{
    bool operator()(const std::string &a, const std::string &b) const
    {
        return a.size() < b.size();
    }
};

int main()
{
    std::set<int, std::greater<int>> desc{1, 3, 2}; // 降序：3 2 1

    std::set<std::string, ByLength> words{"ccc", "a", "bb", "dd"};
    for (const auto &word : words)
    {
        std::cout << word << ' '; // a bb ccc，"dd"与"bb"长度相同，被视为重复
    }
    std::cout << std::endl;

    return 0;
}
```
{: file='自定义排序规则'}

## 无序容器

无序容器用哈希表存储元素，查找、插入、删除的平均复杂度为 O(1)，但遍历顺序不确定

| 容器                                            | 说明                            |
| :---------------------------------------------- | :------------------------------ |
| `std::unordered_set`、`std::unordered_multiset` | 定义在`<unordered_set>`头文件中 |
| `std::unordered_map`、`std::unordered_multimap` | 定义在`<unordered_map>`头文件中 |

用法与对应的关联容器基本相同，区别在于：

- 没有`lower_bound()`、`upper_bound()`，`equal_range(key)`返回所有键等于`key`的元素构成的范围
- 迭代器是前向迭代器，没有`rbegin()`、`rend()`
- 键的类型需要支持哈希和`==`比较，而不是`<`比较

另外提供一组与哈希表有关的成员函数

| 成员函数            | 功能 |
| :------------------ | :--- |
| `bucket_count()`    | 返回桶的数量 |
| `bucket_size(n)`    | 返回第`n`个桶中的元素个数 |
| `bucket(key)`       | 返回键`key`所在桶的编号 |
| `load_factor()`     | 返回负载因子，即`size()`与`bucket_count()`的比值 |
| `max_load_factor()` | 返回或设置最大负载因子，默认为 1.0，超过后自动增加桶并重新哈希 |
| `rehash(n)`         | 把桶的数量设置为至少`n`，并重新哈希 |
| `reserve(n)`        | 预留至少能容纳`n`个元素而不触发重新哈希的桶 |
| `hash_function()`   | 返回容器使用的哈希函数对象 |

重新哈希会使所有迭代器失效，但指向元素的指针和引用仍然有效。已知元素个数时先`reserve()`可以避免多次重新哈希

### 自定义类型作为键

标准库只为内置类型、`std::string`等提供了`std::hash`，自定义类型作为键时需要提供哈希函数和`==`运算符

```c++
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

struct Point
{
    int x;
    int y;

    bool operator==(const Point &other) const
    {
        return x == other.x && y == other.y;
    }
};

struct PointHash
{
    std::size_t operator()(const Point &p) const noexcept
    {
        return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
    }
};

int main()
{
    std::unordered_map<Point, std::string, PointHash> names;

    names[{1, 2}] = "A";
    names[{3, 4}] = "B";

    std::cout << names.at({1, 2}) << std::endl; // A
    std::cout << names.size() << std::endl;     // 2

    return 0;
}
```
{: file='自定义哈希函数'}

## 容器适配器

容器适配器在已有容器之上只暴露特定的接口，没有迭代器，不能遍历

| 适配器                | 默认底层容器  | 特点                     | 主要成员函数                                 |
| :-------------------- | :------------ | :----------------------- | :------------------------------------------- |
| `std::stack`          | `std::deque`  | 后进先出                 | `push()`<br>`pop()`<br>`top()`               |
| `std::queue`          | `std::deque`  | 先进先出                 | `push()`<br>`pop()`<br>`front()`<br>`back()` |
| `std::priority_queue` | `std::vector` | 每次取出优先级最高的元素 | `push()`<br>`pop()`<br>`top()`               |

`std::stack`定义在`<stack>`头文件中，`std::queue`和`std::priority_queue`定义在`<queue>`头文件中

`pop()`只移除元素，不返回值，需要先用`top()`或`front()`取得元素

```c++
#include <functional>
#include <iostream>
#include <queue>
#include <stack>
#include <vector>

int main()
{
    std::stack<int> stack;
    stack.push(1);
    stack.push(2);
    std::cout << stack.top() << std::endl; // 2

    std::queue<int> queue;
    queue.push(1);
    queue.push(2);
    std::cout << queue.front() << std::endl; // 1

    std::priority_queue<int> max_heap; // 默认为大顶堆
    for (int x : {3, 1, 2})
    {
        max_heap.push(x);
    }
    std::cout << max_heap.top() << std::endl; // 3

    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap; // 小顶堆
    for (int x : {3, 1, 2})
    {
        min_heap.push(x);
    }
    std::cout << min_heap.top() << std::endl; // 1

    return 0;
}
```
{: file='容器适配器的使用'}
