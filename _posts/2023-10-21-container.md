---
title: Container
date: 2023-10-21 21:30:00 +0800
categories: [ Notepad, C++ ]
tags: [ C++ ]
math: true
---

# C++容器相关笔记

容器就是一些模板类的集合，但和普通模板类不同的是，容器中封装的是组织数据的方法（也就是数据结构）。STL
提供有3类标准容器，分别是序列容器、关联容器和哈希容器，其中后两类容器有时也统称为关联容器。

| 容器种类 |                                                                         功能                                                                         |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------------------------:|
| 序列容器 |          包括`std::array`数组容器、`std::vector`向量容器、<br>`std::list`列表容器、`std::deque`双端队列容器<br>元素在容器中的位置同元素的值无关，即容器不是排序的，<br>指定在什么位置，元素就会位于什么位置           |
| 关联容器 |                包括`std::set`集合容器、`std::multiset`多重集合容器、<br>`std::map`映射容器、`std::multimap`多重映射容器<br>元素默认是由小到大排序好的，即便是插入元素，元素也会插入到适当位置                |
| 哈希容器 | 包括`std::unordered_set`哈希集合、`std::unordered_multiset`哈希多重集合、<br>`std::unordered_map` 哈希映射以及 `std::unordered_multimap`哈希多重映射<br>和排序容器不同，元素的位置由哈希函数确定 |

## 序列式容器

序列容器包含以下几类容器：

- `std::array<T,N>`: 可以存储`N`个`T`类型的元素。此类容器一旦建立，其长度就是固定不变的，不能增加或删除元素，只能改变某个元素的值
- `std::vector<T>`: 用来存放`T`类型的元素，是一个长度可变的序列容器，在存储空间不足时，会自动申请更多的内存，在尾部增加或删除元素的效率最高
- `std::deque<T>`: 和`std::vector`相似，使用该容器不仅尾部插入和删除元素高效，在头部插入或删除元素也同样高效
- `std::list<T>`: 是一个长度可变的、由`T`类型元素组成的序列，它以双向链表的形式组织元素，在这个序列的任何地方都可以高效地增加或删除元素
- `std::forward_list<T>`: 和`std::list`类似，但是是以单链表的形式组织元素，内部的元素只能从第一个元素开始访问

### std::array

在使用该容器之前，代码中需引入`<array>`头文件，并使用`std`命令空间

- 初始化

```c++
#include <array>

std::array<double, 10> values; // 具有10个double类型元素的array容器

std::array<double, 10> values {}; // 将所有的元素初始化为0.0

std::array<double, 10> values {0.5,1.0,1.5,,2.0}; // 初始化前4个元素，其余为0.0
```
{: file='初始化'}

- 成员函数

|         成员函数          |                         功能                         |
|:---------------------:|:--------------------------------------------------:|
|       `begin()`       |                返回指向容器中第一个元素的随机访问迭代器                |
|        `end()`        |             返回指向容器最后一个元素之后一个位置的随机访问迭代器             |
|      `rbegin()`       |                 返回指向最后一个元素的随机访问迭代器                 |
|       `rend()`        |              返回指向第一个元素之前一个位置的随机访问迭代器               |
|      `cbegin()`       |          和`begin()`功能相同，其基础上增加了`const`属性           |
|       `cend()`        |           和`end()`功能相同，其基础上增加了`const`属性            |
|      `crbegin()`      |          和`rbegin()`功能相同，其基础上增加了`const`属性          |
|       `crend()`       |           和`rend()`功能相同，其基础上增加了`const`属性           |
|       `size()`        |    返回容器中当前元素的数量<br>其值始终等于初始化`array`类的第二个模板参数`N`    |
|     `max_size()`      |   返回容器可容纳元素的最大数量<br>其值始终等于初始化`array`类的第二个模板参数`N`   |
|       `empty()`       |          判断容器是否为空，和通过`size()==0`的判断条件功能相同          |
|        `at(n)`        | 返回容器中`n`位置处元素的引用<br>如果不在有效的范围内，则抛出`out_of_range`异常 |
|       `front()`       |          返回容器中第一个元素的直接引用，不适用于空的`array`容器           |
|       `back()`        |          返回容器中最后一个元素的直接应用，不适用于空的`array`容器          |
|       `data()`        |                  返回一个指向容器首个元素的指针                   |
|      `fill(val)`      |                 将`val`赋值给容器中的每个元素                  |
| `array1.swap(array2)` |    交换`array1`和`array2`容器中的所有元素<br>前提是具有相同的长度和类型    |

### std::vector

在使用该容器之前，代码中需引入`<vector>`头文件，并使用`std`命令空间

- 初始化

```c++
#include <vector>

std::vector<double> values; // 存储double类型的vector容器

values.reserve(20); // 设置容器的内存分配，至少可以容纳20个元素

std::vector<int> primes {2, 3, 5, 7, 11, 13, 17, 19}; // 含有8个素数的vector容器

std::vector<double> values(20); // 开始时就有20个元素，初始值都为0.0
std::vector<double> values(20, 1.0); // 初始值都为1.0

std::vector<char>value1(5, 'c');
std::vector<char>value2(value1); // 5个字符c

int array[]={1,2,3};
std::vector<int>values(array, array+2); // 保存1,2
std::vector<int>value1{1,2,3,4,5};
std::vector<int>value2(std::begin(value1),std::begin(value1)+3); // 保存1,2,3
```
{: file='初始化'}

- 成员函数

|        成员函数        |                 功能                  |
|:------------------:|:-----------------------------------:|
|     `begin()`      |        返回指向容器中第一个元素的随机访问迭代器         |
|      `end()`       |     返回指向容器最后一个元素之后一个位置的随机访问迭代器      |
|     `rbegin()`     |         返回指向最后一个元素的随机访问迭代器          |
|      `rend()`      |       返回指向第一个元素之前一个位置的随机访问迭代器       |
|     `cbegin()`     |   和`begin()`功能相同，其基础上增加了`const`属性   |
|      `cend()`      |    和`end()`功能相同，其基础上增加了`const`属性    |
|    `crbegin()`     |  和`rbegin()`功能相同，其基础上增加了`const`属性   |
|     `crend()`      |   和`rend()`功能相同，其基础上增加了`const`属性    |
|      `size()`      |              返回实际元素个数               |
|    `max_size()`    |      返回元素个数的最大值，一般是$2^{32}-1$       |
|     `resize()`     |              改变实际元素的个数              |
|    `capacity()`    |               返回当前容量                |
|     `empty()`      |              判断容器是否有元素              |
|    `reserve()`     |               增加容器的容量               |
| `shrink _to_fit()` |        将内存减少到等于当前元素实际所使用的大小         |
|   `operator[ ]`    | 重载`[]`运算符，通过下标即可访问、修改`vector`容器中的元素 |
|       `at()`       |           使用经过边界检查的索引访问元素           |
|     `front()`      |             返回第一个元素的引用              |
|      `back()`      |             返回最后一个元素的引用             |
|      `data()`      |           返回指向容器中第一个元素的指针           |
|     `assign()`     |             用新元素替换原有内容              |
|   `push_back()`    |            在容器的尾部添加一个元素             |
|    `pop_back()`    |              移出容器尾部的元素              |
|     `insert()`     |           在指定的位置插入一个或多个元素           |
|     `erase()`      |             移出一个元素或一段元素             |
|     `clear()`      |           移出所有的元素，容器大小变为0           |
|      `swap()`      |             交换两个容器的所有元素             |
|    `emplace()`     |           在指定的位置直接生成一个元素            |
|  `emplace_back()`  |             在容器尾部生成一个元素             |

### std::deque

在使用该容器之前，代码中需引入`<deque>`头文件，并使用`std`命令空间

- 初始化

```c++
#include <deque>

std::deque<int> d; // 没有任何元素的空deque容器

std::deque<int> d(10); // 具有10个元素，初始值都为0的deque容器

std::deque<int> d(10, 5) // 初始值都为5的deque容器

std::deque<int> d1(5);
std::deque<int> d2(d1); // 必须保证新旧容器存储的元素类型一致

int a[] = { 1,2,3,4,5 };
std::deque<int>d(a, a + 5); // 拷贝普通数组，创建deque容器

std::array<int, 5>arr{ 11,12,13,14,15 };
std::deque<int>d(arr.begin()+2, arr.end()); // 拷贝arr容器中的13,14,15
```
{: file='初始化'}

- 成员函数

|        成员函数        |                功能                |
|:------------------:|:--------------------------------:|
|     `begin()`      |       返回指向容器中第一个元素的随机访问迭代器       |
|      `end()`       |    返回指向容器最后一个元素之后一个位置的随机访问迭代器    |
|     `rbegin()`     |        返回指向最后一个元素的随机访问迭代器        |
|      `rend()`      |     返回指向第一个元素之前一个位置的随机访问迭代器      |
|     `cbegin()`     | 和`begin()`功能相同，其基础上增加了`const`属性  |
|      `cend()`      |  和`end()`功能相同，其基础上增加了`const`属性   |
|    `crbegin()`     | 和`rbegin()`功能相同，其基础上增加了`const`属性 |
|     `crend()`      |  和`rend()`功能相同，其基础上增加了`const`属性  |
|      `size()`      |             返回实际元素个数             |
|    `max_size()`    |     返回元素个数的最大值，一般是$2^{32}-1$     |
|     `resize()`     |            改变实际元素的个数             |
|    `capacity()`    |              返回当前容量              |
|     `empty()`      |            判断容器是否有元素             |
|    `reserve()`     |             增加容器的容量              |
| `shrink _to_fit()` |       将内存减少到等于当前元素实际所使用的大小       |
|   `operator[ ]`    |   重载`[]`运算符，通过下标即可访问、修改容器中的元素    |
|       `at()`       |         使用经过边界检查的索引访问元素          |
|     `front()`      |            返回第一个元素的引用            |
|      `back()`      |           返回最后一个元素的引用            |
|      `data()`      |         返回指向容器中第一个元素的指针          |
|     `assign()`     |            用新元素替换原有内容            |
|   `push_back()`    |           在容器的尾部添加一个元素           |
|    `pop_back()`    |            移出容器尾部的元素             |
|   `pop_front()`    |            移除容器头部的元素             |
|     `insert()`     |         在指定的位置插入一个或多个元素          |
|     `erase()`      |           移出一个元素或一段元素            |
|     `clear()`      |         移出所有的元素，容器大小变为0          |
|      `swap()`      |           交换两个容器的所有元素            |
|    `emplace()`     |          在指定的位置直接生成一个元素          |
| `emplace_front()`  |           在容器头部生成一个元素            |
|  `emplace_back()`  |           在容器尾部生成一个元素            |

### std::list

在使用该容器之前，代码中需引入`<list>`头文件，并使用`std`命令空间

- 初始化

```c++
#include <list>

std::list<int> values; // 没有任何元素的空list容器

std::list<int> values(10); // 包含10个元素，初始值都为0的list容器

std::list<int> values(10, 5); // 初始值都为5的list容器

std::list<int> value1(10);
std::list<int> value2(value1); // 必须保证新旧容器存储的元素类型一致

int a[] = { 1,2,3,4,5 }; // 拷贝普通数组，创建list容器
std::list<int> values(a, a+5);

std::array<int, 5>arr{ 11,12,13,14,15 };
std::list<int>values(arr.begin()+2, arr.end()); // 拷贝arr容器中的13,14,15
```
{: file='初始化'}

- 成员函数

|       成员函数        |                功能                |
|:-----------------:|:--------------------------------:|
|     `begin()`     |       返回指向容器中第一个元素的随机访问迭代器       |
|      `end()`      |    返回指向容器最后一个元素之后一个位置的随机访问迭代器    |
|    `rbegin()`     |        返回指向最后一个元素的随机访问迭代器        |
|     `rend()`      |     返回指向第一个元素之前一个位置的随机访问迭代器      |
|    `cbegin()`     | 和`begin()`功能相同，其基础上增加了`const`属性  |
|     `cend()`      |  和`end()`功能相同，其基础上增加了`const`属性   |
|    `crbegin()`    | 和`rbegin()`功能相同，其基础上增加了`const`属性 |
|     `crend()`     |  和`rend()`功能相同，其基础上增加了`const`属性  |
|     `size()`      |             返回实际元素个数             |
|    `resize()`     |             调整容器的大小              |
|   `max_size()`    |     返回元素个数的最大值，一般是$2^{32}-1$     |
|     `front()`     |            返回第一个元素的引用            |
|     `back()`      |           返回最后一个元素的引用            |
|    `assign()`     |            用新元素替换原有内容            |
|   `push_back()`   |           在容器的尾部添加一个元素           |
|  `push_front()`   |           在容器头部插入一个元素            |
|   `pop_back()`    |            移出容器尾部的元素             |
|   `pop_front()`   |            移除容器头部的元素             |
|    `insert()`     |         在指定的位置插入一个或多个元素          |
|     `erase()`     |           移出一个元素或一段元素            |
|     `clear()`     |         移出所有的元素，容器大小变为0          |
|     `swap()`      |           交换两个容器的所有元素            |
|    `emplace()`    |           在指定的位置插入一个元素           |
| `emplace_front()` |           在容器头部生成一个元素            |
| `emplace_back()`  |           在容器尾部生成一个元素            |
|    `splice()`     |      将一个容器中的元素插入到另一个容器的指定位置      |
|   `remove(val)`   |        删除容器中所有等于`val`的元素         |
|   `remove_if()`   |           删除容器中满足条件的元素           |
|    `unique()`     |        删除容器中相邻的重复元素，只保留一个        |
|     `merge()`     |  合并两个事先已排好序的容器，并且合并之后的容器依然是有序的   |
|     `sort()`      |         更改容器中元素的位置，将它们排序         |
|    `reverse()`    |            反转容器中元素的顺序            |

### std::forward_list

在使用该容器之前，代码中需引入`<forward_list>`头文件，并使用`std`命令空间

- 初始化

```c++
#include <forward_list>

std::forward_list<int> values; // 没有任何元素的空forward_list容器

std::forward_list<int> values(10); // 包含10个元素，初始值都为0的forward_list容器

std::forward_list<int> values(10, 5); // 初始值都为5的forward_list容器

std::forward_list<int> value1(10);
std::forward_list<int> value2(value1); // 必须保证新旧容器存储的元素类型一致


int a[] = { 1,2,3,4,5 };
std::forward_list<int> values(a, a+5); // 拷贝普通数组，创建forward_list容器

std::array<int, 5>arr{ 11,12,13,14,15 };
std::forward_list<int>values(arr.begin()+2, arr.end()); // 拷贝arr容器中的13,14,15
```
{: file='初始化'}

- 成员函数

|       成员函数        |                功能                |
|:-----------------:|:--------------------------------:|
|     `begin()`     |       返回指向容器中第一个元素的随机访问迭代器       |
|      `end()`      |    返回指向容器最后一个元素之后一个位置的随机访问迭代器    |
|    `rbegin()`     |        返回指向最后一个元素的随机访问迭代器        |
|     `rend()`      |     返回指向第一个元素之前一个位置的随机访问迭代器      |
|    `cbegin()`     | 和`begin()`功能相同，其基础上增加了`const`属性  |
|     `cend()`      |  和`end()`功能相同，其基础上增加了`const`属性   |
|    `crbegin()`    | 和`rbegin()`功能相同，其基础上增加了`const`属性 |
|     `crend()`     |  和`rend()`功能相同，其基础上增加了`const`属性  |
|     `size()`      |             返回实际元素个数             |
|    `resize()`     |             调整容器的大小              |
|   `max_size()`    |     返回元素个数的最大值，一般是$2^{32}-1$     |
|     `front()`     |            返回第一个元素的引用            |
|     `back()`      |           返回最后一个元素的引用            |
|    `assign()`     |            用新元素替换原有内容            |
|  `push_front()`   |           在容器头部插入一个元素            |
|   `pop_front()`   |            移除容器头部的元素             |
|    `insert()`     |         在指定的位置插入一个或多个元素          |
|     `erase()`     |           移出一个元素或一段元素            |
|     `clear()`     |         移出所有的元素，容器大小变为0          |
|     `swap()`      |           交换两个容器的所有元素            |
|    `emplace()`    |           在指定的位置插入一个元素           |
| `emplace_front()` |           在容器头部生成一个元素            |
| `emplace_back()`  |           在容器尾部生成一个元素            |
| `splice_after()`  |      将一个容器中的元素插入到另一个容器的指定位置      |
|   `remove(val)`   |         删除容器中所有等于val的元素          |
|   `remove_if()`   |           删除容器中满足条件的元素           |
|    `unique()`     |        删除容器中相邻的重复元素，只保留一个        |
|     `merge()`     |  合并两个事先已排好序的容器，并且合并之后的容器依然是有序的   |
|     `sort()`      |         更改容器中元素的位置，将它们排序         |
|    `reverse()`    |            反转容器中元素的顺序            |

## 关联式容器

关联容器包含以下几类容器：

- `std::map`: 定义在`<map>`
  头文件中，使用该容器存储的数据，其各个元素的键必须是唯一的，该容器会根据各元素键的大小，默认进行升序排序（调用`std::less<T>`）
- `std::set`: 定义在`<set>`
  头文件中，使用该容器存储的数据，各个元素键和值完全相同，且各个元素的值不能重复。该容器会自动根据各个元素的键的大小进行升序排序（调用`std::less<T>`）
- `std::multimap`: 定义在`<map>`头文件中，和`map`容器唯一的不同在于，`multimap`容器中存储元素的键可以重复
- `std::multiset`: 定义在`<set>`头文件中，和`set`容器唯一的不同在于，`multiset`容器中存储元素的值可以重复

### std::map

在使用该容器之前，代码中需引入`<map>`头文件，并使用`std`命令空间

```c++
#include <map>

std::map<std::string, int> myMap; // 没有存储任何键值对

std::map<std::string, int> myMap{ {"C++",10},{"Container",20} }; // 初始状态包含2个键值对

std::map<std::string, int> myMap{
                                  std::make_pair("C++",10),
                                  std::make_pair("Container",20)
                                };
std::map<std::string, int> newMap(myMap); // 容器的类型必须完全一致

std::map<std::string, int> myMap{ {"C++",10},{"Container",20} };
std::map<std::string, int> newMap(++myMap.begin(), 
                                  myMap.end()); // 包含{"Container",20}键值对
```
{: file='初始化'}

- 成员函数

|        成员函数        |                                              功能                                               |
|:------------------:|:---------------------------------------------------------------------------------------------:|
|     `begin()`      |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|      `end()`       |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|     `rbegin()`     |               返回指向容器中排好序的最后一个元素的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器               |
|      `rend()`      |            返回指向容器中排好序的第一个元素的前一个位置的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器             |
|     `cbegin()`     |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|      `cend()`      |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|    `crbegin()`     |                        和`rbegin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                        |
|     `crend()`      |                         和`rend()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|    `find(key)`     |   返回指向当前容器中等于`key`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
| `lower_bound(key)` |             返回指向当前容器中第一个大于或等于`key`的键值对的双向迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器              |
| `upper_bound(key)` |                返回指向当前容器中第一个大于`key`的键值对的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器                |
| `equal_range(key)` | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|     `empty()`      |                                           检查容器是否为空                                            |
|      `size()`      |                                        返回当前容器中存有键值对的个数                                        |
|    `max_size()`    |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|  `operator[key]`   |                         重载了`[]`运算符<br>知道容器中某个键值对的键`key`，可以通过键直接获取对应的值                         |
|     `at(key)`      |                            找到容器中`key`键对应的值，找不到会引发`out_of_range`异常                             |
|     `insert()`     |                                           向容器中插入键值对                                           |
|     `erase()`      |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|      `swap()`      |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|     `clear()`      |                                          清空容器中所有的键值对                                          |
|    `emplace()`     |                                      在当前容器中的指定位置处构造新键值对                                       |
|  `emplace_hint()`  |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|    `count(key)`    |                             在当前容器中，查找键为`key`的键值对的个数并返回<br>返回值最大为1                             |

### std::multimap

在使用该容器之前，代码中需引入`<map>`头文件，并使用`std`命令空间

```c++
#include <map>

std::multimap<std::string, std::string> mymultimap; // 没有存储任何键值对

std::multimap<std::string, std::string> mymultimap{ 
                                                      {"C++","10"},
                                                      {"Container","20"} 
                                                  }; // 初始状态包含2个键值对

std::multimap<std::string, std::string> mymultimap{
                                                      std::make_pair("C++","10"),
                                                      std::make_pair("Container","20")
                                                  };
std::multimap<std::string, std::string> newmultimap(mymultimap); // 容器的类型必须完全一致

std::multimap<std::string, std::string> mymultimap{ {"C++",10},{"Container",20} };
std::multimap<std::string, std::string> newmultimap(
                                                        ++mymultimap.begin(), 
                                                        mymultimap.end()
                                                   ); // 包含{"Container","20"}键值对
```
{: file='初始化'}

- 成员函数

|        成员函数        |                                              功能                                               |
|:------------------:|:---------------------------------------------------------------------------------------------:|
|     `begin()`      |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|      `end()`       |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|     `rbegin()`     |               返回指向容器中排好序的最后一个元素的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器               |
|      `rend()`      |            返回指向容器中排好序的第一个元素的前一个位置的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器             |
|     `cbegin()`     |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|      `cend()`      |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|    `crbegin()`     |                        和`rbegin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                        |
|     `crend()`      |                         和`rend()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|    `find(key)`     |   返回指向当前容器中等于`key`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
| `lower_bound(key)` |             返回指向当前容器中第一个大于或等于`key`的键值对的双向迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器              |
| `upper_bound(key)` |                返回指向当前容器中第一个大于`key`的键值对的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器                |
| `equal_range(key)` | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|     `empty()`      |                                           检查容器是否为空                                            |
|      `size()`      |                                        返回当前容器中存有键值对的个数                                        |
|    `max_size()`    |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|     `insert()`     |                                           向容器中插入键值对                                           |
|     `erase()`      |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|      `swap()`      |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|     `clear()`      |                                          清空容器中所有的键值对                                          |
|    `emplace()`     |                                      在当前容器中的指定位置处构造新键值对                                       |
|  `emplace_hint()`  |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|    `count(key)`    |                             在当前容器中，查找键为`key`的键值对的个数并返回<br>返回值最大为1                             |

### std::set

在使用该容器之前，代码中需引入`<set>`头文件，并使用`std`命令空间

```c++
#include <set>

std::set<std::string> myset; // 空的set容器

std::set<std::string> myset{ "C++","Container"}; // 包含2个元素的set容器

std::set<std::string> copyset(myset); // 容器的类型必须完全一致

std::set<std::string> copyset(++myset.begin(), myset.end()); // 包含"Container"键值对
```
{: file='初始化'}

- 成员函数

|        成员函数        |                                              功能                                               |
|:------------------:|:---------------------------------------------------------------------------------------------:|
|     `begin()`      |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|      `end()`       |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|     `rbegin()`     |               返回指向容器中排好序的最后一个元素的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器               |
|      `rend()`      |            返回指向容器中排好序的第一个元素的前一个位置的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器             |
|     `cbegin()`     |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|      `cend()`      |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|    `crbegin()`     |                        和`rbegin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                        |
|     `crend()`      |                         和`rend()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|    `find(val)`     |   返回指向当前容器中等于`val`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
| `lower_bound(val)` |             返回指向当前容器中第一个大于或等于`val`的键值对的双向迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器              |
| `upper_bound(val)` |                返回指向当前容器中第一个大于`val`的键值对的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器                |
| `equal_range(val)` | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|     `empty()`      |                                           检查容器是否为空                                            |
|      `size()`      |                                        返回当前容器中存有键值对的个数                                        |
|    `max_size()`    |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|     `insert()`     |                                           向容器中插入键值对                                           |
|     `erase()`      |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|      `swap()`      |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|     `clear()`      |                                          清空容器中所有的键值对                                          |
|    `emplace()`     |                                      在当前容器中的指定位置处构造新键值对                                       |
|  `emplace_hint()`  |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|    `count(val)`    |                             在当前容器中，查找键为`val`的键值对的个数并返回<br>返回值最大为1                             |

### std::multiset

在使用该容器之前，代码中需引入`<set>`头文件，并使用`std`命令空间

```c++
#include <set>

std::multiset<std::string> mymultiset; // 空的multiset容器

std::multiset<std::string> mymultiset{ "C++","Container"}; // 包含2个元素的multiset容器

std::multiset<std::string> copymultiset(mymultiset); // 容器的类型必须完全一致

std::multiset<std::string> copymultiset(++mymultiset.begin(), 
                                        mymultiset.end()); // 包含"Container"键值对
```
{: file='初始化'}

- 成员函数

|        成员函数        |                                              功能                                               |
|:------------------:|:---------------------------------------------------------------------------------------------:|
|     `begin()`      |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|      `end()`       |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|     `rbegin()`     |               返回指向容器中排好序的最后一个元素的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器               |
|      `rend()`      |            返回指向容器中排好序的第一个元素的前一个位置的反向双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的反向双向迭代器             |
|     `cbegin()`     |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|      `cend()`      |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|    `crbegin()`     |                        和`rbegin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                        |
|     `crend()`      |                         和`rend()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|    `find(val)`     |   返回指向当前容器中等于`val`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
| `lower_bound(val)` |             返回指向当前容器中第一个大于或等于`val`的键值对的双向迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器              |
| `upper_bound(val)` |                返回指向当前容器中第一个大于`val`的键值对的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器                |
| `equal_range(val)` | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|     `empty()`      |                                           检查容器是否为空                                            |
|      `size()`      |                                        返回当前容器中存有键值对的个数                                        |
|    `max_size()`    |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|     `insert()`     |                                           向容器中插入键值对                                           |
|     `erase()`      |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|      `swap()`      |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|     `clear()`      |                                          清空容器中所有的键值对                                          |
|    `emplace()`     |                                      在当前容器中的指定位置处构造新键值对                                       |
|  `emplace_hint()`  |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|    `count(val)`    |                             在当前容器中，查找键为`val`的键值对的个数并返回<br>返回值最大为1                             |

## 无序容器

无序容器包含以下几类容器：

- `std::unordered_map`: 定义在`<unordered_map>`头文件中，使用该容器存储的数据，其各个元素的键必须是唯一的，且该容器中存储的键值对是无序的
- `std::unordered_multimap`: 定义在`<unordered_map>`头文件中，和`unordered_map`容器唯一的不同在于，`unordered_map`
  容器中存储元素的键可以重复
- `std::unordered_set`: 定义在`<unordered_set>`头文件中，不再以键值对的形式存储数据，而是直接存储数据元素本身
- `std::unordered_multiset`: 定义在`<unordered_set>`头文件中，和`unordered_set`容器唯一的不同在于，`unordered_multiset`
  容器中存储元素的值可以重复

### std::unordered_map

在使用该容器之前，代码中需引入`<unordered_map>`头文件，并使用`std`命令空间

```c++
#include <unordered_map>

std::unordered_map<std::string, std::string> umap; // 没有存储任何键值对

std::unordered_map<std::string, std::string> umap{ 
                                                    {"C++", "10"},
                                                    {"Container", "20"} 
                                                 }; // 初始状态包含2个键值对

std::unordered_map<std::string, std::string> umap{std::make_pair("C++","10"),
                                std::make_pair("Container", "20")};
std::unordered_map<std::string, std::string> umap2(umap); // 容器的类型必须完全一致

std::unordered_map<std::string, std::string> umap2(
                                                    ++umap.begin(), 
                                                    umap.end()
                                                  ); // 包含{"Container",20}键值对
```
{: file='初始化'}

- 成员函数

|         成员函数         |                                              功能                                               |
|:--------------------:|:---------------------------------------------------------------------------------------------:|
|      `begin()`       |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|       `end()`        |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|      `cbegin()`      |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|       `cend()`       |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|     `find(key)`      |   返回指向当前容器中等于`key`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
|  `equal_range(key)`  | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|      `empty()`       |                                           检查容器是否为空                                            |
|       `size()`       |                                        返回当前容器中存有键值对的个数                                        |
|     `max_size()`     |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|   `operator[key]`    |                         重载了`[]`运算符<br>知道容器中某个键值对的键`key`，可以通过键直接获取对应的值                         |
|      `at(key)`       |                            找到容器中`key`键对应的值，找不到会引发`out_of_range`异常                             |
|      `insert()`      |                                           向容器中插入键值对                                           |
|      `erase()`       |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|       `swap()`       |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|      `clear()`       |                                          清空容器中所有的键值对                                          |
|     `emplace()`      |                                      在当前容器中的指定位置处构造新键值对                                       |
|   `emplace_hint()`   |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|     `count(key)`     |                             在当前容器中，查找键为`key`的键值对的个数并返回<br>返回值最大为1                             |
|   `bucket_count()`   |                                  返回当前容器底层存储键值对时，使用桶（线性链表）的数量                                  |
| `max_bucket_count()` |                                     返回当前系统中，容器底层最多可以用多少桶                                      |
|   `bucket_size(n)`   |                                       返回第`n`个桶中存储键值对的数量                                       |
|    `bucket(key)`     |                                     返回以`key`为键的键值对所在桶的编号                                      |
|   `load_factor()`    |                   返回容器中当前的负载因子<br>即存储键值对的数量（size()）和使用桶数（bucket_count()）的比值                   |
| `max_load_factor()`  |                                        返回或者设置当前容器的负载因子                                        |
|     `rehash(n)`      |                                      将当前容器底层使用桶的数量设置为`n`                                      |
|     `reserve()`      |              将存储桶的数量（bucket_count()）设置为至少容纳`count`个元<br>（不超过最大负载因子）所需的数量，并重新整理容器              |
|  `hash_function()`   |                                        返回当前容器使用的哈希函数对象                                        |

### std::unordered_multimap

在使用该容器之前，代码中需引入`<unordered_map>`头文件，并使用`std`命令空间

```c++
#include <unordered_map>

std::unordered_multimap<std::string, std::string> myummap; // 没有存储任何键值对

std::unordered_multimap<std::string, std::string> myummap{ 
                                                            {"C++", "10"},
                                                            {"Container", "20"} 
                                                         }; // 初始状态包含2个键值对

std::unordered_multimap<std::string, std::string> myummap{std::make_pair("C++","10"),
                                std::make_pair("Container", "20")};
std::unordered_multimap<std::string, std::string>
                        myummap2(myummap); // 容器的类型必须完全一致

std::unordered_multimap<std::string, std::string> 
                        myummap2(++myummap.begin(), 
                        myummap.end()); // 包含{"Container",20}键值对
```
{: file='初始化'}

- 成员函数

|         成员函数         |                                              功能                                               |
|:--------------------:|:---------------------------------------------------------------------------------------------:|
|      `begin()`       |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|       `end()`        |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|      `cbegin()`      |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|       `cend()`       |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|     `find(key)`      |   返回指向当前容器中等于`key`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
|  `equal_range(key)`  | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|      `empty()`       |                                           检查容器是否为空                                            |
|       `size()`       |                                        返回当前容器中存有键值对的个数                                        |
|     `max_size()`     |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|      `insert()`      |                                           向容器中插入键值对                                           |
|      `erase()`       |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|       `swap()`       |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|      `clear()`       |                                          清空容器中所有的键值对                                          |
|     `emplace()`      |                                      在当前容器中的指定位置处构造新键值对                                       |
|   `emplace_hint()`   |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|     `count(key)`     |                             在当前容器中，查找键为`key`的键值对的个数并返回<br>返回值最大为1                             |
|   `bucket_count()`   |                                  返回当前容器底层存储键值对时，使用桶（线性链表）的数量                                  |
| `max_bucket_count()` |                                     返回当前系统中，容器底层最多可以用多少桶                                      |
|   `bucket_size(n)`   |                                       返回第`n`个桶中存储键值对的数量                                       |
|    `bucket(key)`     |                                     返回以`key`为键的键值对所在桶的编号                                      |
|   `load_factor()`    |                   返回容器中当前的负载因子<br>即存储键值对的数量（size()）和使用桶数（bucket_count()）的比值                   |
| `max_load_factor()`  |                                        返回或者设置当前容器的负载因子                                        |
|     `rehash(n)`      |                                      将当前容器底层使用桶的数量设置为`n`                                      |
|     `reserve()`      |              将存储桶的数量（bucket_count()）设置为至少容纳`count`个元<br>（不超过最大负载因子）所需的数量，并重新整理容器              |
|  `hash_function()`   |                                        返回当前容器使用的哈希函数对象                                        |

### std::unordered_set

在使用该容器之前，代码中需引入`<unordered_set>`头文件，并使用`std`命令空间

```c++
#include <unordered_set>

std::unordered_set<std::string> uset; // 空的unordered_set容器

std::unordered_set<std::string> uset{ "C++", "Container"}; // 初始状态包含2个键值对

std::unordered_set<std::string> uset2(uset); // 容器的类型必须完全一致

std::unordered_set<std::string> uset2(++uset.begin(), uset.end()); // 包含"Container"
```
{: file='初始化'}

- 成员函数

|         成员函数         |                                              功能                                               |
|:--------------------:|:---------------------------------------------------------------------------------------------:|
|      `begin()`       |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|       `end()`        |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|      `cbegin()`      |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|       `cend()`       |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|     `find(key)`      |   返回指向当前容器中等于`key`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
|  `equal_range(key)`  | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|      `empty()`       |                                           检查容器是否为空                                            |
|       `size()`       |                                        返回当前容器中存有键值对的个数                                        |
|     `max_size()`     |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|      `insert()`      |                                           向容器中插入键值对                                           |
|      `erase()`       |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|       `swap()`       |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|      `clear()`       |                                          清空容器中所有的键值对                                          |
|     `emplace()`      |                                      在当前容器中的指定位置处构造新键值对                                       |
|   `emplace_hint()`   |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|     `count(key)`     |                             在当前容器中，查找键为`key`的键值对的个数并返回<br>返回值最大为1                             |
|   `bucket_count()`   |                                  返回当前容器底层存储键值对时，使用桶（线性链表）的数量                                  |
| `max_bucket_count()` |                                     返回当前系统中，容器底层最多可以用多少桶                                      |
|   `bucket_size(n)`   |                                       返回第`n`个桶中存储键值对的数量                                       |
|    `bucket(key)`     |                                     返回以`key`为键的键值对所在桶的编号                                      |
|   `load_factor()`    |                   返回容器中当前的负载因子<br>即存储键值对的数量（size()）和使用桶数（bucket_count()）的比值                   |
| `max_load_factor()`  |                                        返回或者设置当前容器的负载因子                                        |
|     `rehash(n)`      |                                      将当前容器底层使用桶的数量设置为`n`                                      |
|     `reserve()`      |              将存储桶的数量（bucket_count()）设置为至少容纳`count`个元<br>（不超过最大负载因子）所需的数量，并重新整理容器              |
|  `hash_function()`   |                                        返回当前容器使用的哈希函数对象                                        |

### std::unordered_multiset

在使用该容器之前，代码中需引入`<unordered_set>`头文件，并使用`std`命令空间

```c++
#include <unordered_set>

std::unordered_multiset<std::string> uset; // 空的unordered_multiset容器

std::unordered_multiset<std::string> uset{ "C++", "Container"}; // 初始状态包含2个键值对

std::unordered_multiset<std::string> uset2(uset); // 容器的类型必须完全一致

std::unordered_multiset<std::string> uset2(++uset.begin(), 
                                           uset.end()); // 包含"Container"
```
{: file='初始化'}

- 成员函数

|         成员函数         |                                              功能                                               |
|:--------------------:|:---------------------------------------------------------------------------------------------:|
|      `begin()`       |                 返回指向容器中排好序的第一个键值对的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器                 |
|       `end()`        |              返回指向容器中排好序的最后一个元素的后一个位置的双向迭代器<br>如果容器用`const`限定，返回的是`const`类型的双向迭代器              |
|      `cbegin()`      |                        和`begin()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                         |
|       `cend()`       |                         和`end()`功能相同<br>增加了`const`属性，不能用于修改容器内存储的键值对                          |
|     `find(key)`      |   返回指向当前容器中等于`key`的键值对的双向迭代器<br>如果没找到，返回和`end()`一样的迭代器<br>如果容器用`const`限定，返回`const`类型的双向迭代器    |
|  `equal_range(key)`  | 返回一个`pair`对象<br>`pair.first`和`lower_bound()`方法的返回值等价<br>`pair.second`和`upper_bound()`方法的返回值等价 |
|      `empty()`       |                                           检查容器是否为空                                            |
|       `size()`       |                                        返回当前容器中存有键值对的个数                                        |
|     `max_size()`     |                              返回容器所能容纳键值对的最大个数<br>不同的操作系统，其返回值不同                               |
|      `insert()`      |                                           向容器中插入键值对                                           |
|      `erase()`       |                                   删除容器指定位置、指定键值或者指定区域内的键值对                                    |
|       `swap()`       |                                  交换2个容器中存储的键值对，2个键值对的类型必须相同                                   |
|      `clear()`       |                                          清空容器中所有的键值对                                          |
|     `emplace()`      |                                      在当前容器中的指定位置处构造新键值对                                       |
|   `emplace_hint()`   |                   和`emplace()`构造新键值对的方式一样<br>但必须提供一个指示键值对生成位置的迭代器，并作为第一个参数                    |
|     `count(key)`     |                             在当前容器中，查找键为`key`的键值对的个数并返回<br>返回值最大为1                             |
|   `bucket_count()`   |                                  返回当前容器底层存储键值对时，使用桶（线性链表）的数量                                  |
| `max_bucket_count()` |                                     返回当前系统中，容器底层最多可以用多少桶                                      |
|   `bucket_size(n)`   |                                       返回第`n`个桶中存储键值对的数量                                       |
|    `bucket(key)`     |                                     返回以`key`为键的键值对所在桶的编号                                      |
|   `load_factor()`    |                   返回容器中当前的负载因子<br>即存储键值对的数量（size()）和使用桶数（bucket_count()）的比值                   |
| `max_load_factor()`  |                                        返回或者设置当前容器的负载因子                                        |
|     `rehash(n)`      |                                      将当前容器底层使用桶的数量设置为`n`                                      |
|     `reserve()`      |              将存储桶的数量（bucket_count()）设置为至少容纳`count`个元<br>（不超过最大负载因子）所需的数量，并重新整理容器              |
|  `hash_function()`   |                                        返回当前容器使用的哈希函数对象                                        |
