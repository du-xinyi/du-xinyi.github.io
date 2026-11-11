---
title: ZeroMQ Messaging
description: 介绍 ZeroMQ 的安装与源码编译、通信模式、传输协议、CURVE 加密认证，以及 C++ 与 Python 下的用法
date: 2025-05-17 12:30:00 +0800
categories: [ Development, Distributed Systems ]
tags: [ ZeroMQ, C++, Python ]
---

`ZeroMQ`是一个高性能的异步消息库，提供套接字风格的 API，但收发的是完整的消息而不是字节流，并且内置了消息队列、自动重连和多种通信模式，适用于构建分布式、并发和多线程系统

> ZeroMQ = Socket + Queue + Threading + 异步通信能力，但无中心服务器、轻量级、极快

## 安装

```bash
sudo apt install -y libzmq3-dev
```
{: file='C/C++（libzmq）'}

```bash
pip install pyzmq
```
{: file='Python'}

`libzmq`提供的是 C 接口，头文件为`zmq.h`，C 和 C++ 程序都可以直接使用，本文的 C++ 示例均直接调用该接口。编译时需要链接`libzmq`

```bash
g++ server.cpp -o server -lzmq
```

### 源码编译

需要比`apt`仓库更新的版本，或者要自行控制编译选项时，可以从源码编译。`pyzmq`的安装包自带了`libzmq`，不受系统中版本的影响，不需要另外处理

```bash
sudo apt install -y build-essential cmake git pkg-config libsodium-dev
```
{: file='安装编译依赖'}

```bash
git clone https://github.com/zeromq/libzmq.git
cd libzmq
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DWITH_LIBSODIUM=ON -DENABLE_CURVE=ON \
    -DBUILD_TESTS=OFF -DWITH_DOCS=OFF
cmake --build build -j$(nproc)
sudo cmake --install build
sudo ldconfig
```
{: file='编译安装 libzmq'}

| 选项                      | 作用                             |
| :------------------------ | :------------------------------- |
| `-DWITH_LIBSODIUM=ON`     | 使用`libsodium`提供加密算法      |
| `-DENABLE_CURVE=ON`       | 启用 CURVE 加密认证，依赖上一项  |
| `-DBUILD_TESTS=OFF`       | 不编译测试程序，缩短编译时间     |
| `-DWITH_DOCS=OFF`         | 不生成文档                       |
| `-DBUILD_SHARED=OFF`      | 不生成动态库，只需要静态库时使用 |
| `-DCMAKE_INSTALL_PREFIX=` | 指定安装位置，默认为`/usr/local` |

**`WITH_LIBSODIUM`和`ENABLE_CURVE`默认是关闭的，不加这两个选项编译出的库不支持 CURVE**，设置相关选项时会报错

安装完成后，确认实际使用的版本以及是否启用了 CURVE

```c++
#include <iostream>
#include <zmq.h>

int main()
{
    int major, minor, patch;
    zmq_version(&major, &minor, &patch);

    std::cout << "version: " << major << "." << minor << "." << patch << std::endl;
    std::cout << "curve: " << zmq_has("curve") << std::endl; // 1表示已启用

    return 0;
}
```
{: file='check.cpp'}

```bash
g++ check.cpp -o check -lzmq && ./check
```
{: file='编译并运行'}

`/usr/local`中的头文件和库会优先于`apt`安装的版本被使用。两者不要同时安装，否则容易出现头文件与库版本不一致的问题，可先执行`sudo apt remove libzmq3-dev`

使用 CMake 的工程可以通过`pkg-config`引入，这种写法对`apt`安装和源码编译的`libzmq`都适用

```cmake
cmake_minimum_required(VERSION 3.16)
project(demo CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)

add_executable(server server.cpp)
target_link_libraries(server PRIVATE PkgConfig::ZMQ)
```
{: file='CMakeLists.txt'}

## 通信模式

| 模式          | 描述                       | 特点                                               |
| :------------ | :------------------------- | :------------------------------------------------- |
| REQ/REP       | 请求-响应（客户端/服务器） | 必须严格按"一发一收"交替进行                       |
| PUB/SUB       | 发布-订阅（广播）          | 单向，订阅端按消息前缀过滤，没有订阅者时消息被丢弃 |
| PUSH/PULL     | 任务分发（流水线）         | 单向，消息在多个接收端之间轮询分配                 |
| PAIR          | 一对一通信                 | 只连接一个对端，常用于线程间通信                   |
| ROUTER/DEALER | 异步的请求-响应            | 不要求收发交替，用于代理、负载均衡等复杂拓扑       |

![请求-响应、发布-订阅、任务分发三种通信模式的消息流向](posts/2025-05-17-zeromq-messaging/patterns.png)

## 传输协议

| 协议        | 说明 |
| :---------- | :--- |
| `tcp://`    | 跨主机通信，如`tcp://192.168.1.10:5555`，绑定所有网卡写作`tcp://*:5555` |
| `ipc://`    | 同一台主机的进程间通信，基于 Unix 域套接字，如`ipc:///tmp/demo.sock` |
| `inproc://` | 同一进程的线程间通信，速度最快，两端必须使用同一个`context` |

同一个套接字可以同时`bind`或`connect`多个地址。通常由位置固定的一端`bind`，数量可变的一端`connect`，与谁先启动无关，连接断开后会自动重连

## 请求-响应

服务端使用 C++，客户端使用 Python，两种语言之间可以直接通信

```c++
#include <iostream>
#include <string>
#include <zmq.h>

int main()
{
    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REP);

    if (zmq_bind(socket, "tcp://*:5555") != 0)
    {
        std::cerr << "bind失败: " << zmq_strerror(zmq_errno()) << std::endl;
        return 1;
    }

    while (true)
    {
        zmq_msg_t request;
        zmq_msg_init(&request);
        if (zmq_msg_recv(&request, socket, 0) < 0)
        {
            zmq_msg_close(&request);
            break;
        }

        std::string text(static_cast<char *>(zmq_msg_data(&request)), zmq_msg_size(&request));
        zmq_msg_close(&request);
        std::cout << "收到: " << text << std::endl;

        zmq_send(socket, "World", 5, 0);
    }

    zmq_close(socket);
    zmq_ctx_term(context);

    return 0;
}
```
{: file='server.cpp'}

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5555")

for i in range(3):
    socket.send_string("Hello")
    print("收到:", socket.recv_string())
```
{: file='client.py'}

C 接口的函数出错时返回`-1`，错误原因通过`zmq_errno()`和`zmq_strerror()`获取。接收消息有两种方式：`zmq_msg_recv()`由库分配内存，可以接收任意长度的消息，用完后需要`zmq_msg_close()`；`zmq_recv()`写入调用者提供的缓冲区，消息超出缓冲区时会被截断，适合长度已知的短消息

`REQ`必须先发后收，`REP`必须先收后发，顺序错误时 Python 会抛出异常，C 接口返回`-1`并把错误码设为`EFSM`。服务端没有回复时`REQ`会一直阻塞，实际使用中应设置接收超时

```python
socket.setsockopt(zmq.RCVTIMEO, 1000)  # 接收超时1000毫秒，超时抛出zmq.Again
socket.setsockopt(zmq.LINGER, 0)       # 关闭时不等待未发出的消息
```
{: file='设置超时'}

## 发布-订阅

```python
import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

count = 0
while True:
    socket.send_string(f"status {count}")
    count += 1
    time.sleep(0.5)
```
{: file='publisher.py'}

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, "status")  # 只接收以status开头的消息

while True:
    print(socket.recv_string())
```
{: file='subscriber.py'}

- `SUB`必须设置订阅前缀，否则收不到任何消息，订阅全部消息使用空字符串`""`
- 订阅端建立连接需要时间，连接完成之前发布的消息会丢失，不能假定能收到第一条消息
- 订阅端处理过慢时，消息在队列中堆积到高水位（`SNDHWM`/`RCVHWM`，默认 1000 条）后会被丢弃

## 任务分发

`PUSH`端把任务轮询分配给所有已连接的`PULL`端，每条消息只会被一个接收端处理

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5557")

for i in range(10):
    socket.send_json({"task": i})
```
{: file='producer.py'}

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://127.0.0.1:5557")

while True:
    print(socket.recv_json())
```
{: file='worker.py'}

与`PUB`不同，`PUSH`在没有接收端或队列写满时会阻塞，不会丢弃消息

## 同时监听多个套接字

`recv`会一直阻塞，一个线程需要同时处理多个套接字时使用`Poller`，哪个套接字有消息就处理哪个

```python
import zmq

context = zmq.Context()

subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://127.0.0.1:5556")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

receiver = context.socket(zmq.PULL)
receiver.connect("tcp://127.0.0.1:5557")

poller = zmq.Poller()
poller.register(subscriber, zmq.POLLIN)
poller.register(receiver, zmq.POLLIN)

while True:
    events = dict(poller.poll(timeout=1000))  # 最多等待1000毫秒

    if subscriber in events:
        print("订阅:", subscriber.recv_string())
    if receiver in events:
        print("任务:", receiver.recv_json())
    if not events:
        print("超时，没有消息")
```
{: file='poller.py'}

## 多客户端与多服务端

`REQ/REP`只能一问一答，一个`REP`同一时间只能处理一个请求。在中间加一个由`ROUTER`和`DEALER`组成的代理，客户端和服务端都连接到代理，就可以让多个服务端并行处理多个客户端的请求，并且两边都可以随时增减

- `ROUTER`面向客户端，收到请求时会记下来源，回复时据此送回对应的客户端
- `DEALER`面向服务端，把请求轮询分配给各个服务端

![多个客户端通过 ROUTER/DEALER 代理连接多个服务端](posts/2025-05-17-zeromq-messaging/broker.png)

```python
import zmq

context = zmq.Context()

frontend = context.socket(zmq.ROUTER)
frontend.bind("tcp://*:5559")

backend = context.socket(zmq.DEALER)
backend.bind("tcp://*:5560")

zmq.proxy(frontend, backend)  # 在两个套接字之间转发消息，一直阻塞
```
{: file='broker.py'}

```python
import os
import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect("tcp://127.0.0.1:5560")  # 服务端也使用connect

while True:
    request = socket.recv_string()
    socket.send_string(f"{request} handled by {os.getpid()}")
```
{: file='worker.py'}

```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5559")

for i in range(4):
    socket.send_string(f"request {i}")
    print(socket.recv_string())
```
{: file='client.py'}

启动一个`broker.py`和多个`worker.py`后运行`client.py`，可以看到请求被不同的进程轮流处理

## CURVE 加密与认证

ZeroMQ 默认以明文传输，也不验证对端身份，任何能访问端口的程序都可以连接。CURVE 是内置的安全机制，基于椭圆曲线加密，同时提供加密和认证

- 每一端都有一对密钥：公钥可以公开，私钥只能自己保存。密钥用 Z85 编码表示，长度为 40 个字符
- 客户端必须事先拿到服务端的公钥，连接时用它加密并确认对方确实持有对应的私钥，因此无法被冒充的服务端欺骗
- 服务端在握手时会得到客户端的公钥，可以据此判断是否允许连接，这一步称为 ZAP（ZeroMQ Authentication Protocol）认证

![CURVE 握手与 ZAP 认证的流程](posts/2025-05-17-zeromq-messaging/curve.png)

| 套接字选项            | 服务端     | 客户端     |
| :-------------------- | :--------- | :--------- |
| `ZMQ_CURVE_SERVER`    | 设为`1`    | 不设置     |
| `ZMQ_CURVE_SECRETKEY` | 服务端私钥 | 客户端私钥 |
| `ZMQ_CURVE_PUBLICKEY` | 可选       | 客户端公钥 |
| `ZMQ_CURVE_SERVERKEY` | 不设置     | 服务端公钥 |

这些选项必须在`bind`或`connect`之前设置。`pyzmq`中对应的是套接字的`curve_server`、`curve_secretkey`等同名小写属性

### Python

`pyzmq`提供了生成证书文件和进行认证的工具

```python
import os
import shutil

import zmq.auth

os.makedirs("certs/authorized", exist_ok=True)

# 每次调用生成一对文件：name.key（公钥）和name.key_secret（公钥+私钥）
zmq.auth.create_certificates("certs", "server")
zmq.auth.create_certificates("certs", "client")

# 服务端只信任authorized目录中的公钥
shutil.copy("certs/client.key", "certs/authorized/")
```
{: file='keygen.py'}

```python
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

context = zmq.Context()

# 认证线程：只允许公钥在certs/authorized目录中的客户端连接
auth = ThreadAuthenticator(context)
auth.start()
auth.configure_curve(domain="*", location="certs/authorized")

server_public, server_secret = zmq.auth.load_certificate("certs/server.key_secret")

socket = context.socket(zmq.REP)
socket.curve_publickey = server_public
socket.curve_secretkey = server_secret
socket.curve_server = True
socket.bind("tcp://*:5555")

while True:
    print("收到:", socket.recv_string())
    socket.send_string("World")
```
{: file='server.py'}

```python
import zmq
import zmq.auth

context = zmq.Context()

client_public, client_secret = zmq.auth.load_certificate("certs/client.key_secret")
server_public, _ = zmq.auth.load_certificate("certs/server.key")

socket = context.socket(zmq.REQ)
socket.curve_publickey = client_public
socket.curve_secretkey = client_secret
socket.curve_serverkey = server_public
socket.setsockopt(zmq.RCVTIMEO, 3000)
socket.setsockopt(zmq.LINGER, 0)
socket.connect("tcp://127.0.0.1:5555")

socket.send_string("Hello")
print("收到:", socket.recv_string())
```
{: file='client.py'}

先运行一次`keygen.py`生成证书，再分别启动`server.py`和`client.py`。把`certs/authorized/client.key`删除后重启服务端，客户端的握手会被拒绝，表现为一直收不到回复，直到超时抛出`zmq.Again`

不需要验证客户端身份、只想加密时，把`location`设为`zmq.auth.CURVE_ALLOW_ANY`，此时任何持有服务端公钥的客户端都可以连接

### C++

`libzmq`只负责握手和加密，不自带认证逻辑：握手时它会把客户端的公钥发送到固定地址`inproc://zeromq.zap.01`，需要自己在这个地址上启动一个`REP`套接字来回答是否放行。下面的示例把服务端、客户端和认证线程放在同一个程序中

```c++
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <zmq.h>

// 接收一条完整的多帧消息，出错时返回空
std::vector<std::string> recv_frames(void *socket)
{
    std::vector<std::string> frames;
    int more = 1;
    while (more)
    {
        zmq_msg_t frame;
        zmq_msg_init(&frame);
        if (zmq_msg_recv(&frame, socket, 0) < 0)
        {
            zmq_msg_close(&frame);
            return {};
        }
        frames.emplace_back(static_cast<char *>(zmq_msg_data(&frame)), zmq_msg_size(&frame));
        more = zmq_msg_more(&frame);
        zmq_msg_close(&frame);
    }
    return frames;
}

// 认证线程：握手时libzmq把客户端的公钥发到这里，由它决定是否放行
void zap_loop(void *zap, std::set<std::string> allowed_keys)
{
    while (true)
    {
        // 请求帧：版本、请求序号、域、地址、身份、机制、凭据（CURVE为32字节的客户端公钥）
        std::vector<std::string> request = recv_frames(zap);
        if (request.size() < 2)
        {
            break; // context关闭时退出
        }

        bool allowed = false;
        if (request.size() >= 7 && request[5] == "CURVE" && request[6].size() == 32)
        {
            char key_text[41];
            zmq_z85_encode(key_text, reinterpret_cast<const uint8_t *>(request[6].data()), 32);
            allowed = allowed_keys.count(key_text) > 0;
        }

        // 回复帧：版本、请求序号、状态码、状态说明、用户标识、元数据
        const std::string reply[6] = {
            "1.0",
            request[1],
            allowed ? "200" : "400",
            allowed ? "OK" : "Unknown key",
            "",
            ""};
        for (int i = 0; i < 6; i++)
        {
            zmq_send(zap, reply[i].data(), reply[i].size(), i < 5 ? ZMQ_SNDMORE : 0);
        }
    }

    zmq_close(zap);
}

int main()
{
    // 实际使用时密钥应提前生成并保存，这里为了演示直接在程序中生成
    char server_public[41], server_secret[41];
    char client_public[41], client_secret[41];
    if (zmq_curve_keypair(server_public, server_secret) != 0 ||
        zmq_curve_keypair(client_public, client_secret) != 0)
    {
        std::cerr << "libzmq未启用CURVE" << std::endl;
        return 1;
    }

    void *context = zmq_ctx_new();

    // 认证套接字的地址是固定的，并且必须先于服务端套接字bind
    void *zap = zmq_socket(context, ZMQ_REP);
    zmq_bind(zap, "inproc://zeromq.zap.01");
    std::thread zap_thread(zap_loop, zap, std::set<std::string>{client_public});

    // 服务端：声明为CURVE服务端，只需要自己的私钥
    void *server = zmq_socket(context, ZMQ_REP);
    int as_server = 1;
    zmq_setsockopt(server, ZMQ_ZAP_DOMAIN, "global", 6);
    zmq_setsockopt(server, ZMQ_CURVE_SERVER, &as_server, sizeof(as_server));
    zmq_setsockopt(server, ZMQ_CURVE_SECRETKEY, server_secret, 40);
    zmq_bind(server, "tcp://127.0.0.1:5555");

    // 客户端：需要服务端的公钥和自己的密钥对
    void *client = zmq_socket(context, ZMQ_REQ);
    zmq_setsockopt(client, ZMQ_CURVE_SERVERKEY, server_public, 40);
    zmq_setsockopt(client, ZMQ_CURVE_PUBLICKEY, client_public, 40);
    zmq_setsockopt(client, ZMQ_CURVE_SECRETKEY, client_secret, 40);
    zmq_connect(client, "tcp://127.0.0.1:5555");

    zmq_send(client, "Hello", 5, 0);

    char request[16] = {0};
    zmq_recv(server, request, sizeof(request) - 1, 0);
    zmq_send(server, "World", 5, 0);

    char reply[16] = {0};
    zmq_recv(client, reply, sizeof(reply) - 1, 0);
    std::cout << request << " " << reply << std::endl; // Hello World

    zmq_close(client);
    zmq_close(server);
    zmq_ctx_shutdown(context); // 使认证线程中阻塞的接收返回
    zap_thread.join();
    zmq_ctx_term(context);

    return 0;
}
```
{: file='curve.cpp'}

```bash
g++ -std=c++17 curve.cpp -o curve -lzmq -pthread
```
{: file='编译'}

- 密钥以 40 个字符的 Z85 字符串传给`zmq_setsockopt()`，长度参数填`40`
- 认证线程阻塞在接收上，退出时先调用`zmq_ctx_shutdown()`使其返回，等线程关闭套接字后再调用`zmq_ctx_term()`
- 没有在`inproc://zeromq.zap.01`上启动认证套接字时，不会进行客户端认证，任何持有服务端公钥的客户端都能连接，此时 CURVE 只起加密作用
- 认证套接字必须和服务端套接字使用同一个`context`，并且先于服务端套接字`bind`
- 状态码`200`表示放行，`400`表示拒绝。被拒绝的客户端不会收到错误，只是始终收不到回复，客户端应设置接收超时
- 密钥对可以用`zmq_curve_keypair()`生成，源码编译安装的`libzmq`还带有命令行工具`curve_keygen`

### 密钥的保管

- 私钥文件的权限设为`600`，不要提交到代码仓库，也不要写死在源码中
- 服务端公钥需要通过可信的途径分发给客户端，公钥被替换等同于认证失效
- 需要撤销某个客户端时，从服务端的授权列表中删除它的公钥即可

## 注意事项

- 一个进程通常只创建一个`context`，它是线程安全的；套接字不是线程安全的，不要在多个线程之间共用同一个套接字
- ZeroMQ 只负责传输字节，消息内容的序列化（JSON、Protobuf、MessagePack 等）需要自行处理
- 一条消息可以由多帧组成（multipart），常见用法是第一帧放主题、第二帧放数据，对应`send_multipart`/`recv_multipart`
- 默认不加密也不认证，暴露在公网的端口需要启用上文的 CURVE，或放在受信任的网络内
