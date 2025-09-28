Firecrawl Agent Integration with LangChain MCP

## 1. Giới thiệu

Đoạn code này xây dựng một **chat agent** có khả năng:

* Kết nối với **Firecrawl MCP Server** (dựa trên `mcp`).
* Tải các **MCP Tools** (công cụ để crawl, scrape, trích xuất dữ liệu web).
* Tích hợp với **LangChain ReAct Agent** để xử lý hội thoại.
* Dùng **Ollama model** (`qwen3:0.6b`) làm mô hình ngôn ngữ.

Mục tiêu: Tạo ra một chatbot CLI có thể nhận lệnh từ người dùng → sử dụng Firecrawl để thu thập dữ liệu web → trả lời kết quả.

---

## 2. Các thư viện sử dụng

* **mcp**: Cung cấp cơ chế kết nối client–server theo chuẩn MCP (Model Context Protocol).
* **langchain_mcp_adapters**: Cho phép load các MCP tools để dùng trong LangChain.
* **langgraph**: Cung cấp hàm dựng sẵn `create_react_agent` (agent ReAct).
* **dotenv**: Đọc biến môi trường từ file `.env`.
* **langchain_ollama**: Tích hợp mô hình ngôn ngữ Ollama.
* **asyncio**: Dùng để chạy các hàm bất đồng bộ (asynchronous).

---

## 3. Cấu hình mô hình AI

```python
model = ChatOllama(model="qwen3:0.6b")
```

* Dùng mô hình **Qwen 0.6B** từ Ollama.
* Đây là mô hình LLM sẽ đứng sau agent để suy luận.

---

## 4. Cấu hình Firecrawl Server

```python
server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
    },
    args=["firecrawl-mcp"]
)
```

* MCP server được khởi chạy bằng `npx firecrawl-mcp`.
* Biến môi trường **FIRECRAWL_API_KEY** được lấy từ `.env`.
* Giao tiếp thông qua chuẩn **stdio** (standard input/output).

---

## 5. Hàm `main()`

Đây là **hàm điều khiển chính** của chương trình.

### 5.1. Kết nối tới Firecrawl

```python
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
```

* Tạo kết nối client-server với Firecrawl.
* Khởi tạo một **phiên làm việc** (session).

### 5.2. Load công cụ từ MCP

```python
tools = await load_mcp_tools(session)
```

* Tải danh sách các **Firecrawl tools** mà MCP cung cấp.
* Ví dụ: crawl URL, scrape nội dung, extract data.

### 5.3. Tạo LangChain Agent

```python
agent = create_react_agent(model, tools)
```

* Tạo một agent theo **ReAct pattern** (Reason + Act).
* Agent có thể **suy luận** và **gọi tool Firecrawl** khi cần.

### 5.4. Thiết lập thông điệp hệ thống

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
    }
]
```

* Thông điệp hệ thống định nghĩa hành vi của agent:
  *“Bạn là một trợ lý có khả năng thu thập dữ liệu web bằng Firecrawl.”*

### 5.5. Vòng lặp hội thoại

```python
while True:
    user_input = input("\nYou: ")
    if user_input == "quit":
        break
```

* Người dùng nhập lệnh từ terminal.
* Nếu nhập `"quit"` thì kết thúc.

### 5.6. Gọi agent xử lý

```python
agent_response = await agent.ainvoke({"messages": messages})
ai_message = agent_response["messages"][-1].content
```

* Truyền toàn bộ lịch sử hội thoại (`messages`) cho agent.
* Agent sẽ suy luận và có thể gọi Firecrawl tool.
* Lấy ra **tin nhắn cuối cùng** của agent để hiển thị cho người dùng.

---

## 6. Xử lý lỗi

```python
except McpError as e:
    print(f"Lỗi MCP: {e}")
except Exception as e:
    print(f"Một lỗi không mong muốn đã xảy ra: {e}")
```

* Nếu có lỗi từ MCP → in ra.
* Nếu lỗi chung → báo lỗi tổng quát.

---

## 7. Điểm nổi bật

* 🎯 Kết hợp **LangChain + MCP + Firecrawl**.
* 🤖 Có khả năng **tự động dùng tool** (crawl/scrape/extract).
* 📝 Xây dựng giao diện CLI đơn giản, dễ mở rộng.
* 🔒 Dùng `.env` để bảo mật API key.

---

## 8. Cách chạy

1. Cài đặt thư viện:

   ```bash
   pip install mcp langchain langchain-ollama langchain-mcp-adapters python-dotenv
   npm install -g firecrawl-mcp
   ```
2. Tạo file `.env`:

   ```env
   FIRECRAWL_API_KEY=your_api_key_here
   ```
3. Chạy script:

   ```bash
   python main.py
   ```
4. Chat trong terminal:

   ```
   You: crawl https://example.com
   Agent: Đã crawl xong, dữ liệu như sau...
   ```

---

Tóm lại, code này triển khai một **AI Agent CLI** có khả năng **scrape dữ liệu web bằng Firecrawl**, tích hợp trong **LangChain**, dùng mô hình Ollama làm nền tảng ngôn ngữ.

