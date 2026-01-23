# MCP 快速配置指南

## 🚀 5 分钟快速开始

### 步骤 1: 安装核心 MCP 服务器

```bash
# 安装常用的 MCP 服务器
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-brave-search

# Python MCP 服务器
pip install mcp-server-sqlite
pip install fastmcp
```

### 步骤 2: 配置 Claude Desktop

**Windows**: 编辑 `%APPDATA%\Claude\claude_desktop_config.json`

**macOS**: 编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`

### 步骤 3: 基础配置

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Administrator\\PycharmProjects\\AI-Practices"
      ]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### 步骤 4: 重启 Claude Desktop

重启后，Claude 将自动连接到配置的 MCP 服务器。

---

## 📋 推荐配置模板

### 模板 1: AI-Practices 项目专用

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Administrator\\PycharmProjects\\AI-Practices"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_GITHUB_TOKEN>"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "sqlite": {
      "command": "python",
      "args": [
        "-m",
        "mcp_server_sqlite",
        "--db-path",
        "C:\\Users\\Administrator\\PycharmProjects\\AI-Practices\\data\\experiments.db"
      ]
    }
  }
}
```

### 模板 2: 完整开发环境

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Administrator\\PycharmProjects\\AI-Practices"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_GITHUB_TOKEN>"
      }
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "<YOUR_BRAVE_API_KEY>"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/ai_practices"
      }
    },
    "sqlite": {
      "command": "python",
      "args": [
        "-m",
        "mcp_server_sqlite",
        "--db-path",
        "C:\\Users\\Administrator\\PycharmProjects\\AI-Practices\\data\\experiments.db"
      ]
    }
  }
}
```

### 模板 3: 轻量级配置（推荐新手）

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\Administrator\\PycharmProjects\\AI-Practices"
      ]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

---

## 🔑 获取 API Keys

### GitHub Token
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 选择权限: `repo`, `read:org`, `read:user`
4. 生成并复制 token

### Brave Search API Key
1. 访问 https://brave.com/search/api/
2. 注册账号
3. 创建 API key
4. 免费套餐: 2000 次查询/月

### OpenAI API Key
1. 访问 https://platform.openai.com/api-keys
2. 创建新的 API key
3. 复制并保存

---

## 🧪 测试 MCP 服务器

### 方法 1: 使用 MCP Inspector

```bash
# 测试 GitHub 服务器
npx @modelcontextprotocol/inspector npx @modelcontextprotocol/server-github

# 测试文件系统服务器
npx @modelcontextprotocol/inspector npx @modelcontextprotocol/server-filesystem .
```

### 方法 2: 在 Claude Desktop 中测试

重启 Claude Desktop 后，尝试以下命令：

```
# 测试文件系统
请列出当前项目的所有 Python 文件

# 测试 GitHub
查看 AI-Practices 仓库的最新 commits

# 测试记忆
记住：我正在开发 AI-Practices 项目
```

---

## ⚠️ 常见问题

### Q1: MCP 服务器无法启动

**解决方案**:
```bash
# 检查 Node.js 版本
node --version  # 需要 >= 18.0.0

# 检查 Python 版本
python --version  # 需要 >= 3.9

# 重新安装 MCP 服务器
npm install -g @modelcontextprotocol/server-github --force
```

### Q2: 找不到配置文件

**Windows**:
```powershell
# 创建配置目录
mkdir %APPDATA%\Claude

# 创建配置文件
notepad %APPDATA%\Claude\claude_desktop_config.json
```

**macOS**:
```bash
# 创建配置目录
mkdir -p ~/Library/Application\ Support/Claude

# 创建配置文件
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

### Q3: 权限错误

**解决方案**:
```bash
# Windows: 以管理员身份运行
# macOS/Linux: 使用 sudo
sudo npm install -g @modelcontextprotocol/server-github
```

---

## 📊 推荐的 MCP 服务器组合

### 组合 1: Python 开发者
- ✅ filesystem
- ✅ github
- ✅ memory
- ✅ sqlite
- ✅ brave-search

### 组合 2: 全栈开发者
- ✅ filesystem
- ✅ github
- ✅ memory
- ✅ postgres
- ✅ brave-search
- ✅ puppeteer

### 组合 3: 数据科学家
- ✅ filesystem
- ✅ memory
- ✅ postgres
- ✅ sqlite
- ✅ bigquery
- ✅ jupyter

### 组合 4: DevOps 工程师
- ✅ github
- ✅ docker
- ✅ kubernetes
- ✅ aws
- ✅ circleci

---

## 🎯 下一步

1. **选择模板** - 根据需求选择配置模板
2. **获取 API Keys** - 申请必要的 API 密钥
3. **配置文件** - 编辑 Claude Desktop 配置
4. **测试验证** - 重启并测试 MCP 功能
5. **探索更多** - 查看 [MCP_EXTENSION_GUIDE.md](./MCP_EXTENSION_GUIDE.md)

---

## 📚 相关文档

- [MCP 扩展指南](./MCP_EXTENSION_GUIDE.md) - 完整的 MCP 功能扩展方案
- [MCP 官方文档](https://modelcontextprotocol.io/)
- [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)

---

**提示**: 从轻量级配置开始，逐步添加更多 MCP 服务器！

**更新日期**: 2026-01-23
