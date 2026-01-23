# CLI 工具快速参考卡

## 🔧 Rust CLI 工具替代

| 传统 | 现代 | 示例 |
|------|------|------|
| `cat file` | `bat file` | 语法高亮 |
| `ls -la` | `eza -la --git` | 图标+Git状态 |
| `grep pattern` | `rg pattern` | 10-100x 更快 |
| `find -name` | `fd pattern` | 更简洁语法 |
| `du -sh *` | `dust` | 可视化 |
| `diff a b` | `delta a b` | 语法高亮 |

## ⌨️ 常用命令

### ripgrep (rg)
```bash
rg "pattern"              # 递归搜索
rg "pattern" -t py        # 只搜索 Python 文件
rg "pattern" -g "*.js"    # glob 匹配
rg "pattern" -i           # 忽略大小写
rg "pattern" -C 3         # 显示上下文
```

### fd
```bash
fd "pattern"              # 查找文件
fd -e py                  # 按扩展名
fd -t d                   # 只目录
fd -t f -x cmd {}         # 执行命令
```

### bat
```bash
bat file.py               # 查看文件
bat -A file               # 显示不可见字符
bat -l json data          # 指定语言
bat --diff a.py b.py      # 差异对比
```

### fzf
```bash
fzf                       # 交互式选择
Ctrl+r                    # 历史搜索
Ctrl+t                    # 文件搜索
vim $(fzf)                # 结合命令
```

### zoxide
```bash
z project                 # 跳转到匹配目录
zi                        # 交互式选择
z -                       # 返回上一个目录
```

## 📺 tmux 快捷键

| 操作 | 快捷键 |
|------|--------|
| 新建会话 | `tmux new -s name` |
| 分离 | `Ctrl+b d` |
| 重连 | `tmux attach -t name` |
| 水平分屏 | `Ctrl+b %` |
| 垂直分屏 | `Ctrl+b "` |
| 切换窗格 | `Ctrl+b 方向键` |
| 新窗口 | `Ctrl+b c` |
| 下一窗口 | `Ctrl+b n` |
| 上一窗口 | `Ctrl+b p` |
| 关闭窗格 | `Ctrl+b x` |

## 🔀 Git 增强

### lazygit
```bash
lg                        # 启动 TUI
?                         # 帮助
q                         # 退出
space                     # stage/unstage
c                         # 提交
p                         # 推送
```

### gh CLI
```bash
gh pr create              # 创建 PR
gh pr list                # 列出 PR
gh issue create           # 创建 Issue
gh repo clone owner/repo  # 克隆仓库
```

## 🤖 AI CLI 工具

### Aider
```bash
aider file.py             # 编辑文件
/add file.py              # 添加文件
/drop file.py             # 移除文件
/diff                     # 查看更改
/commit                   # 提交
/help                     # 帮助
```

### Shell-GPT
```bash
sgpt "问题"               # 询问
sgpt --shell "描述"       # 生成命令
sgpt --code "描述"        # 生成代码
```

## 📦 快速安装

### Windows (PowerShell)
```powershell
irm get.scoop.sh | iex
scoop install ripgrep fd bat fzf eza starship lazygit
```

### macOS
```bash
brew install ripgrep fd bat fzf eza starship lazygit zoxide
```

### Linux
```bash
sudo apt install ripgrep fd-find bat fzf
curl -sS https://starship.rs/install.sh | sh
```

## ⚙️ 推荐别名

```bash
alias cat='bat'
alias ls='eza'
alias ll='eza -la --git'
alias grep='rg'
alias find='fd'
alias lg='lazygit'
alias g='git'
alias gs='git status'
alias ..='cd ..'
```

---

**版本**: 1.0 | **更新**: 2026-01-23
