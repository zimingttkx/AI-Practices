# Droid CLI 深度优化指南

## 概述

基于 GitHub 和互联网的最新研究，本指南提供全面的终端 CLI 优化方案，旨在提升 Droid 的开发效率和质量。

---

## 1. 现代 Rust CLI 工具替代方案

### 1.1 核心工具替换

| 传统命令 | Rust 替代 | 优势 | 安装命令 |
|---------|----------|------|---------|
| `cat` | **bat** | 语法高亮、Git 集成、行号 | `cargo install bat` |
| `ls` | **eza** | 颜色、图标、树形视图、Git 状态 | `cargo install eza` |
| `grep` | **ripgrep (rg)** | 速度快 10-100x、智能忽略 | `cargo install ripgrep` |
| `find` | **fd** | 更简洁语法、默认忽略 .gitignore | `cargo install fd-find` |
| `du` | **dust** | 可视化磁盘使用、更直观 | `cargo install du-dust` |
| `diff` | **delta** | 语法高亮、行号、Git 集成 | `cargo install git-delta` |
| `sed` | **sd** | 更简单的语法 | `cargo install sd` |
| `top` | **bottom (btm)** | 现代化 UI、更多信息 | `cargo install bottom` |
| `ps` | **procs** | 彩色输出、树形视图 | `cargo install procs` |
| `hexdump` | **hexyl** | 彩色十六进制查看 | `cargo install hexyl` |

### 1.2 Windows 安装 (推荐 Scoop)

```powershell
# 安装 Scoop
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# 添加 extras bucket
scoop bucket add extras

# 安装 Rust CLI 工具
scoop install bat eza ripgrep fd dust delta bottom procs
```

### 1.3 macOS/Linux 安装

```bash
# Homebrew
brew install bat eza ripgrep fd dust git-delta bottom procs

# 或使用 Cargo
cargo install bat eza ripgrep fd-find du-dust git-delta bottom procs
```

---

## 2. Shell 优化

### 2.1 Zsh + Starship (推荐)

**Starship** 是一个快速、可定制的跨平台 Shell 提示符，比 Oh-My-Zsh 更轻量。

#### 安装 Starship

```bash
# Windows (PowerShell)
winget install starship

# macOS
brew install starship

# Linux
curl -sS https://starship.rs/install.sh | sh
```

#### 配置 Starship

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
eval "$(starship init bash)"  # Bash
eval "$(starship init zsh)"   # Zsh

# PowerShell: 添加到 $PROFILE
Invoke-Expression (&starship init powershell)
```

#### Starship 配置文件 (~/.config/starship.toml)

```toml
# 开发者优化配置
format = """
$username\
$hostname\
$directory\
$git_branch\
$git_status\
$python\
$nodejs\
$rust\
$golang\
$docker_context\
$cmd_duration\
$line_break\
$character"""

[character]
success_symbol = "[➜](bold green)"
error_symbol = "[✗](bold red)"

[directory]
truncation_length = 3
truncate_to_repo = true

[git_branch]
symbol = " "
format = "[$symbol$branch]($style) "

[git_status]
format = '([\[$all_status$ahead_behind\]]($style) )'

[python]
symbol = " "
format = '[${symbol}${pyenv_prefix}(${version} )(\($virtualenv\) )]($style)'

[nodejs]
symbol = " "
format = "[$symbol($version )]($style)"

[rust]
symbol = " "
format = "[$symbol($version )]($style)"

[cmd_duration]
min_time = 500
format = "[$duration]($style) "
```

### 2.2 Oh-My-Zsh 优化 (如果使用)

```bash
# ~/.zshrc 优化配置
DISABLE_AUTO_UPDATE="true"
DISABLE_MAGIC_FUNCTIONS="true"
DISABLE_COMPFIX="true"

# 优化 compinit 加载
autoload -Uz compinit
if [ "$(date +'%j')" != "$(stat -f '%Sm' -t '%j' ~/.zcompdump 2>/dev/null)" ]; then
    compinit
else
    compinit -C
fi
```

### 2.3 必备 Zsh 插件

```bash
# 仅安装必要插件
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
    fzf
)
```

---

## 3. 终端多路复用器 (tmux)

### 3.1 安装

```bash
# Windows (需要 WSL)
sudo apt install tmux

# macOS
brew install tmux

# Linux
sudo apt install tmux  # Debian/Ubuntu
sudo dnf install tmux  # Fedora
```

### 3.2 核心快捷键

| 操作 | 快捷键 |
|------|--------|
| 新建会话 | `tmux new -s name` |
| 分离会话 | `Ctrl+b d` |
| 重连会话 | `tmux attach -t name` |
| 水平分屏 | `Ctrl+b %` |
| 垂直分屏 | `Ctrl+b "` |
| 切换窗格 | `Ctrl+b 方向键` |
| 新建窗口 | `Ctrl+b c` |
| 切换窗口 | `Ctrl+b n/p` |
| 关闭窗格 | `Ctrl+b x` |

### 3.3 优化配置 (~/.tmux.conf)

```bash
# 修改前缀键为 Ctrl+a
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# 启用鼠标
set -g mouse on

# 从 1 开始编号
set -g base-index 1
setw -g pane-base-index 1

# 更好的分屏快捷键
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# 快速重载配置
bind r source-file ~/.tmux.conf \; display "Reloaded!"

# 状态栏
set -g status-style 'bg=#333333 fg=#5eacd3'
set -g status-left-length 50
set -g status-right '%Y-%m-%d %H:%M '

# 256 色支持
set -g default-terminal "screen-256color"

# 减少延迟
set -sg escape-time 0

# 增加历史记录
set -g history-limit 50000
```

### 3.4 推荐插件 (TPM)

```bash
# 安装 TPM
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# ~/.tmux.conf 添加插件
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'

# 初始化 TPM (放在配置文件最后)
run '~/.tmux/plugins/tpm/tpm'
```

---

## 4. 模糊搜索工具 (fzf)

### 4.1 安装

```bash
# Windows
scoop install fzf

# macOS
brew install fzf

# Linux
sudo apt install fzf
```

### 4.2 Shell 集成

```bash
# Bash (~/.bashrc)
eval "$(fzf --bash)"

# Zsh (~/.zshrc)
source <(fzf --zsh)

# PowerShell
# 需要 PSFzf 模块
Install-Module PSFzf -Scope CurrentUser
Set-PsFzfOption -PSReadlineChordProvider 'Ctrl+t' -PSReadlineChordReverseHistory 'Ctrl+r'
```

### 4.3 常用命令

```bash
# 文件搜索
fzf

# 历史命令搜索
Ctrl+r

# 目录跳转 (配合 cd)
cd **<TAB>

# 与其他命令结合
vim $(fzf)
cat $(fzf)
```

### 4.4 自定义配置

```bash
# ~/.bashrc 或 ~/.zshrc
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_DEFAULT_OPTS='--height 40% --layout=reverse --border'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'
```

---

## 5. AI 编程助手 CLI 工具

### 5.1 主流 AI CLI 工具对比

| 工具 | 特点 | 安装 |
|------|------|------|
| **Claude Code** | Anthropic 官方、强大的代理能力 | `npm install -g @anthropic-ai/claude-code` |
| **Aider** | 开源、多模型支持、Git 集成 | `pip install aider-chat` |
| **Codex CLI** | OpenAI 官方 | `npm install -g @openai/codex` |
| **Continue** | VS Code/JetBrains 插件、多模型 | IDE 扩展 |
| **Tabby** | 自托管、开源 | Docker 或本地安装 |
| **Shell-GPT** | 命令行 AI 助手 | `pip install shell-gpt` |

### 5.2 Aider 配置

```bash
# 安装
pip install aider-chat

# 使用 (支持多种模型)
aider --model claude-3-5-sonnet  # Claude
aider --model gpt-4              # GPT-4
aider --model ollama/llama3.2    # 本地模型

# 常用命令
aider file1.py file2.py  # 添加文件到上下文
/add file.py             # 在会话中添加文件
/drop file.py            # 移除文件
/diff                    # 查看更改
/commit                  # 提交更改
```

### 5.3 Shell-GPT 配置

```bash
# 安装
pip install shell-gpt

# 配置 API Key
export OPENAI_API_KEY="your-key"

# 使用
sgpt "解释这段代码的作用"
sgpt --shell "查找大于 100MB 的文件"
sgpt --code "写一个快速排序算法"
```

### 5.4 本地 AI (Ollama)

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull llama3.2
ollama pull codellama
ollama pull deepseek-coder

# 使用
ollama run codellama "写一个 Python 函数计算斐波那契数列"
```

---

## 6. 开发效率工具

### 6.1 目录跳转 (zoxide)

```bash
# 安装
scoop install zoxide   # Windows
brew install zoxide    # macOS
sudo apt install zoxide # Linux

# 配置 (~/.bashrc 或 ~/.zshrc)
eval "$(zoxide init bash)"  # Bash
eval "$(zoxide init zsh)"   # Zsh

# 使用
z project    # 跳转到包含 "project" 的目录
zi           # 交互式选择
```

### 6.2 JSON 处理 (jq)

```bash
# 安装
scoop install jq   # Windows
brew install jq    # macOS
sudo apt install jq # Linux

# 使用
cat data.json | jq '.name'
curl api.example.com | jq '.[0].id'
```

### 6.3 HTTP 客户端 (httpie, curlie)

```bash
# HTTPie
pip install httpie
http GET api.example.com/users

# Curlie (curl 的友好替代)
brew install curlie
curlie GET api.example.com/users
```

### 6.4 Git 增强

```bash
# lazygit - TUI Git 客户端
scoop install lazygit   # Windows
brew install lazygit    # macOS

# git-delta - 更好的 diff
git config --global core.pager delta
git config --global interactive.diffFilter "delta --color-only"

# gh - GitHub CLI
scoop install gh
gh pr create
gh issue list
gh repo clone owner/repo
```

---

## 7. 终端模拟器推荐

### 7.1 跨平台推荐

| 终端 | 平台 | 特点 |
|------|------|------|
| **Warp** | macOS, Linux | AI 集成、现代 UI、块编辑 |
| **Alacritty** | 全平台 | Rust 编写、GPU 加速、极快 |
| **WezTerm** | 全平台 | Lua 配置、多路复用、GPU 加速 |
| **Kitty** | macOS, Linux | GPU 加速、图像支持 |
| **Windows Terminal** | Windows | 微软官方、多标签、可定制 |

### 7.2 Windows Terminal 配置

```json
{
    "profiles": {
        "defaults": {
            "font": {
                "face": "JetBrainsMono Nerd Font",
                "size": 12
            },
            "colorScheme": "One Half Dark",
            "opacity": 95,
            "useAcrylic": true
        }
    },
    "actions": [
        { "command": "paste", "keys": "ctrl+v" },
        { "command": "copy", "keys": "ctrl+c" },
        { "command": "find", "keys": "ctrl+shift+f" }
    ]
}
```

---

## 8. Droid 深度优化建议

### 8.1 核心工具链

```bash
# 必装工具 (优先级从高到低)
1. ripgrep (rg)     # 代码搜索，Droid 已内置
2. fd               # 文件查找
3. bat              # 文件查看
4. fzf              # 模糊搜索
5. delta            # Git diff
6. eza              # 目录列表
7. zoxide           # 目录跳转
8. lazygit          # Git TUI
```

### 8.2 Shell 配置优先级

```
1. Starship prompt  - 快速、信息丰富
2. fzf 集成         - 历史搜索、文件查找
3. zoxide           - 智能目录跳转
4. 别名配置         - 常用命令简化
```

### 8.3 推荐别名配置

```bash
# ~/.bashrc 或 ~/.zshrc

# Rust CLI 工具别名
alias cat='bat'
alias ls='eza'
alias ll='eza -la --git'
alias tree='eza --tree'
alias grep='rg'
alias find='fd'
alias du='dust'
alias top='btm'
alias ps='procs'

# Git 别名
alias g='git'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'
alias lg='lazygit'

# 开发别名
alias py='python'
alias pip='pip3'
alias venv='python -m venv .venv'
alias activate='source .venv/bin/activate'

# 快捷操作
alias ..='cd ..'
alias ...='cd ../..'
alias c='clear'
alias h='history'
alias ports='netstat -tulanp'
```

### 8.4 环境变量优化

```bash
# 编辑器
export EDITOR='code'
export VISUAL='code'

# 历史记录
export HISTSIZE=50000
export HISTFILESIZE=50000
export HISTCONTROL=ignoreboth:erasedups

# 颜色
export CLICOLOR=1
export LSCOLORS=GxFxCxDxBxegedabagaced

# FZF
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_DEFAULT_OPTS='--height 40% --layout=reverse --border --preview "bat --color=always {}"'
```

---

## 9. 自动化安装脚本

### 9.1 Windows (PowerShell)

```powershell
# install-cli-tools.ps1

# 安装 Scoop
if (!(Get-Command scoop -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    irm get.scoop.sh | iex
}

# 添加 bucket
scoop bucket add extras
scoop bucket add nerd-fonts

# 安装工具
$tools = @(
    "git",
    "ripgrep",
    "fd",
    "bat",
    "eza",
    "dust",
    "delta",
    "bottom",
    "procs",
    "fzf",
    "zoxide",
    "lazygit",
    "jq",
    "starship"
)

foreach ($tool in $tools) {
    scoop install $tool
}

# 安装 Nerd Font
scoop install JetBrainsMono-NF

Write-Host "CLI tools installed successfully!" -ForegroundColor Green
```

### 9.2 macOS/Linux (Bash)

```bash
#!/bin/bash
# install-cli-tools.sh

# 检测包管理器
if command -v brew &> /dev/null; then
    PKG_MANAGER="brew"
elif command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
else
    echo "Unsupported package manager"
    exit 1
fi

# 安装工具
TOOLS="ripgrep fd-find bat eza dust git-delta bottom procs fzf zoxide lazygit jq starship"

if [ "$PKG_MANAGER" = "brew" ]; then
    brew install $TOOLS
elif [ "$PKG_MANAGER" = "apt" ]; then
    sudo apt update
    sudo apt install -y ripgrep fd-find bat fzf jq
    # 其他工具需要 cargo
    cargo install eza du-dust git-delta bottom procs zoxide starship
elif [ "$PKG_MANAGER" = "dnf" ]; then
    sudo dnf install -y ripgrep fd-find bat fzf jq
    cargo install eza du-dust git-delta bottom procs zoxide starship
fi

echo "CLI tools installed successfully!"
```

---

## 10. 参考资源

### GitHub 仓库

- [awesome-cli-apps](https://github.com/agarrharr/awesome-cli-apps) - 18.5k stars
- [awesome-shell](https://github.com/alebcay/awesome-shell) - Shell 工具大全
- [awesome-devtools](https://github.com/devtoolsd/awesome-devtools) - 开发工具集合
- [rust-cli-recommendations](https://gist.github.com/sts10/daadbc2f403bdffad1b6d33aff016c0a) - Rust CLI 工具列表
- [tldr-pages](https://github.com/tldr-pages/tldr) - 命令速查手册
- [google/zx](https://github.com/google/zx) - 更好的脚本编写

### 学习资源

- [Command Line Interface Guidelines](https://clig.dev/) - CLI 设计指南
- [The Art of Command Line](https://github.com/jlevy/the-art-of-command-line)
- [Starship 官方文档](https://starship.rs/)
- [tmux Cheat Sheet](https://tmuxcheatsheet.com/)

### 文章推荐

- "15 Rust Tools to Level Up Your Linux Terminal" - DEV Community
- "13 CLI Tools Every Developer Should Master in 2025" - HostZealot
- "My Favorite 8 CLI Tools for Everyday Development" - Medium

---

## 总结

### 核心优化要点

1. **替换传统工具** - 使用 Rust 编写的现代替代品 (bat, eza, ripgrep, fd)
2. **优化 Shell** - Starship prompt + 精简插件配置
3. **终端多路复用** - tmux 管理多个会话
4. **模糊搜索** - fzf 加速文件和历史查找
5. **AI 辅助** - Aider, Shell-GPT, Ollama 本地模型
6. **Git 增强** - lazygit, delta, gh CLI
7. **别名和环境变量** - 减少重复输入

### 预期效果

- **搜索效率提升**: ripgrep 比 grep 快 10-100x
- **导航效率提升**: zoxide + fzf 减少 80% 目录切换时间
- **可视化提升**: bat + eza + delta 提供更好的信息展示
- **工作流优化**: tmux + lazygit 提升多任务处理能力
- **AI 辅助**: 自动化重复任务，加速编码

---

**文档版本**: 1.0  
**更新日期**: 2026-01-23  
**适用平台**: Windows, macOS, Linux
