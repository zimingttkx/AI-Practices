#!/bin/bash
# Droid CLI 工具安装脚本 (macOS/Linux)
# 用法: ./install_cli_tools.sh [--full] [--minimal]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 参数处理
FULL=false
MINIMAL=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --full) FULL=true ;;
        --minimal) MINIMAL=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${CYAN}========================================"
echo -e "  Droid CLI Tools Installer             "
echo -e "========================================${NC}"
echo ""

# 检测操作系统和包管理器
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    elif [[ -f /etc/debian_version ]]; then
        OS="debian"
        PKG_MANAGER="apt"
    elif [[ -f /etc/redhat-release ]]; then
        OS="redhat"
        PKG_MANAGER="dnf"
    elif [[ -f /etc/arch-release ]]; then
        OS="arch"
        PKG_MANAGER="pacman"
    else
        echo -e "${RED}Unsupported operating system${NC}"
        exit 1
    fi
    echo -e "${GREEN}Detected: $OS (using $PKG_MANAGER)${NC}"
}

# 安装 Homebrew (macOS)
install_homebrew() {
    if [[ "$OS" == "macos" ]] && ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Installing Homebrew...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
}

# 安装 Rust (用于 cargo install)
install_rust() {
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}Installing Rust...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
}

# 安装核心工具
install_core_tools() {
    echo -e "\n${YELLOW}Installing core CLI tools...${NC}"
    
    case $PKG_MANAGER in
        brew)
            brew install ripgrep fd bat fzf jq starship
            ;;
        apt)
            sudo apt update
            sudo apt install -y ripgrep fd-find bat fzf jq
            # fd 在 Debian/Ubuntu 上叫 fd-find，创建别名
            if [[ ! -L /usr/local/bin/fd ]]; then
                sudo ln -sf $(which fdfind) /usr/local/bin/fd 2>/dev/null || true
            fi
            # bat 在某些系统上叫 batcat
            if [[ ! -L /usr/local/bin/bat ]]; then
                sudo ln -sf $(which batcat) /usr/local/bin/bat 2>/dev/null || true
            fi
            # Starship 需要单独安装
            curl -sS https://starship.rs/install.sh | sh -s -- -y
            ;;
        dnf)
            sudo dnf install -y ripgrep fd-find bat fzf jq
            curl -sS https://starship.rs/install.sh | sh -s -- -y
            ;;
        pacman)
            sudo pacman -S --noconfirm ripgrep fd bat fzf jq starship
            ;;
    esac
}

# 安装扩展工具
install_extended_tools() {
    echo -e "\n${YELLOW}Installing extended CLI tools...${NC}"
    
    case $PKG_MANAGER in
        brew)
            brew install eza dust git-delta bottom procs zoxide lazygit gh
            ;;
        apt|dnf)
            install_rust
            cargo install eza du-dust git-delta bottom procs zoxide
            # lazygit 需要单独安装
            if [[ "$PKG_MANAGER" == "apt" ]]; then
                LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*')
                curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
                tar xf lazygit.tar.gz lazygit
                sudo install lazygit /usr/local/bin
                rm lazygit lazygit.tar.gz
            fi
            ;;
        pacman)
            sudo pacman -S --noconfirm eza dust git-delta bottom procs zoxide lazygit github-cli
            ;;
    esac
}

# 配置 Shell
configure_shell() {
    echo -e "\n${YELLOW}Configuring shell...${NC}"
    
    # 检测当前 shell
    SHELL_NAME=$(basename "$SHELL")
    
    case $SHELL_NAME in
        bash)
            RC_FILE="$HOME/.bashrc"
            STARSHIP_INIT='eval "$(starship init bash)"'
            ZOXIDE_INIT='eval "$(zoxide init bash)"'
            FZF_INIT='eval "$(fzf --bash)"'
            ;;
        zsh)
            RC_FILE="$HOME/.zshrc"
            STARSHIP_INIT='eval "$(starship init zsh)"'
            ZOXIDE_INIT='eval "$(zoxide init zsh)"'
            FZF_INIT='source <(fzf --zsh)'
            ;;
        *)
            echo -e "${RED}Unsupported shell: $SHELL_NAME${NC}"
            return
            ;;
    esac
    
    # 备份原配置
    if [[ -f "$RC_FILE" ]]; then
        cp "$RC_FILE" "$RC_FILE.backup"
    fi
    
    # 添加配置
    CONFIG_BLOCK="
# ========== Droid CLI Tools Configuration ==========

# Starship prompt
$STARSHIP_INIT

# Zoxide (smart cd)
$ZOXIDE_INIT

# FZF
$FZF_INIT
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_DEFAULT_OPTS='--height 40% --layout=reverse --border'

# CLI Tool Aliases
alias cat='bat'
alias ls='eza'
alias ll='eza -la --git'
alias tree='eza --tree'
alias grep='rg'
alias find='fd'
alias du='dust'
alias top='btm'
alias lg='lazygit'

# Git Aliases
alias g='git'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'

# Quick Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias c='clear'

# ========== End Droid CLI Tools Configuration ==========
"
    
    # 检查是否已配置
    if ! grep -q "Droid CLI Tools Configuration" "$RC_FILE" 2>/dev/null; then
        echo "$CONFIG_BLOCK" >> "$RC_FILE"
        echo -e "${GREEN}  Added configuration to $RC_FILE${NC}"
    else
        echo -e "${GREEN}  Configuration already exists in $RC_FILE${NC}"
    fi
}

# 创建 Starship 配置
create_starship_config() {
    echo -e "\n${YELLOW}Creating Starship configuration...${NC}"
    
    mkdir -p "$HOME/.config"
    
    cat > "$HOME/.config/starship.toml" << 'EOF'
# Starship Configuration for Droid
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

[golang]
symbol = " "
format = "[$symbol($version )]($style)"

[cmd_duration]
min_time = 500
format = "[$duration]($style) "
EOF
    
    echo -e "${GREEN}  Created ~/.config/starship.toml${NC}"
}

# 显示摘要
show_summary() {
    echo -e "\n${CYAN}========================================"
    echo -e "  Installation Complete!               "
    echo -e "========================================${NC}"
    echo ""
    echo -e "${YELLOW}Installed tools:${NC}"
    echo -e "  - ripgrep (rg)  : Fast search"
    echo -e "  - fd            : File finder"
    echo -e "  - bat           : Better cat"
    echo -e "  - fzf           : Fuzzy finder"
    echo -e "  - starship      : Shell prompt"
    
    if [[ "$MINIMAL" != true ]]; then
        echo -e "  - eza           : Modern ls"
        echo -e "  - dust          : Disk usage"
        echo -e "  - delta         : Git diff"
        echo -e "  - lazygit       : Git TUI"
        echo -e "  - zoxide        : Smart cd"
    fi
    
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Restart your terminal or run: source $RC_FILE"
    echo -e "  2. Install a Nerd Font for icons"
    echo -e "  3. Run 'starship config' to customize prompt"
    echo ""
}

# 主执行流程
main() {
    detect_os
    
    if [[ "$OS" == "macos" ]]; then
        install_homebrew
    fi
    
    install_core_tools
    
    if [[ "$MINIMAL" != true ]]; then
        install_extended_tools
    fi
    
    configure_shell
    create_starship_config
    show_summary
}

main
