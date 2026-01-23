# Droid CLI 工具安装脚本 (Windows PowerShell)
# 用法: .\install_cli_tools.ps1 [-Full] [-Minimal]

param(
    [switch]$Full,
    [switch]$Minimal
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Droid CLI Tools Installer (Windows)  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查并安装 Scoop
function Install-Scoop {
    if (!(Get-Command scoop -ErrorAction SilentlyContinue)) {
        Write-Host "Installing Scoop..." -ForegroundColor Yellow
        Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Invoke-RestMethod get.scoop.sh | Invoke-Expression
    } else {
        Write-Host "Scoop is already installed" -ForegroundColor Green
    }
}

# 添加必要的 bucket
function Add-ScoopBuckets {
    Write-Host "Adding Scoop buckets..." -ForegroundColor Yellow
    scoop bucket add extras 2>$null
    scoop bucket add nerd-fonts 2>$null
}

# 安装核心工具
function Install-CoreTools {
    Write-Host "`nInstalling core CLI tools..." -ForegroundColor Yellow
    
    $coreTools = @(
        "git",
        "ripgrep",      # 快速搜索
        "fd",           # 文件查找
        "bat",          # 文件查看
        "fzf",          # 模糊搜索
        "jq",           # JSON 处理
        "starship"      # Shell prompt
    )
    
    foreach ($tool in $coreTools) {
        Write-Host "  Installing $tool..." -ForegroundColor Gray
        scoop install $tool 2>$null
    }
}

# 安装扩展工具
function Install-ExtendedTools {
    Write-Host "`nInstalling extended CLI tools..." -ForegroundColor Yellow
    
    $extendedTools = @(
        "eza",          # 现代 ls
        "dust",         # 磁盘使用
        "delta",        # Git diff
        "bottom",       # 系统监控
        "procs",        # 进程查看
        "zoxide",       # 智能目录跳转
        "lazygit",      # Git TUI
        "gh"            # GitHub CLI
    )
    
    foreach ($tool in $extendedTools) {
        Write-Host "  Installing $tool..." -ForegroundColor Gray
        scoop install $tool 2>$null
    }
}

# 安装开发工具
function Install-DevTools {
    Write-Host "`nInstalling development tools..." -ForegroundColor Yellow
    
    $devTools = @(
        "nodejs-lts",   # Node.js
        "python",       # Python
        "curl",         # HTTP 客户端
        "wget",         # 下载工具
        "7zip"          # 压缩工具
    )
    
    foreach ($tool in $devTools) {
        Write-Host "  Installing $tool..." -ForegroundColor Gray
        scoop install $tool 2>$null
    }
}

# 安装字体
function Install-Fonts {
    Write-Host "`nInstalling Nerd Fonts..." -ForegroundColor Yellow
    scoop install JetBrainsMono-NF 2>$null
    scoop install FiraCode-NF 2>$null
}

# 配置环境
function Configure-Environment {
    Write-Host "`nConfiguring environment..." -ForegroundColor Yellow
    
    # 创建 PowerShell profile 目录
    $profileDir = Split-Path $PROFILE
    if (!(Test-Path $profileDir)) {
        New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    }
    
    # 添加 Starship 初始化
    $starshipInit = 'Invoke-Expression (&starship init powershell)'
    
    if (!(Test-Path $PROFILE)) {
        New-Item -ItemType File -Path $PROFILE -Force | Out-Null
    }
    
    $profileContent = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
    if ($profileContent -notmatch "starship init") {
        Add-Content $PROFILE "`n# Starship prompt`n$starshipInit"
        Write-Host "  Added Starship to PowerShell profile" -ForegroundColor Green
    }
    
    # 添加别名
    $aliases = @"

# CLI Tool Aliases
Set-Alias -Name cat -Value bat -Option AllScope
Set-Alias -Name ls -Value eza -Option AllScope
Set-Alias -Name grep -Value rg -Option AllScope
Set-Alias -Name find -Value fd -Option AllScope
Set-Alias -Name lg -Value lazygit -Option AllScope

# Zoxide
Invoke-Expression (& { (zoxide init powershell | Out-String) })

# FZF
Set-PsFzfOption -PSReadlineChordProvider 'Ctrl+t' -PSReadlineChordReverseHistory 'Ctrl+r'
"@
    
    if ($profileContent -notmatch "CLI Tool Aliases") {
        Add-Content $PROFILE $aliases
        Write-Host "  Added aliases to PowerShell profile" -ForegroundColor Green
    }
}

# 创建 Starship 配置
function Create-StarshipConfig {
    Write-Host "`nCreating Starship configuration..." -ForegroundColor Yellow
    
    $starshipDir = "$env:USERPROFILE\.config"
    if (!(Test-Path $starshipDir)) {
        New-Item -ItemType Directory -Path $starshipDir -Force | Out-Null
    }
    
    $starshipConfig = @"
# Starship Configuration for Droid
format = """
`$username\
`$hostname\
`$directory\
`$git_branch\
`$git_status\
`$python\
`$nodejs\
`$rust\
`$cmd_duration\
`$line_break\
`$character"""

[character]
success_symbol = "[>](bold green)"
error_symbol = "[x](bold red)"

[directory]
truncation_length = 3
truncate_to_repo = true

[git_branch]
symbol = " "
format = "[`$symbol`$branch](`$style) "

[git_status]
format = '([\[`$all_status`$ahead_behind\]](`$style) )'

[python]
symbol = " "
format = '[`${symbol}`${pyenv_prefix}(`${version} )(\(`$virtualenv\) )](`$style)'

[nodejs]
symbol = " "
format = "[`$symbol(`$version )](`$style)"

[cmd_duration]
min_time = 500
format = "[`$duration](`$style) "
"@
    
    $starshipConfig | Out-File -FilePath "$starshipDir\starship.toml" -Encoding UTF8
    Write-Host "  Created starship.toml" -ForegroundColor Green
}

# 显示安装结果
function Show-Summary {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Installation Complete!               " -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Installed tools:" -ForegroundColor Yellow
    Write-Host "  - ripgrep (rg)  : Fast search" -ForegroundColor Gray
    Write-Host "  - fd            : File finder" -ForegroundColor Gray
    Write-Host "  - bat           : Better cat" -ForegroundColor Gray
    Write-Host "  - fzf           : Fuzzy finder" -ForegroundColor Gray
    Write-Host "  - starship      : Shell prompt" -ForegroundColor Gray
    
    if (!$Minimal) {
        Write-Host "  - eza           : Modern ls" -ForegroundColor Gray
        Write-Host "  - dust          : Disk usage" -ForegroundColor Gray
        Write-Host "  - delta         : Git diff" -ForegroundColor Gray
        Write-Host "  - lazygit       : Git TUI" -ForegroundColor Gray
        Write-Host "  - zoxide        : Smart cd" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Restart PowerShell to apply changes" -ForegroundColor Gray
    Write-Host "  2. Set terminal font to 'JetBrainsMono Nerd Font'" -ForegroundColor Gray
    Write-Host "  3. Run 'starship config' to customize prompt" -ForegroundColor Gray
    Write-Host ""
}

# 主执行流程
try {
    Install-Scoop
    Add-ScoopBuckets
    Install-CoreTools
    
    if (!$Minimal) {
        Install-ExtendedTools
    }
    
    if ($Full) {
        Install-DevTools
        Install-Fonts
    }
    
    Configure-Environment
    Create-StarshipConfig
    Show-Summary
}
catch {
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
