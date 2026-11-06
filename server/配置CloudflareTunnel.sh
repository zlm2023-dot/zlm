#!/usr/bin/env bash
# 在「识别服务所在的服务器」上安装 cloudflared，把本机 8080 通过 Cloudflare Tunnel 暴露出去。
# 隧道是服务器主动出站长连接，不需要公网 IP 可达，也不需要在安全组放通 22 / 8080 入站。
#
# Token 从 Cloudflare Zero Trust → Networks → Tunnels 复制。
# 推荐先把 token 写入 /root/.cf_tunnel_token（该文件已被 .gitignore 排除，切勿提交仓库）：
#     echo '粘贴TOKEN' > /root/.cf_tunnel_token && chmod 600 /root/.cf_tunnel_token
# 然后执行：
#     sudo bash 配置CloudflareTunnel.sh
# 也可临时用参数传入（会留在 shell 历史里，不推荐）：
#     sudo bash 配置CloudflareTunnel.sh '<TOKEN>'
set -euo pipefail

PORT="${PORT:-8080}"
TOKEN_FILE="${QSCAN_TUNNEL_TOKEN_FILE:-/root/.cf_tunnel_token}"
TOKEN="${1:-}"

[ "$(id -u)" -eq 0 ] || { echo "请用 root 执行：sudo bash 配置CloudflareTunnel.sh"; exit 1; }
if [ -z "$TOKEN" ] && [ -f "$TOKEN_FILE" ]; then TOKEN="$(cat "$TOKEN_FILE")"; fi
[ -n "$TOKEN" ] || { echo "未找到 Tunnel Token。请把 token 写入 $TOKEN_FILE 后重试"; exit 1; }

echo "== 1/4 确认本机识别服务在线 =="
if ! curl -fsS -m 5 "http://127.0.0.1:${PORT}/ping"; then
  echo "本机服务不通。请先启动识别服务，例如："
  echo "  cd /root/qscan-online && bash 一键部署.sh"
  exit 1
fi
echo

echo "== 2/4 安装 cloudflared =="
if ! command -v cloudflared >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64|amd64)  PKG="cloudflared-linux-amd64" ;;
    aarch64|arm64) PKG="cloudflared-linux-arm64" ;;
    *) echo "未知 CPU 架构：$(uname -m)"; exit 1 ;;
  esac
  DL="https://github.com/cloudflare/cloudflared/releases/latest/download/${PKG}"
  if ! curl -fsSL --max-time 60 -o /usr/local/bin/cloudflared "$DL"; then
    echo "GitHub 直连下载失败，改用加速镜像重试"
    curl -fsSL --max-time 120 -o /usr/local/bin/cloudflared "https://ghproxy.net/${DL}" || {
      echo "自动下载失败。请改用 Cloudflare 官方仓库安装："
      echo "  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg"
      echo "  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' > /etc/apt/sources.list.d/cloudflared.list"
      echo "  apt-get update && apt-get install -y cloudflared"
      exit 1
    }
  fi
  chmod +x /usr/local/bin/cloudflared
fi
cloudflared --version

echo "== 3/4 注册为系统服务（开机自启） =="
cloudflared service install "$TOKEN"
systemctl enable cloudflared >/dev/null 2>&1 || true
systemctl restart cloudflared || true
systemctl is-active cloudflared || true

echo "== 4/4 下一步请在 Cloudflare 控制台完成 =="
echo "Zero Trust → Networks → Tunnels → 选中刚出现的隧道 → Configure → Public Hostname"
echo "  Add a public hostname："
echo "    Subdomain : api"
echo "    Domain    : 你注册的域名"
echo "    Path      : 留空"
echo "    Service   : HTTP"
echo "    URL       : localhost:${PORT}"
echo "保存后 Cloudflare 会自动添加 DNS 记录。随后验证："
echo "  curl https://api.你的域名/ping"
echo
echo "最后把 docs/config.js 里的 window.QSCAN_API 改成 https://api.你的域名 ，"
echo "commit + push，Cloudflare Pages 会自动重新部署前端。"
