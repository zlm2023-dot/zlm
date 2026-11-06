# 问卷扫描卷识别与核验系统 · GitHub + Cloudflare 部署仓库

用户打开一个网址即可完成：导入题本 → 解析结构 → 上传扫描卷 → 云端识别 → 识别概览 / 人工核验 → 导出 CSV / Excel。
浏览器端不需要安装任何东西，也不需要启动本地服务。

## 架构（重要）

GitHub Pages / Cloudflare Pages **只能托管静态文件**，跑不了 Python + OpenCV 识别引擎。
所以必须拆成两半，只把网页放到 GitHub，识别服务另外跑：

```
浏览器  https://app.你的域名
   │  GitHub Pages 或 Cloudflare Pages（由本仓库 docs/ 发布，纯静态）
   │  前端读 docs/config.js 里的 window.QSCAN_API
   ▼
https://api.你的域名
   │  Cloudflare Tunnel（服务器主动出站，不需要公网 IP / 安全组入站）
   │  或 Render / Railway / Fly.io 从本仓库 Dockerfile 部署
   ▼
Python + OpenCV 识别服务（api/，监听 8080）
   GET  /ping            健康检查
   POST /api/recognize   扫描卷识别
```

服务端已内置 `Access-Control-Allow-Origin: *` 与 OPTIONS 预检，前端跨域调用可用。
静态目录特意叫 `docs/`：GitHub Pages 只允许从 `/ (root)` 或 `/docs` 发布，
而 Cloudflare Pages 可以把任意目录设为输出目录，所以一份代码两边都能用。

## 目录

```
qscan-deploy/
├─ docs/                     静态网页 = Pages 发布目录
│  ├─ index.html             主页面（与原版功能一致）
│  ├─ config.js              ← 部署时只改这一行：填 API 域名
│  ├─ pdf.min.js
│  └─ pdf.worker.min.js
├─ api/                      识别服务（Python / OpenCV）
│  ├─ 混合识别服务.py
│  ├─ 混合识别.py
│  ├─ 问卷识别工具.py
│  ├─ 手写数字模型.pkl
│  └─ requirements.txt
├─ server/
│  └─ 配置CloudflareTunnel.sh   在服务器上装 cloudflared 并注册服务
├─ Dockerfile                供 Render / Railway / Fly.io 构建 API
└─ .gitignore
```

## 路线 A（推荐）：GitHub 托管网页 + Cloudflare 域名 + 已有腾讯云服务器

Tunnel 是服务器主动出站长连接，**不需要公网 IP 可达，也不需要放通 22/8080 入站**，
可绕开服务器端口全部超时的情况。服务器 SSH 不通时，用腾讯云控制台实例页的【登录】按钮开网页版终端。

### 1. 推送仓库到 GitHub

在 GitHub 新建一个**私有**仓库（不要勾选 Add README）。本机终端执行：

```bash
cd "/Users/zoulinmao/Documents/网站搭建/识别扫描手写打勾平台/问卷识别_网页发送包/qscan-deploy"
git config --global user.name  "你的GitHub用户名"
git config --global user.email "注册GitHub的邮箱"
git init
git add .
git commit -m "init: qscan online"
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

HTTPS 推送不能用账号密码：GitHub → 头像 → **Settings → Developer settings →
Personal access tokens → Tokens (classic) → Generate new token**，勾选 `repo`，
把生成的 `ghp_...` 当作 push 时的密码输入。Token 不要写进仓库。

### 2. 在 Cloudflare 注册域名

https://dash.cloudflare.com/sign-up 注册账号 → 左侧 **Domain Registration → Register Domains**
→ 搜索想要的域名 → **Purchase** → 填注册人信息并付款。注册完成后域名自动进入该账号的 DNS 管理，NS 无需手工修改。
（若域名是在别处买的：Dashboard → **Add a site** → 输入域名 → 选 Free 计划 → 按提示把 NS 改成 Cloudflare 给的两个地址。）

### 3. 发布网页（两种，选一种）

**A1 · GitHub Pages（网页真的托管在 GitHub）**
仓库 → **Settings → Pages** → Build and deployment → Source 选 **Deploy from a branch**
→ Branch `main`、文件夹 **`/docs`** → Save。
然后同一页 **Custom domain** 填 `app.你的域名` → Save。
到 Cloudflare → 域名 → **DNS → Records → Add record**：
`Type=CNAME`、`Name=app`、`Target=你的GitHub用户名.github.io`、**Proxy status 必须选 DNS only（灰云）**。
代理开着会导致 GitHub 无法验证域名、证书一直签不下来。等证书签发后勾选 **Enforce HTTPS**。
得到 `https://app.你的域名`。

**A2 · Cloudflare Pages（更省事，代码仍在 GitHub）**
Dashboard → **Workers & Pages → Create application → Pages → Connect to Git**
→ 授权 GitHub 并勾选该仓库 → Begin setup：
Project name 自定、Production branch `main`、Framework preset **None**、
Build command **留空**、Build output directory 填 **`docs`**、Root directory 留空 → **Save and Deploy**。
拿到 `xxx.pages.dev` 后进项目 → **Custom domains** → 绑定 `app.你的域名`（DNS 记录自动添加）。

> 建议用子域（app / www），不要直接用裸域：子域可以一条 CNAME 搞定，裸域要配 4 条 A 记录。

### 4. 让识别服务跑起来，并用 Tunnel 暴露

服务器上先跑起识别服务（用已有的 `线上版/qscan-online.tar.gz`，通过控制台【文件管理】上传或 scp）：

```bash
cd /root && tar -xzf qscan-online.tar.gz && cd qscan-online && bash 一键部署.sh
curl http://127.0.0.1:8080/ping     # 应返回 {"ok": true, "engine": true, ...}
```

Cloudflare → **Zero Trust → Networks → Tunnels → Create a tunnel** → 选 **Cloudflared**
→ 命名 `qscan-api` → Save → 复制页面给出的 **Tunnel Token**。
在服务器网页终端粘贴（最短版，不依赖额外脚本文件）：

```bash
curl -fsSL -o /usr/local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
echo '粘贴TOKEN' > /root/.cf_tunnel_token && chmod 600 /root/.cf_tunnel_token
cloudflared service install "$(cat /root/.cf_tunnel_token)"
systemctl enable --now cloudflared
```

（也可用本仓库 `server/配置CloudflareTunnel.sh`，它会先检查本机 `/ping` 再安装。）

回到 Tunnel 配置页 → **Public Hostname → Add a public hostname**：
Subdomain `api`、Domain 选你的域名、Path 留空、Service `HTTP`、URL `localhost:8080` → Save（DNS 记录自动添加）。
验证：`curl https://api.你的域名/ping`

### 5. 把前端指向 API

编辑 `docs/config.js`，把空值改成你的 API 域名：

```js
window.QSCAN_API = 'https://api.你的域名';
```

```bash
git add docs/config.js && git commit -m "set api domain" && git push
```
Pages 会自动重新发布。打开 `https://app.你的域名`，右上角应显示「服务运行中」。

## 路线 B：不想碰服务器 → GitHub + Cloudflare Pages + 容器平台

1. 完成路线 A 的第 1~3 步（GitHub 仓库、Cloudflare 域名、前端发布）。
2. 在 Render / Railway / Fly.io 新建服务 → **Deploy from GitHub repository** → 选该仓库
   → 环境选 **Docker**（Dockerfile 用仓库根目录那个），端口 `8080`（平台注入 `PORT`，程序自动读取）。
   **内存至少 2GB**，免费套餐常只有 512MB，OpenCV 会 OOM，且会冷启动休眠。
3. 平台给的 `https://xxx.onrender.com` 可直接用；要自定义域就在平台绑定 `api.你的域名`，
   再到 Cloudflare DNS 加它要求的 CNAME 记录，**建议设为 DNS only（灰云）**以避开 CDN 100 秒超时。
4. 把该 https 地址填进 `docs/config.js` 并 push。

## 关键配置与限制

| 项 | 说明 |
|---|---|
| `docs/config.js` | 只填 API 域名，必须 `https://`，结尾不带斜杠；留空 = 与网页同源 |
| 临时覆盖 | `https://app.你的域名/?svc=https://另一个API域名`，不改文件即可临时切换 |
| 混合内容 | 前端是 https，API 也必须是 https，填 `http://` 会被浏览器拦截 |
| 请求体 | 服务端上限 300MB，扫描卷 PDF 通常远小于此 |
| **超时 524** | 经 Cloudflare CDN 代理时，源站约 100 秒无响应会返回 524；单份扫描卷实测约 90 秒，接近临界。若出现 524：减少单次页数、一次只传一份，或走路线 B 并把 DNS 设为 DNS only；彻底解决需把识别改成「提交任务 + 轮询」异步接口 |
| GitHub Pages 证书 | 自定义域的 CNAME 必须保持 **DNS only**，否则 Let's Encrypt 验证失败、Enforce HTTPS 一直 pending |
| 公开访问 | 服务端 CORS 为 `*`，API 域名一旦公开，任何人都能调用。正式使用前建议用 Cloudflare Access / WAF 限制来源，或先不公开分享网址 |
| 凭据 | Tunnel Token、GitHub PAT、SSH 密钥**绝不能**提交到 GitHub；`.gitignore` 已排除常见形式 |

## 本地验证（不依赖云端）

```bash
cd qscan-deploy/docs
python3 -m http.server 8081
# 浏览器打开 http://127.0.0.1:8081/?svc=http://127.0.0.1:8080
# 前提是本机或服务器上的识别服务已在 8080 运行
```

## 数据说明

上传的题本与扫描卷只在识别请求期间使用；模板缓存按内容 MD5 存于 `QSCAN_CACHE`（默认系统临时目录），
请求结束会清理临时文件。导出由浏览器直接下载，服务端不保存任何导出文件。
