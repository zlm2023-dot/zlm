/*
 * 识别引擎 API 地址配置 —— 部署到 Cloudflare Pages 时必须填写
 *
 * 网页是纯静态文件（Cloudflare Pages / GitHub Pages 托管），
 * 识别引擎是 Python + OpenCV 服务，必须单独部署，并用一个 https 域名对外提供。
 *
 * 部署时把下面引号里改成你的 API 域名，例如：
 *   window.QSCAN_API = 'https://api.yourdomain.com';
 *
 * 注意事项：
 *   · 必须以 https:// 开头（网页也是 https，http 会被浏览器当作混合内容拦截）
 *   · 结尾不要带斜杠，不要带 /ping 或 /api/recognize
 *   · 改完 commit + push，Cloudflare Pages 会自动重新部署
 *   · 留空 = 与网页同源（适合单进程部署 / 本地预览）
 *   · 临时覆盖：访问 https://你的网页/?svc=https://另一个API域名 即可，不用改文件
 *
 * 这个文件只放域名，不要放任何密钥、Token 或密码。
 */
window.QSCAN_API = '';
