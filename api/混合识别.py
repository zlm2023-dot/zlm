# -*- coding: utf-8 -*-
"""
混合识别 v2.0: 本地 OpenCV 秒级过一遍 + 云端视觉模型只复核 + 全量二次确认

v2.0 改进:
  1. 更智能的"拿不准"筛选 (归一化分数 + 形态学特征)
  2. 全量二次确认: OK 项做低成本确认 (每批只让模型答"对/错")
  3. 分阶段云端调用: 先复核 → 再确认 → 按需补漏
  4. 多 Key 轮换 + 更激进的限流退避
  5. 引擎并发改造 (多进程并行处理多份问卷)

用法
  python 混合识别.py --scan 扫描卷.pdf --book 题本.pdf
  python 混合识别.py --scan-dir 扫描目录 --book 题本.pdf --workers 2
  python 混合识别.py --scan-dir 扫描目录 --key <智谱Key>
  python 混合识别.py --scan-dir 扫描目录 --no-vlm              # 纯本地
"""
import argparse
import base64
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(HERE), "第一版")


def resource_dir():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(sys.executable)
    return HERE

DEFAULT_ENDPOINT = "http://127.0.0.1:8765/api/paas/v4/chat/completions"
DEFAULT_MODEL = "glm-4.6v-flash"


def load_local_engine():
    cands = [os.path.join(V1, "问卷识别工具.py"),
             os.path.join(resource_dir(), "问卷识别工具.py")]
    p = next((c for c in cands if os.path.exists(c)), None)
    if not p:
        raise FileNotFoundError("找不到本地引擎 问卷识别工具.py")
    spec = importlib.util.spec_from_file_location("local_engine", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ENGINE = None

import cv2
import numpy as np


def imwrite_utf8(path, img, params=None):
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def imread_utf8(path, flag=cv2.IMREAD_GRAYSCALE):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    except Exception:
        return None


cv2.imwrite = imwrite_utf8


def img_b64(path, max_side=900):
    im = imread_utf8(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode()


def call_vlm(endpoint, keys, model, text, imgs, max_tokens=2000, timeout=300, retries=3):
    content = [{"type": "text", "text": text}]
    for b in imgs:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}})
    body = {
        "model": model, "temperature": 0.01, "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": content}],
    }

    def _opener(url):
        if re.match(r"https?://(127\.0\.0\.1|localhost|\[::1\])", url):
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener()

    last = None
    for attempt in range(retries):
        is429 = False
        for ki, key in enumerate(keys):
            try:
                req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Authorization", "Bearer " + key)
                with _opener(endpoint).open(req, timeout=timeout) as r:
                    j = json.loads(r.read().decode())
                txt = j["choices"][0]["message"]["content"]
                if not str(txt).strip():
                    last = "返回为空"
                    continue
                return txt, ki
            except Exception as e:
                last = str(e)[:110]
                if re.search(r"429|Too Many|速率|限流|1302|1305", last):
                    is429 = True
        if attempt < retries - 1:
            wait = 25 * (attempt + 1) if is429 else 4
            print("    %s → 等待 %ds 后重试 (%d/%d)"
                  % ("429 限流" if is429 else "调用失败", wait, attempt + 1, retries - 1))
            time.sleep(wait)
    raise RuntimeError("全部 Key 均失败，最后错误：" + str(last))


def extract_json(t):
    t = str(t).strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("无 JSON")
    s = t[a:b + 1]
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = re.sub(r",\s*([}\]])", r"\1", s)
    s2 = s2.replace("\u201c", '"').replace("\u201d", '"')
    try:
        return json.loads(s2)
    except Exception:
        pass
    out = {}
    for m in re.finditer(r'\{[^{}]*?"id"\s*:\s*"([^"]+)"[^{}]*?"value"\s*:\s*("[^"]*"|null|-?\d+)[^{}]*\}', s):
        try:
            v = json.loads(m.group(2))
        except Exception:
            v = None
        out[m.group(1)] = v
    if out:
        return {"_salvaged": out}
    raise ValueError("JSON 解析失败")


def review_choices(tasks, endpoint, keys, model, batch=12):
    """tasks: [(qid, qtext, opt_text, crop_path)] -> {qid: (value, conf)}"""
    out = {}
    for i in range(0, len(tasks), batch):
        chunk = tasks[i:i + batch]
        b64s, metas = [], []
        for qid, qtext, opt, cp in chunk:
            b = img_b64(cp) if cp and os.path.exists(cp) else None
            if not b:
                continue
            b64s.append(b)
            metas.append((qid, qtext, opt))
        if not b64s:
            continue
        lines = []
        for n, (qid, qtext, opt) in enumerate(metas):
            lines.append("- 图%d | id=%s | 题干:%s | 选项:%s" % (n + 1, qid, qtext or "", opt or ""))
        prompt = (
            "下面是 %d 张问卷局部截图，每张图对应一道题的作答区域（已按顺序编号）。\n"
            "受访者通常用打勾✓、画圈○、涂黑●、打叉×、划线或在数字上做标记来表示选择；\n"
            "若有两个及以上选项有笔迹，取笔迹最重/最清晰的一个；完全无笔迹则填 null。\n\n"
            "%s\n\n"
            '严格输出 JSON：{"answers":[{"id":"题id","value":选项数字或null,"c":把握0~1}]}\n'
            "id 必须与上面给出的完全一致，只输出 JSON。"
            % (len(metas), "\n".join(lines))
        )
        try:
            txt, _ki = call_vlm(endpoint, keys, model, prompt, b64s, max_tokens=len(metas) * 40 + 400)
            obj = extract_json(txt)
            arr = obj.get("answers")
            if obj.get("_salvaged"):
                arr = [{"id": k, "value": v} for k, v in obj["_salvaged"].items()]
            got = 0
            for a in (arr or []):
                qid = str(a.get("id", ""))
                if qid and a.get("value") is not None:
                    out[qid] = (a.get("value"), a.get("c") if isinstance(a.get("c"), (int, float)) else 0.9)
                    got += 1
            print("    复核 %d 题 → 云端改写 %d 题" % (len(metas), got))
        except Exception as e:
            print("    复核失败（保留本地结果）：%s" % str(e)[:110])
    return out


def confirm_choices(tasks, endpoint, keys, model, batch=16):
    """
    v2.0 新增: 全量二次确认 (低成本)
    对本地识别为 OK 的题, 让云端判断 "对/错", 只在发现错时纠正.
    返回: {qid: corrected_value} (仅包含云端判错的题)
    """
    corrections = {}
    for i in range(0, len(tasks), batch):
        chunk = tasks[i:i + batch]
        b64s, metas = [], []
        for qid, local_val, qtext, opt, cp in chunk:
            b = img_b64(cp) if cp and os.path.exists(cp) else None
            if not b:
                continue
            b64s.append(b)
            metas.append((qid, local_val, qtext, opt))
        if not b64s:
            continue
        lines = []
        for n, (qid, local_val, qtext, opt) in enumerate(metas):
            lines.append("- 图%d | id=%s | 本地答案=%s | 题干:%s | 选项:%s"
                         % (n + 1, qid, local_val, qtext or "", opt or ""))
        prompt = (
            "下面是 %d 张问卷局部截图，每张图对应一道题的作答区域。\n"
            "每道题已给出本地识别的答案，请判断这个答案是否正确。\n"
            "判断标准: 受访者用打勾✓、画圈○、涂黑●、打叉×、划线等方式标记选择。\n"
            "如果本地答案正确, ok=true; 如果错误, 给出正确值 correct; 如果看不清填 ok=false correct=null。\n\n"
            "%s\n\n"
            '严格输出 JSON: {"checks":[{"id":"题id","ok":true或false,"correct":正确选项数字或null}]}\n'
            "只输出 JSON, 不输出其他内容。"
            % (len(metas), "\n".join(lines))
        )
        try:
            txt, _ki = call_vlm(endpoint, keys, model, prompt, b64s, max_tokens=len(metas) * 30 + 300)
            obj = extract_json(txt)
            arr = obj.get("checks") or obj.get("answers") or []
            if obj.get("_salvaged"):
                arr = [{"id": k, "ok": True, "correct": v} for k, v in obj["_salvaged"].items()]
            for a in arr:
                qid = str(a.get("id", ""))
                if not qid:
                    continue
                ok = a.get("ok")
                correct = a.get("correct")
                if ok is False and correct is not None:
                    corrections[qid] = correct
                elif isinstance(ok, str) and ok.lower() in ("false", "no", "wrong") and correct is not None:
                    corrections[qid] = correct
            print("    确认 %d 题 → 云端纠正 %d 题" % (len(metas), len([q for q in metas if q[0] in corrections])))
        except Exception as e:
            print("    确认失败（保留本地结果）：%s" % str(e)[:110])
    return corrections


_DIGIT_ANCHOR_RE = re.compile(r'学号|年龄|出生|年月|共有|排行|班|编号|人数|数量|年|答案')
_DIGIT_LEN_HINT = {'出生日期': (4, 4), '年': (1, 2), '月': (1, 2), '年龄': (1, 3),
                   '班级': (1, 3), '班别': (1, 4), '学号': (8, 12)}


def _digits_of(s):
    return ''.join(ch for ch in str(s or '') if ch.isdigit())


def _digit_len_hint(anchor):
    for k, rng in _DIGIT_LEN_HINT.items():
        if k in (anchor or ''):
            return rng
    return None


def review_fills(rows, endpoint, keys, model, crops_dir, batch=10):
    """表头填空 OCR"""
    out = {}
    items = []
    for r in rows:
        cp = os.path.join(crops_dir, r["截图"]) if r.get("截图") else ""
        if not os.path.exists(cp):
            continue
        items.append((r["字段"], (r.get("锚点文本") or ""), cp))
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        b64s, metas = [], []
        for name, anchor, cp in chunk:
            b = img_b64(cp, max_side=800)
            if not b:
                continue
            b64s.append(b)
            metas.append((name, anchor))
        if not b64s:
            continue
        lines = []
        for n, (nm, an) in enumerate(metas):
            hint = _digit_len_hint(an)
            extra = ''
            if _DIGIT_ANCHOR_RE.search(an or ''):
                if hint:
                    extra = ' | 应为%d~%d位数字' % hint
                else:
                    extra = ' | 应为纯数字'
            lines.append("- 图%d | id=%s | 提示文字:%s%s" % (n + 1, nm, an, extra))
        prompt = (
            "下面是 %d 张问卷表头填写处的小截图，每张对应一个待识别的填写项。\n"
            "请仔细辨认手写内容，输出你读到的原文（不要推测、不要补字）。\n"
            "数字类字段只输出数字本身，不要带「班/岁/年」等单位；\n"
            "看不清某一位就输出整项 null，不要猜。\n"
            "若确实空白或无法辨认，填 null。\n\n%s\n\n"
            '严格输出 JSON：{"fills":[{"id":"上面的id","value":"读到的内容或null"}]}\n'
            % (len(metas), "\n".join(lines))
        )
        try:
            txt, _ki = call_vlm(endpoint, keys, model, prompt, b64s, max_tokens=len(metas) * 40 + 400)
            obj = extract_json(txt)
            arr = obj.get("fills")
            if obj.get("_salvaged"):
                arr = [{"id": k, "value": v} for k, v in obj["_salvaged"].items()]
            got = 0
            unmatched = list(metas)
            for a in (arr or []):
                nm = str(a.get("id", "")).strip()
                if not nm or a.get("value") in (None, "", "null"):
                    continue
                out[nm] = str(a.get("value"))
                got += 1
            # 模型回显的 id 可能被改动/截断 —— 严格匹配失败的条目按「锚点相似度+顺序」兜底,
            # 否则云端明明读出了值却静默丢弃 (2026-11-02 广东第一页实测 6 项全丢)。
            import difflib
            left_keys = [k for k in out if k not in {m[0] for m in metas}]
            for k in left_keys:
                v = out.pop(k)
                best, bi = 0.0, None
                for mi, (nm, an) in enumerate(unmatched):
                    s = difflib.SequenceMatcher(None, k, nm).ratio()
                    if an and k and (an in k or k in an):
                        s = max(s, 0.9)
                    if s > best:
                        best, bi = s, (mi, nm)
                if bi and best >= 0.4:
                    out[bi[1]] = v
                    unmatched.pop(bi[0])
                else:
                    print("    警告: 云端填空返回 id=%r 无法对应本地字段, 该值已丢弃" % k[:40])
            matched = sum(1 for m in metas if m[0] in out)
            if matched < got:
                print("    警告: 云端读出 %d 项但仅 %d 项对上本地字段" % (got, matched))
            print("    填空 OCR %d 项 → 读出 %d 项, 回写 %d 项" % (len(metas), got, matched))
        except Exception as e:
            print("    填空 OCR 失败：%s" % str(e)[:110])
    return out


# ---------- A3 双页拼版 ----------
def looks_like_spread(path):
    try:
        if path.lower().endswith(".pdf"):
            import pymupdf
            d = pymupdf.open(path)
            r = d[0].rect
            ar = r.width / float(r.height)
            d.close()
            return ar > 1.15
        im = imread_utf8(path)
        if im is None:
            return False
        h, w = im.shape[:2]
        return w / float(h) > 1.15
    except Exception:
        return False


def _cut_two(img, name, workdir):
    h, w = img.shape[:2]
    lo, hi = int(w * 0.40), int(w * 0.60)
    bin_ = (img < 150).astype(np.uint8)
    colink = bin_.sum(axis=0).astype(np.float32)
    seg = colink[lo:hi]
    k = max(5, w // 300)
    sm = np.convolve(seg, np.ones(k) / k, mode="same")
    thr = sm.min() + max(3.0, (sm.max() - sm.min()) * 0.12)
    best_len, best_s, cur_s = 0, None, None
    for i, v in enumerate(sm):
        if v <= thr:
            if cur_s is None:
                cur_s = i
        else:
            if cur_s is not None:
                if i - cur_s > best_len:
                    best_len, best_s = i - cur_s, cur_s
                cur_s = None
    if cur_s is not None and len(sm) - cur_s > best_len:
        best_len, best_s = len(sm) - cur_s, cur_s
    if best_s is not None and best_len >= w * 0.02:
        cut = lo + best_s + best_len // 2
    else:
        cut = w // 2
    return [(tag, np.ascontiguousarray(img[:, x0:x1]))
            for tag, x0, x1 in (("L", 0, cut), ("R", cut, w))]


def expand_spread(path, workdir):
    os.makedirs(workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    imgs = []
    if path.lower().endswith(".pdf"):
        if not looks_like_spread(path):
            return [path]
        import pymupdf
        d = pymupdf.open(path)
        for i in range(d.page_count):
            pix = d[i].get_pixmap(matrix=pymupdf.Matrix(200 / 72.0, 200 / 72.0))
            im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2GRAY)
            elif pix.n == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                im = im.reshape(pix.height, pix.width)
            im = np.ascontiguousarray(im)
            h, w = im.shape[:2]
            if w / float(h) > 1.15:
                imgs += [sub for _, sub in _cut_two(im, "%s_p%02d" % (base, i + 1), workdir)]
            else:
                imgs.append(im)
        d.close()
    else:
        im = imread_utf8(path)
        if im is None:
            return [path]
        h, w = im.shape[:2]
        imgs = [sub for _, sub in _cut_two(im, base, workdir)] if w / float(h) > 1.15 else [im]
    if not imgs:
        return [path]
    import pymupdf
    out_pdf = os.path.join(workdir, base + "_split.pdf")
    doc = pymupdf.open()
    for im in imgs:
        page = doc.new_page(width=595, height=842)
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(out_pdf)
    doc.close()
    print("  拼版/图片输入 → 标准化 %d 页 A4 PDF" % len(imgs))
    return [out_pdf]


def merge_parts(parts, outdir, name):
    all_rows, all_fills = [], []
    dst_crops = os.path.join(outdir, name, "crops")
    os.makedirs(dst_crops, exist_ok=True)
    for p in parts:
        pt = os.path.splitext(os.path.basename(p))[0]
        src_crops = os.path.join(outdir, pt, "crops")
        for suffix, dst in (("识别结果", all_rows), ("填空题", all_fills)):
            fp = find_result_csv(outdir, pt, suffix)
            if not fp:
                continue
            for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
                cp = r.get("校对图") or r.get("截图") or ""
                if cp and os.path.exists(os.path.join(src_crops, cp)):
                    newcp = "%s_%s" % (pt, cp)
                    try:
                        shutil.copyfile(os.path.join(src_crops, cp), os.path.join(dst_crops, newcp))
                        if "校对图" in r:
                            r["校对图"] = newcp
                        else:
                            r["截图"] = newcp
                    except Exception:
                        pass
                dst.append(r)

    def qkey(r):
        m = re.match(r"p(\d+)", r.get("题目ID", ""))
        return (int(m.group(1)) if m else 9999, r.get("题目ID", ""))

    rank = {"OK": 0, "云端复核": 0, "云端确认": 0, "配准失败": 2, "未检出": 2}
    best = {}
    for r in all_rows:
        q = r["题目ID"]
        sc = rank.get(r.get("状态", ""), 1)
        if q not in best or sc < rank.get(best[q].get("状态", ""), 1):
            best[q] = r
    rows = [best[k] for k in sorted(best, key=lambda q: qkey(best[q]))]

    seen = set()
    fills = []
    for r in all_fills:
        if r.get("字段") in seen:
            continue
        seen.add(r.get("字段"))
        fills.append(r)

    csv_path = os.path.join(outdir, name + "_识别结果.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["题目ID", "题干", "识别答案", "状态", "墨迹分数", "校对图"])
        for r in rows:
            w.writerow([r["题目ID"], r.get("题干", ""), r.get("识别答案", ""),
                        r.get("状态", ""), r.get("墨迹分数", ""), r.get("校对图", "")])
    fill_path = os.path.join(outdir, name + "_填空题.csv")
    if fills:
        with open(fill_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["字段", "锚点文本", "截图", "状态", "识别/录入值"])
            for r in fills:
                w.writerow([r.get("字段", ""), r.get("锚点文本", ""), r.get("截图", ""),
                            r.get("状态", ""), r.get("识别/录入值", "")])
    print("  合并 %d 个单页 → %d 题 / %d 项填空" % (len(parts), len(rows), len(fills)))
    return rows, fills


# ---------- 单份处理 (v2.0 分阶段策略) ----------
def find_result_csv(outdir, name, suffix):
    """找 <name>_<suffix>.csv; 引擎写不进去时会落到 .bak。
    按修改时间取最新 —— 被占用的旧文件不会盖过刚写出来的 .bak。"""
    import glob as _glob
    cand = _glob.glob(os.path.join(outdir, "%s_%s.csv" % (name, suffix)))
    cand += _glob.glob(os.path.join(outdir, "%s_%s.csv.bak" % (name, suffix)))
    cand = [p for p in cand if os.path.isfile(p)]
    return max(cand, key=os.path.getmtime) if cand else None


def process_one(scan_path, tmpl, book, outdir, cfg):
    name = os.path.splitext(os.path.basename(scan_path))[0]
    t_all = time.time()
    print("  本地识别中…")
    t0 = time.time()
    parts = expand_spread(scan_path, os.path.join(outdir, "_spread"))
    for p in parts:
        ENGINE.recognize_scan(p, tmpl, book, outdir)
    t_local = time.time() - t0

    res_dir = os.path.join(outdir, name)
    crops_dir = os.path.join(res_dir, "crops")
    csv_path = find_result_csv(outdir, name, "识别结果")
    fill_path = find_result_csv(outdir, name, "填空题")

    if len(parts) > 1 or os.path.splitext(os.path.basename(parts[0]))[0] != name:
        rows, fills = merge_parts(parts, outdir, name)
    else:
        # 旧版直接 open(固定文件名): CSV 被 Excel/WPS 占用时引擎改写/跳过,
        # 这里就抛 FileNotFoundError → 整次识别白跑。现在找不到任何落盘文件
        # 才报结构化错误 (HTTP 层转成 4xx/友好提示, 不再 500)。
        if not csv_path:
            raise RuntimeError("识别已完成，但结果文件未能读取（可能被 Excel/WPS 占用）。"
                               "请关闭正在打开“%s_识别结果.csv”的程序后重试。" % name)
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
        fills = list(csv.DictReader(open(fill_path, encoding="utf-8-sig"))) if fill_path else []

    # --- 筛选云端复核候选 ---
    need_review = []   # 拿不准的: 未检出/多选/配准失败/低分OK
    ok_items = []      # 正常OK项: 全量二次确认
    optmap = cfg["optmap"]
    for r in rows:
        st = (r.get("状态") or "")
        sc = float(r.get("墨迹分数") or 0)
        qinfo = optmap.get(r["题目ID"], {})
        qtext = r.get("题干") or qinfo.get("qtext") or ""
        opt = "  ".join("%s:%s" % (o["value"], (o["text"] or "").strip())
                       for o in qinfo.get("options", []))
        cp = os.path.join(crops_dir, r.get("校对图", "")) if r.get("校对图") else ""

        if st != "OK" or (cfg.get("borderline", 0) > 0 and 0 < sc < cfg["borderline"]):
            # 拿不准的 → 直接复核 (让云端重新判)
            need_review.append((r["题目ID"], qtext, opt, cp))
        elif sc < cfg.get("confirm_norm", 2.5):
            # OK 但墨迹占比偏低 (边际带) → 二次确认; 高把握 OK 项不再全量送审
            ok_items.append((r["题目ID"], r.get("识别答案", ""), qtext, opt, cp))

    t_vlm = 0.0
    n_vlm = 0
    fixed_review = 0
    fixed_confirm = 0
    fixed_f = 0

    def vlm_left():
        return cfg["vlm_budget"] - t_vlm

    if cfg["keys"] and vlm_left() > 10:
        t0 = time.time()
        print("  云端阶段 1: 复核 %d 个拿不准的格子…" % len(need_review))
        if need_review and vlm_left() > 5:
            res = review_choices(need_review, cfg["endpoint"], cfg["keys"], cfg["model"], batch=cfg["batch"])
            for r in rows:
                if r["题目ID"] in res:
                    v, c = res[r["题目ID"]]
                    if str(r["识别答案"]) != str(v):
                        r["_old"] = r["识别答案"]
                        r["识别答案"] = v
                        fixed_review += 1
                    r["状态"] = "云端复核(%.2f)" % c if isinstance(c, float) else "云端复核"
            if res:
                n_vlm += 1

        t_mid = time.time() - t0
        if vlm_left() > 15:
            print("  云端阶段 2: 二次确认 %d 个低边际 OK 项(墨迹占比<%.1f%%)…" % (len(ok_items), cfg.get("confirm_norm", 2.5)))
            corrections = confirm_choices(ok_items, cfg["endpoint"], cfg["keys"], cfg["model"], batch=cfg["batch"] + 4)
            for r in rows:
                if r["题目ID"] in corrections:
                    correct_val = corrections[r["题目ID"]]
                    if str(r.get("识别答案")) != str(correct_val):
                        r["_old"] = r["识别答案"]
                        r["识别答案"] = correct_val
                        r["状态"] = "云端纠正"
                        fixed_confirm += 1
            if corrections:
                n_vlm += 1

        if cfg["fill_ocr"] and fills and vlm_left() > 5:
            print("  云端阶段 3: OCR 表头填空 %d 项…" % len(fills))
            fr = review_fills(fills, cfg["endpoint"], cfg["keys"], cfg["model"], crops_dir)
            # 双源融合: 本地(数字模型+macOS Vision) 与 云端 VLM 独立读数。
            #   一致 → 双引擎互相印证, 自动采纳;
            #   不一致 → 采纳云端值但保留双方读数, 强制人工核对 (广东卷实测:
            #   本地班级 3→5、云端学号两次三样 —— 单一来源都有静默错读风险);
            #   云端 null → 保留本地结果原状态;
            #   位数先验不符 (学号<8位等) → 无论谁读的都必核。
            for r in fills:
                cv_ = fr.get(r["字段"])
                if cv_ in (None, "", "null"):
                    continue
                anchor = r.get("锚点文本") or ""
                local_v = _digits_of(r.get("识别/录入值") or "")
                cloud_v = str(cv_).strip()
                fixed_f += 1
                if _DIGIT_ANCHOR_RE.search(anchor):
                    cd = _digits_of(cloud_v)
                    hint = _digit_len_hint(anchor)
                    hint_ok = bool(cd and (not hint or hint[0] <= len(cd) <= hint[1]))
                    if not hint_ok:
                        r["识别/录入值"] = cd or cloud_v
                        r["状态"] = "云端OCR位数异常,请核对"
                    elif local_v and cd == local_v:
                        r["识别/录入值"] = cd
                        r["状态"] = "OK(双源一致)"
                    elif local_v:
                        r["识别/录入值"] = cd
                        r["状态"] = "低置信(本地%s/云端%s),请核对" % (local_v, cd)
                    else:
                        r["识别/录入值"] = cd
                        r["状态"] = "云端OCR,请核对"
                else:
                    r["识别/录入值"] = cloud_v
                    r["状态"] = "云端OCR,请核对"
            if fr:
                n_vlm += 1

        t_vlm = time.time() - t0

    # 回写（写不进去时退到 .bak，不因文件被占用丢掉整次结果）
    wpath = write_back = None
    for wpath in (csv_path, (csv_path + ".bak") if csv_path else None):
        try:
            with open(wpath, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["题目ID", "题干", "识别答案", "状态", "墨迹分数", "校对图"])
                for r in rows:
                    w.writerow([r["题目ID"], r["题干"], r["识别答案"], r["状态"], r["墨迹分数"], r["校对图"]])
            write_back = wpath
            break
        except (OSError, TypeError):
            continue
    if write_back != csv_path:
        print("  提示: %s 被占用, 云端复核结果写入 %s" % (csv_path, write_back))
    if fills and fill_path:
        try:
            with open(fill_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["字段", "锚点文本", "截图", "状态", "识别/录入值"])
                for r in fills:
                    w.writerow([r["字段"], r["锚点文本"], r["截图"], r["状态"], r["识别/录入值"]])
        except OSError:
            with open(fill_path + ".bak", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["字段", "锚点文本", "截图", "状态", "识别/录入值"])
                for r in fills:
                    w.writerow([r["字段"], r["锚点文本"], r["截图"], r["状态"], r["识别/录入值"]])

    total = time.time() - t_all
    stat = {
        "文件": name, "题数": len(rows), "填空数": len(fills),
        "本地耗时": round(t_local, 1), "云端耗时": round(t_vlm, 1),
        "云端请求次数": n_vlm, "总耗时": round(total, 1),
        "复核": len(need_review), "二次确认": len(ok_items),
        "云端改写(复核)": fixed_review, "云端纠正(确认)": fixed_confirm,
        "云端读出填空": fixed_f,
        "未作答": sum(1 for r in rows if not str(r["识别答案"]).strip()),
        "可疑": sum(1 for r in rows if "OK" not in (r["状态"] or "")),
    }
    print("  ✔ %s  本地%.1fs + 云端%.1fs(%d次) = %.1fs | 未作答%d 可疑%d | 云端: 复核改%d 确认改%d"
          % (name, t_local, t_vlm, n_vlm, total, stat["未作答"], stat["可疑"], fixed_review, fixed_confirm))
    return stat


def main():
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:
        print("=" * 56)
        print("  问卷混合识别 v2.0")
        print("=" * 56)
        sd = input("扫描卷文件夹（回车 = 本程序所在目录）：").strip().strip('"') or os.path.dirname(sys.executable)
        bk = input("题本 PDF 完整路径：").strip().strip('"')
        key = input("智谱Key（多个用逗号分隔；回车 = 纯本地不联网）：").strip()
        argv = ["--scan-dir", sd, "--book", bk] + (["--key", key] if key else ["--no-vlm"])
        sys.argv = [sys.executable] + argv

    ap = argparse.ArgumentParser(description="混合识别 v2.0")
    ap.add_argument("--scan")
    ap.add_argument("--scan-dir")
    ap.add_argument("--book", required=True)
    _def_out = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else HERE
    ap.add_argument("--outdir", default=os.path.join(_def_out, "识别结果"))
    ap.add_argument("--key", default=os.environ.get("ZHIPU_KEY", ""))
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--borderline", type=float, default=0)
    ap.add_argument("--vlm-budget", type=float, default=90)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-fill-ocr", action="store_true")
    a = ap.parse_args()

    global ENGINE
    ENGINE = load_local_engine()

    scans = []
    if a.scan:
        scans.append(a.scan)
    if a.scan_dir:
        book_abs = os.path.abspath(a.book)
        for f in sorted(os.listdir(a.scan_dir)):
            if not f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                continue
            p = os.path.join(a.scan_dir, f)
            if os.path.abspath(p) == book_abs or "题本" in f:
                continue
            scans.append(p)
    if not scans:
        print("未指定 --scan 或 --scan-dir")
        return 1

    os.makedirs(a.outdir, exist_ok=True)

    tmpl_path = os.path.join(a.outdir, "template.json")
    t0 = time.time()
    if os.path.exists(tmpl_path):
        tmpl = json.load(open(tmpl_path, encoding="utf-8"))
        print("模板已缓存: %d 页" % len(tmpl["pages"]))
    else:
        print("首次构建模板（仅一次）…")
        ENGINE.build_template(a.book, tmpl_path)
        tmpl = json.load(open(tmpl_path, encoding="utf-8"))
        print("模板完成 %.1fs, %d 页" % (time.time() - t0, len(tmpl["pages"])))

    optmap = {}
    for p in tmpl["pages"]:
        for q in p.get("questions", []):
            optmap[q["id"]] = q

    cfg = {
        "keys": [] if a.no_vlm else [k.strip() for k in a.key.replace(chr(10), ",").split(",") if k.strip()],
        "endpoint": a.endpoint, "model": a.model, "batch": a.batch,
        "borderline": a.borderline, "optmap": optmap,
        "review": True, "fill_ocr": not a.no_fill_ocr, "vlm_budget": a.vlm_budget,
    }

    t_all = time.time()
    stats = []
    for s in scans:
        print("[%s]" % os.path.basename(s))
        stats.append(process_one(s, tmpl, a.book, a.outdir, cfg))

    wall = time.time() - t_all
    n = len(stats)
    print("\n" + "=" * 62)
    print("★ 混合识别 v2.0 汇总：%d 份，总墙钟 %.1f 秒" % (n, wall))
    print("  平均每份 %.1f 秒" % (wall / n))
    if any(s["云端请求次数"] > 0 for s in stats):
        print("  云端阶段: 复核 %.0f 题, 二次确认 %.0f 题, 纠正 %.0f 题"
              % (sum(s["复核"] for s in stats), sum(s["二次确认"] for s in stats),
                 sum(s["云端纠正(确认)"] for s in stats)))
    print("  平均未作答 %.1f 题，平均可疑 %.1f 题"
          % (sum(s["未作答"] for s in stats) / n, sum(s["可疑"] for s in stats) / n))
    print("  输出目录:", a.outdir)
    json.dump({"wall": round(wall, 1), "per_scan": round(wall / n, 1), "stats": stats},
              open(os.path.join(a.outdir, "混合识别_耗时统计.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
