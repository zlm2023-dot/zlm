# -*- coding: utf-8 -*-
"""
混合识别服务：把「混合识别.py」的本地 OpenCV 引擎挂成 HTTP 接口，
供《问卷扫描卷识别与核验系统2.html》网页直接调用。

接口
  GET  /ping            → {"ok":true, "engine":true, "template":"md5xxxx" 或 null}
  POST /api/recognize   → multipart 表单：
        scan     扫描卷文件（PDF/图片，可多个，多个会自动合并成一份）
        book     题本 PDF（首次建模板需要，之后可省略）
        structure 网页题本结构 JSON（S.tpl，用于把答案对回网页题号）
        options   JSON 字符串 {keys:"key1,key2", vlm_budget:90, batch:12}
      返回 {"ok":true, "answers":{网页题号:{value,conf,note}}, "meta":{表头字段:值},
            "unmatched":[...], "extra_fills":[...], "stats":{...}}

只监听 127.0.0.1，数据不出本机（除非填了智谱 Key，云端仅复核拿不准的格子）。

启动：双击「启动本地识别服务.bat」，或
  python 混合识别服务.py
"""
import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import time
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.environ.get("QSCAN_WEB", os.path.join(os.path.dirname(HERE), "web"))
HOST = os.environ.get("QSCAN_HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
CACHE = os.environ.get("QSCAN_CACHE", os.path.join(tempfile.gettempdir(), "qscan_online_cache"))
os.makedirs(CACHE, exist_ok=True)
ACTIVE_FILE = os.path.join(CACHE, "active_template.txt")

# ---------- 载入混合识别引擎 ----------
_spec = importlib.util.spec_from_file_location("hybrid", os.path.join(HERE, "混合识别.py"))
hybrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hybrid)

ENGINE = hybrid.load_local_engine()
hybrid.ENGINE = ENGINE          # process_one 内部用 hybrid.ENGINE
print("[服务] 本地引擎加载完成")

# v2.0: 每请求独立临时目录，引擎在函数栈内运行，不再需要全局锁
# (题本模板缓存仍通过 _BOOK_CACHE 模块级共享，进程内单例)

# ---------- 文本匹配：本地题号 <-> 网页题号 ----------
def _norm(s):
    s = str(s or "")
    s = re.sub(r"^[\s\(（]*\d{1,2}\s*[.、．)）]?\s*", "", s)
    s = re.sub(r"[\s，。：:；;、？?！!（）()\[\]【】._\-~—·\"'“”‘’]", "", s)
    return s

def _sim(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92 + 0.06 * min(len(a), len(b)) / max(len(a), len(b))
    try:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0

def _pair_score(h, q):
    """网页题 ↔ 模板题的匹配分。

    题干是主证据；选项文字只能当加权辅证——同一份问卷里选项高度重复
    （「完全不符合/比较不符合/…」在几十道题里一模一样），若让选项分覆盖
    题干分，同选项数的题会全部拿到 1.0，配对就退化成随机乱点。
    """
    sq = _sim(h.get("stem"), q.get("qtext") or "")
    ho, qo = h.get("options") or [], q.get("options") or []
    s = sq
    if ho and qo:
        so = _sim(" ".join((o.get("label") or "") for o in ho),
                  " ".join((o.get("text") or "") for o in qo))
        if len(ho) == len(qo):
            so = max(so, sum(_sim(ho[i].get("label"), qo[i].get("text"))
                             for i in range(len(ho))) / len(ho))
        s = 0.72 * sq + 0.28 * so
        if len(ho) == len(qo):
            s += 0.04
    return min(1.0, s)

def _match_page(hs, lqs, th=0.5):
    """页内乱序最优配对。

    不能假设网页题单与模板题的页内顺序一致：网页 items 按「大题号→小题号」
    排序，模板按页面从上到下的 y 顺序排；一道大题跨页时（如第 4 题从上一页
    延续到本页上半部），两边顺序会整体错开，顺序对齐会把整页配对带偏。
    这里改为按相似度全局贪心配对，只在分数打平时用相对位置做 tie-break。
    """
    n, m = len(hs), len(lqs)
    if not n or not m:
        return []
    S = [[_pair_score(h, q) for q in lqs] for h in hs]
    cands = []
    for i in range(n):
        for j in range(m):
            if S[i][j] >= th:
                # 同分时，优先选「相对位置接近」的那对
                pos = abs((i / max(1, n - 1)) - (j / max(1, m - 1))) if n > 1 and m > 1 else 0.0
                cands.append((-S[i][j], pos, i, j))
    cands.sort()
    used_h, used_l, out = set(), set(), []
    for negs, _pos, i, j in cands:
        if i in used_h or j in used_l:
            continue
        out.append((i, j, -negs, "main"))
        used_h.add(i)
        used_l.add(j)
    # ---- 二段式: 主阈值没配上的题, 在「同样没配上」的剩余题里放宽再配 ----
    # 典型场景: 题干被截断/重建不完整(「2在周末通常情况下你平均每.，，．．」),
    # 相似度 0.48 差一点到主阈值, 但它就是正确的那一对——不补配就会整题漏答。
    left_h = [i for i in range(n) if i not in used_h]
    left_l = [j for j in range(m) if j not in used_l]
    if left_h and left_l:
        cands2 = []
        for i in left_h:
            for j in left_l:
                if S[i][j] >= 0.4:
                    cands2.append((-S[i][j], i, j))
        cands2.sort()
        for negs, i, j in cands2:
            if i in used_h or j in used_l:
                continue
            out.append((i, j, -negs, "relax"))
            used_h.add(i)
            used_l.add(j)
        # 最后剩一对一且选项数一致 → 按序补齐(如题干为空的模板行)
        left_h = [i for i in range(n) if i not in used_h]
        left_l = [j for j in range(m) if j not in used_l]
        if len(left_h) == 1 and len(left_l) == 1:
            i, j = left_h[0], left_l[0]
            no_h = len((hs[i] or {}).get("options") or [])
            no_q = len((lqs[j] or {}).get("options") or [])
            if S[i][j] >= 0.2 and no_h and no_h == no_q:
                out.append((i, j, S[i][j], "forced"))
    return out

def _align(hs, lqs, gap=-0.45, fallback_th=0.62):
    """Needleman-Wunsch 顺序对齐 + 乱序最佳匹配回退，返回 [(h下标, l下标, 相似度)]。
    顺序对齐后仍未配上的题，在同页剩余题里按相似度贪心配对（≥fallback_th 才采纳），
    用于抵御网页题单与模板题序不一致导致的整段错位。"""
    n, m = len(hs), len(lqs)
    if not n or not m:
        return []
    S = [[_pair_score(hs[i], lqs[j]) for j in range(m)] for i in range(n)]
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    BT = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + gap
        BT[i][0] = 1
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + gap
        BT[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = D[i - 1][j - 1] + S[i - 1][j - 1]
            bt = 0
            if D[i - 1][j] + gap > best:
                best, bt = D[i - 1][j] + gap, 1
            if D[i][j - 1] + gap > best:
                best, bt = D[i][j - 1] + gap, 2
            D[i][j], BT[i][j] = best, bt
    out, i, j = [], n, m
    while i > 0 and j > 0:
        bt = BT[i][j]
        if bt == 0:
            out.append((i - 1, j - 1, S[i - 1][j - 1]))
            i -= 1
            j -= 1
        elif bt == 1:
            i -= 1
        else:
            j -= 1
    out.reverse()
    # ---- 乱序回退：顺序对齐漏掉的题，在同页剩余题里按分数贪心配对 ----
    paired_h = {hi for hi, _, _ in out}
    paired_l = {li for _, li, _ in out}
    cands = []
    for hi in range(n):
        if hi in paired_h:
            continue
        for li in range(m):
            if li in paired_l:
                continue
            if S[hi][li] >= fallback_th:
                cands.append((S[hi][li], hi, li))
    cands.sort(reverse=True)
    for s, hi, li in cands:
        if hi in paired_h or li in paired_l:
            continue
        out.append((hi, li, s))
        paired_h.add(hi)
        paired_l.add(li)
    out.sort(key=lambda t: t[0])
    return out

def map_to_structure(rows, fills, structure, tmpl):
    """本地识别结果(题目ID p2t002 格式) → 网页题号(S1B1T3 格式)。
    rows: 识别结果CSV行; fills: 填空题CSV行; structure: S.tpl {items, metaFields};
    tmpl: 本地引擎模板（提供每题题干/选项/出现顺序）"""
    # 本地题按「题本页 + 模板出现顺序」组织（不能按 ID 排序：要保模板顺序）
    rowmap = {r.get("题目ID", ""): r for r in rows}
    loc = {}
    for p in sorted(tmpl.get("pages", []), key=lambda x: x.get("page", 0)):
        for q in p.get("questions", []):
            if q["id"] in rowmap:
                loc.setdefault(p.get("page", 0), []).append(q)
    lfills = {}
    for f in fills:
        m = re.match(r"p(\d+)_fill", f.get("字段", ""))
        if m:
            lfills.setdefault(int(m.group(1)), []).append(f)

    lqmap = {}      # 本地题ID -> {qtext, options, page}
    lfillmap = {}   # 填空字段名 -> {anchor, page}
    for p in tmpl.get("pages", []):
        for q in p.get("questions", []):
            lqmap[q["id"]] = q
        for f in p.get("fills", []):
            lfillmap[f["name"]] = f

    answers, unmatched, extra = {}, [], []
    html = {}
    for it in structure.get("items", []):
        html.setdefault(it.get("page") or 0, []).append(it)

    n_hi = n_ok = 0
    for pg in sorted(html):
        hs = [x for x in html[pg] if x.get("type") != "fill"]
        # loc 里已是模板题（按模板出现顺序），挂上识别结果行
        lqs = []
        for q in loc.get(pg, []):
            q2 = dict(q)
            q2["_row"] = rowmap.get(q["id"], {})
            lqs.append(q2)
        n_hi += len(hs)
        pairs = _match_page(hs, lqs)
        paired_h = {hi for hi, _, _, _ in pairs}
        for hi, li, sc, src in pairs:
            h, q = hs[hi], lqs[li]
            row = q.get("_row", {})
            nopts_h = len(h.get("options") or [])
            nopts_q = len(q.get("options") or [])
            if sc >= 0.5 or (sc >= 0.3 and nopts_h and nopts_h == nopts_q):
                # conf 只表达「答案本身可不可信」；题干匹配分只影响 note，
                # 不再把答案压进必核档（题号对齐问题是页面级的，不逐题刷）。
                # 二段式/兜底配对的双方都是各自仅剩的候选（按排除法唯一），
                # 配对结果不因文字相似度低而变差，答案置信不降档。
                conf = 0.92 if (sc >= 0.62 or src in ("relax", "forced")) else 0.65
                st = row.get("状态", "")
                blank = (st == "未检出")
                if blank:
                    conf = 0.9       # 引擎确认的空白(墨迹占比<0.6%)是高置信结论, 不是"看不清"
                elif st == "低墨迹待核":
                    conf = min(conf, 0.65)   # 边际带: 有唯一明确墨迹只是淡 → 提示级, 不进必核
                elif st == "涂改已跳过":
                    # 检测到涂抹的墨团并已改用另一格的勾 → 自动更正。
                    # 替勾本身墨迹充足(≥低墨迹带) → 高置信; 替勾也淡 → 保持提示级
                    try:
                        sc_ink = float(row.get("墨迹分数", "") or 0)
                    except Exception:
                        sc_ink = 0.0
                    conf = min(conf, 0.85 if sc_ink >= 1.2 else 0.6)
                elif st == "疑似涂改待核":
                    conf = min(conf, 0.4)    # 只有涂抹痕迹、没有重勾的答案 → 必核
                elif st not in ("OK", "") and "云端" not in st:
                    conf = min(conf, 0.4)          # 配准失败/多选可疑等 → 必核
                try:
                    v = int(str(row.get("识别答案", "")).strip())
                except Exception:
                    v = str(row.get("识别答案", "")).strip() or None
                if blank:
                    # 真空白是正常数据(未作答)，note 带「空白」让网页端降为提示而非漏答必核
                    note = "本地未检出(空白)"
                else:
                    note = ("本地" + (st or "OK")) + ("" if sc >= 0.62 else ";题干匹配分低")
                answers[h["id"]] = {
                    "value": v,
                    "conf": round(max(0.05, min(1.0, conf)), 2),
                    "note": note
                }
                n_ok += 1
            else:
                unmatched.append(h["id"])
        # 本页没配上对的题也要明说：静默丢失比标「未覆盖」更糟
        for hi, h in enumerate(hs):
            if hi not in paired_h:
                unmatched.append(h["id"])

    # 表头字段：按锚点文本匹配本地填空
    meta = {}
    meta_best = {}   # key -> 最佳匹配的本地填空行（供 covered 判定复用）
    meta_status = {}  # key -> 本地填空行的状态（双源一致 → 表头识别 conf 升档）
    for mf in structure.get("metaFields", []):
        key = mf.get("key", "")
        best, bv = 0.0, None
        for f in fills:
            fm = lfillmap.get(f.get("字段", ""), {})
            s = _sim(key, fm.get("anchor") or f.get("锚点文本") or "")
            if s > best:
                best, bv = s, f
        meta_best[key] = bv if best >= 0.5 else None
        if best >= 0.5 and bv:
            val = str(bv.get("识别/录入值", "") or "").strip()
            if val and val.lower() != "null":
                meta[key] = val
                meta_status[key] = str(bv.get("状态", "") or "")

    # ---- 网页里的「填空题」(type=fill)：表头项走 meta，正文填空走本地同页 fill ----
    mkeys = [mf.get("key", "") for mf in structure.get("metaFields", []) if mf.get("key")]
    used_fill = set()
    for pg in sorted(html):
        for it in [x for x in html[pg] if x.get("type") == "fill"]:
            stem = str(it.get("stem") or "")
            iid = it["id"]

            # ① 题干里点了名要哪几个表头字段（学号/班级/姓名…）→ 按出现顺序拼 meta 值
            hit = sorted([(stem.find(k), k) for k in mkeys if k and k in stem],
                         key=lambda t: t[0])
            parts = [meta[k] for _, k in hit if meta.get(k)]
            if parts:
                hit_sts = [meta_status.get(k, "") for _, k in hit if meta.get(k)]
                dual = hit_sts and all("双源一致" in s or "OK" in s for s in hit_sts)
                answers[iid] = {"value": " ".join(parts),
                                "conf": 0.92 if dual else 0.7,
                                "note": "表头识别(" + "/".join(k for _, k in hit if meta.get(k)) + ")，请核对"}
                n_ok += 1
                continue

            # ② 否则找本页 anchor 最像的本地填空（如「我的答案是：」）
            best, bv, banchor = 0.0, None, ""
            for f in lfills.get(pg, []):
                fm = lfillmap.get(f.get("字段", ""), {})
                a = fm.get("anchor") or f.get("锚点文本") or ""
                sc = _sim(stem, a) if a else 0.0
                if sc > best:
                    best, bv, banchor = sc, f, a
            if bv is not None and best >= 0.6:
                used_fill.add(bv.get("字段", ""))
                val = str(bv.get("识别/录入值", "") or "").strip()
                stt = str(bv.get("状态", "") or "")
                if val and val.lower() != "null":
                    if "双源一致" in stt:
                        # 本地(数字模型/Vision)与云端 VLM 独立读数相同 → 自动采纳
                        answers[iid] = {"value": val, "conf": 0.92,
                                        "note": "填空(" + banchor.strip("：: ") + ")，双源一致"}
                    else:
                        low = ("低置信" in stt) or ("请核对" in stt) or ("请对照" in stt)
                        # 本地未标低置信的填空(数字模型高把握) → 0.82 直接采纳;
                        # 标了低置信的 → 0.6 保持提示, 仍需人工核对
                        answers[iid] = {"value": val, "conf": 0.6 if low else 0.82,
                                        "note": "填空(" + banchor.strip("：: ") + ")" + ("，本地低置信请核对" if low else "")}
                    n_ok += 1
                else:
                    # 有墨迹但本地没读出来（多为中文）→ 标 0 置信，进核验清单
                    answers[iid] = {"value": None, "conf": 0.05,
                                    "note": "填空未识别(" + banchor.strip("：: ") + ")，请人工录入"}
            else:
                unmatched.append(iid)

    # 本地识别到、但网页结构里没有对应题的填空（如正文中的填空），列出来供人工查看。
    # covered = 值真正进了 meta/题单 的填空。注意不能按「相似度≥0.5 就算覆盖」：
    # 「性别」与「班别」共字「别」sim=0.5，会把班别的识别值吞掉（2026-11-02 修）。
    covered = set(used_fill)
    for bv in meta_best.values():
        if bv is not None:
            val = str(bv.get("识别/录入值", "") or "").strip()
            if val and val.lower() != "null":
                covered.add(bv.get("字段"))
    for f in fills:
        if f.get("字段") not in covered:
            val = str(f.get("识别/录入值", "") or "").strip()
            st = str(f.get("状态", "") or "")
            if val and val.lower() != "null":
                extra.append({"字段": f.get("字段", ""), "值": val})
            elif st:
                # 有槽位但没有读出值（多为手写中文）→ 如实上报待人工录入，
                # 不能静默丢弃：否则姓名/性别这类填空在网页端完全不可见
                extra.append({"字段": f.get("字段", ""), "值": "", "状态": st})

    return answers, meta, unmatched, extra, {"结构题数": n_hi, "已匹配": n_ok}


def build_boxes(tmpl, rows, book_path, reg=None):
    """识别概览数据：每题全部选项的 rect + 本地识别的选中值与状态。
    绿=识别的学生选项(状态OK)、黄=可疑/待核的选中项与填空区、红=其余已识别选项。

    reg: 引擎输出的「页面配准」信息(扫描像素→模板像素 仿射 + 每题实测格子)。
    有它就把格子换算回**扫描页坐标**再返回——识别框才会贴住扫描件上的真实格子；
    没有就退回题本坐标(旧行为，扫描件与题本有偏移时框会整体错位)。"""
    dims = {}
    try:
        import pymupdf
        if book_path and os.path.exists(book_path):
            doc = pymupdf.open(book_path)
            for i, pg in enumerate(doc, 1):
                dims[i] = [round(pg.rect.width, 1), round(pg.rect.height, 1)]
    except Exception:
        pass

    # 模板页 → (扫描页号, 模板px→扫描px 仿射(=M的逆), 扫描页 pt 尺寸, SCALE)
    # 注意：引擎 align_scan 返回的 M 是 cv2.warpAffine 正向语义 = 「扫描px→模板px」
    # (aligned = warpAffine(scan, M) 是把扫描件拉回模板帧)。因此题本/aligned 坐标
    # 映射回扫描件必须用 M 的逆变换 —— 照片件 M 平移可达百余px, 方向错了框会整体
    # 飞出页面 (2026-11-02 SKM 卷概览框错位根因); 清晰件 M≈单位阵才一直没暴露。
    p2s = {}
    if reg:
        for sp, info in (reg.get("pages") or {}).items():
            try:
                M = [[float(v) for v in r] for r in info["M"]]
                a, b, c = M[0]
                d, e, f = M[1]
                det = a * e - b * d
                if abs(det) < 1e-9:
                    continue
                Minv = [[e / det, -b / det, (b * f - e * c) / det],
                        [-d / det, a / det, (d * c - a * f) / det]]
                p2s[int(info["tpl_page"])] = (int(sp), Minv, info.get("size_pt"),
                                              float(info.get("scale") or 1.0))
            except Exception:
                continue
    reg_cells = (reg or {}).get("cells") or {}

    def to_scan_page(pno, rect):
        """题本 pt 坐标 → 扫描页 pt 坐标（M 逆变换: 模板px→扫描px）"""
        if pno not in p2s:
            return [round(float(v), 1) for v in rect]
        _sp, M, _size, sc = p2s[pno]
        xs, ys = [], []
        for x, y in ((rect[0], rect[1]), (rect[2], rect[3])):
            px, py = x * sc, y * sc
            nx = M[0][0] * px + M[0][1] * py + M[0][2]
            ny = M[1][0] * px + M[1][1] * py + M[1][2]
            xs.append(nx / sc)
            ys.append(ny / sc)
        return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]

    rowmap = {r.get("题目ID", ""): r for r in rows}
    out = []
    for p in tmpl.get("pages", []):
        pno = p.get("page", 0)
        tuned = pno in p2s and p2s[pno][2]
        items = []
        for q in p.get("questions", []):
            row = rowmap.get(q["id"], {})
            cells = reg_cells.get(q["id"])
            rects = cells if (cells and len(cells) == len(q.get("options", []))) \
                else [o["rect"] for o in q.get("options", [])]
            items.append({
                "id": q["id"],
                "vals": [o.get("value") for o in q.get("options", [])],
                "opts": [to_scan_page(pno, r) for r in rects],
                "picked": str(row.get("识别答案", "") or "").strip(),
                "st": row.get("状态", "") or ""})
        for f in p.get("fills", []):
            items.append({"fill": 1, "id": f.get("name", ""),
                          "rect": to_scan_page(pno, f.get("rect", [0, 0, 0, 0])),
                          "anchor": f.get("anchor", "")})
        pg_dims = (p2s[pno][2] if tuned else dims.get(pno))
        out.append({"page": pno, "dims": pg_dims, "aligned": bool(tuned), "items": items})
    return out




# ---------- 模板管理 ----------
def active_md5():
    try:
        return open(ACTIVE_FILE, encoding="utf-8").read().strip() or None
    except Exception:
        return None

def load_active_template():
    md5 = active_md5()
    if not md5:
        return None, None
    p = os.path.join(CACHE, "template_%s.json" % md5)
    if not os.path.exists(p):
        return md5, None
    return md5, json.load(open(p, encoding="utf-8"))

def _engine_fingerprint():
    """引擎源码指纹：问卷识别工具.py / 混合识别.py 任一变动都会变。
    模板缓存按 md5 永久存盘，但引擎修了漏题算法后旧模板必须重建，
    否则网页端会一直用到缺题的旧模板（2026-11-01 15 条漏答的根因）。"""
    xs = []
    for fn in ("问卷识别工具.py", "混合识别.py"):
        p = os.path.join(HERE, fn)
        try:
            st = os.stat(p)
            xs.append("%s:%d:%d" % (fn, st.st_mtime_ns, st.st_size))
        except OSError:
            pass
    return hashlib.md5("|".join(xs).encode("utf-8")).hexdigest()[:12]


_TPL_LOCK = threading.Lock()


def ensure_template(book_bytes):
    """有 book_bytes 就按 md5 建/取模板；没有则用当前激活模板。返回 (md5, tmpl, book_path)

    ThreadingHTTPServer 下两个请求同时首建同一本书的模板时, 旧实现会并发
    写同一个 template_*.json（中途读到半截 JSON）并互相覆盖 ACTIVE_FILE。
    构建加锁串行化（构建本身很罕见, 全局一把锁即可）, 且先写临时文件再
    原子替换, 并发读者永远读到完整 JSON。"""
    if book_bytes:
        md5 = hashlib.md5(book_bytes).hexdigest()[:12]
        lock = _TPL_LOCK  # 同一本书的构建互斥（md5 级别的锁意义不大, 构建很罕见）
        with lock:
            p = os.path.join(CACHE, "template_%s.json" % md5)
            book_path = os.path.join(CACHE, "题本_%s.pdf" % md5)
            if not os.path.exists(book_path):
                with open(book_path, "wb") as f:
                    f.write(book_bytes)
            ver_path = os.path.join(CACHE, "template_%s.ver" % md5)
            ver_now = _engine_fingerprint()
            ver_ok = (os.path.exists(p) and os.path.exists(ver_path)
                      and open(ver_path, encoding="utf-8").read().strip() == ver_now)
            if ver_ok:
                tmpl = json.load(open(p, encoding="utf-8"))
                print("[服务] 模板已缓存 %s（%d 页）" % (md5, len(tmpl["pages"])))
            else:
                if os.path.exists(p):
                    print("[服务] 引擎已更新，重建模板 %s …" % md5)
                t0 = time.time()
                print("[服务] 首次构建模板…（约1分钟，仅一次）")
                # 先写临时文件再原子替换: 并发读者不会读到半截 JSON
                tmp_p = p + ".building"
                try:
                    ENGINE.build_template(book_path, tmp_p)
                    os.replace(tmp_p, p)
                finally:
                    if os.path.exists(tmp_p):
                        try:
                            os.remove(tmp_p)
                        except OSError:
                            pass
                tmpl = json.load(open(p, encoding="utf-8"))
                with open(ver_path, "w", encoding="utf-8") as f:
                    f.write(ver_now)
                print("[服务] 模板完成 %.1fs，%d 页" % (time.time() - t0, len(tmpl["pages"])))
            with open(ACTIVE_FILE, "w", encoding="utf-8") as f:
                f.write(md5)
            return md5, tmpl, book_path
    md5, tmpl = load_active_template()
    if not tmpl:
        raise RuntimeError("本机还没有题本模板：请在网页「本地混合识别」设置里选择一次题本 PDF")
    book_path = os.path.join(CACHE, "题本_%s.pdf" % md5)
    return md5, tmpl, (book_path if os.path.exists(book_path) else None)

# ---------- multipart 解析（Python 3.13 已移除 cgi） ----------
def parse_multipart(body, ctype):
    m = re.search(r'boundary=("?)([^";]+)\1', ctype)
    if not m:
        raise ValueError("缺少 multipart boundary")
    bnd = ("--" + m.group(2)).encode()
    parts, buf = [], []
    for line in body.split(bnd):
        if line in (b"", b"--", b"--\r\n"):
            continue
        if line.startswith(b"\r\n"):
            line = line[2:]
        if line.endswith(b"\r\n"):
            line = line[:-2]
        parts.append(line)

    def ph(line):
        s = line.split(b"\r\n\r\n", 1)
        if len(s) != 2:
            return None
        head, data = s[0], s[1]
        name = re.search(rb'name="([^"]*)"', head)
        fn = re.search(rb'filename="([^"]*)"', head)
        return ((name.group(1).decode("utf-8", "replace") if name else ""),
                (fn.group(1).decode("utf-8", "replace") if fn else None),
                data)

    out = {}
    for p in parts:
        r = ph(p)
        if r:
            name, fn, data = r
            out.setdefault(name, []).append({"filename": fn, "data": data})
    return out

def images_to_pdf(img_list, out_path):
    """多张图片打包成一个 PDF（一份问卷多张扫描图的情况）"""
    import cv2
    import numpy as np
    import pymupdf
    doc = pymupdf.open()
    for it in img_list:
        arr = np.frombuffer(it["data"], dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            h, w = im.shape[:2]
            # 页面尺寸跟随图片宽高比：A3 横版拼页必须保持 >1.15 的比例，
            # expand_spread 才能识别出双页拼版并切分（不能固定成 595x842 竖版）
            if w / float(h) > 1.15:
                page = doc.new_page(width=842, height=round(842 * h / w))
            else:
                page = doc.new_page(width=595, height=round(595 * h / w))
            page.insert_image(page.rect, stream=buf.tobytes())
    if not doc.page_count:
        doc.close()
        raise RuntimeError("图片全部无法解码")
    doc.save(out_path)
    doc.close()

# ---------- 识别一次 ----------
def recognize(scan_files, structure, options):
    import cv2  # noqa: F401  (hybrid 已 monkey-patch cv2.imwrite)
    import tempfile
    keys = [k.strip() for k in str(options.get("keys") or "").replace("\n", ",").split(",") if k.strip()]
    vlm_budget = float(options.get("vlm_budget") or (90 if keys else 0))
    batch = int(options.get("batch") or 12)

    # 工作目录放系统临时目录（避免撑大程序目录；清理失败也不影响结果）
    workdir = tempfile.mkdtemp(prefix="qscan_svc_")

    # 合并扫描文件：多个（或多张图片）→ 一个 PDF
    if len(scan_files) == 1 and scan_files[0]["filename"] and \
            scan_files[0]["filename"].lower().endswith(".pdf"):
        scan_path = os.path.join(workdir, "scan.pdf")
        with open(scan_path, "wb") as f:
            f.write(scan_files[0]["data"])
    else:
        scan_path = os.path.join(workdir, "scan.pdf")
        imgs = [it for it in scan_files if it["filename"] and
                re.search(r"\.(png|jpe?g|bmp|webp)$", it["filename"], re.I)]
        if len(imgs) == len(scan_files):
            images_to_pdf(imgs, scan_path)
        else:
            raise RuntimeError("扫描文件类型混杂，请一次只传一份问卷（一个 PDF 或一组图片）")

    # v2.0: 无锁并发 (每请求独立 workdir)
    md5, tmpl, book_path = ensure_template(options.get("_book_bytes"))
    optmap = {}
    for p in tmpl["pages"]:
        for q in p.get("questions", []):
            optmap[q["id"]] = q
    cfg = {
        "keys": keys, "endpoint": hybrid.DEFAULT_ENDPOINT,
        "model": hybrid.DEFAULT_MODEL, "batch": batch,
        "borderline": 0, "optmap": optmap,
        "review": True, "fill_ocr": True, "vlm_budget": vlm_budget,
    }
    stat = hybrid.process_one(scan_path, tmpl, book_path, workdir, cfg)

    # 读回结果（CSV 被占用时引擎会改写到 .bak, 这里按通配兜底查找）
    def read_csv(p):
        import csv as _csv
        with open(p, encoding="utf-8-sig") as f:
            return list(_csv.DictReader(f))

    def find_csv(suffix):
        import glob as _glob
        cand = []
        for pat in ("scan_%s.csv", "scan_%s.csv.bak"):
            cand += [p for p in _glob.glob(os.path.join(workdir, pat % suffix))
                     if os.path.isfile(p)]
        return max(cand, key=os.path.getmtime) if cand else None

    rows_path = find_csv("识别结果")
    if not rows_path:
        raise RuntimeError("识别已完成，但结果文件未能写入或读取（结果 CSV 可能被其他程序占用），"
                           "请关闭正在打开该文件的程序后重试")
    rows = read_csv(rows_path)
    fill_path = find_csv("填空题")
    fills = read_csv(fill_path) if fill_path else []

    answers, meta, unmatched, extra, mstat = map_to_structure(rows, fills, structure, tmpl)

    # 页面配准信息（引擎侧算出，用于把识别框贴到扫描件实际格子上）
    reg = None
    try:
        import glob as _glob
        gp = _glob.glob(os.path.join(workdir, "*_页面配准.json"))
        if gp:
            reg = json.load(open(gp[0], encoding="utf-8"))
    except Exception:
        reg = None

    try:
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass
    return {
        "ok": True,
        "answers": answers,
        "meta": meta,
        "unmatched": unmatched,
        "extra_fills": extra,
        "boxes": build_boxes(tmpl, rows, book_path, reg),
        "stats": dict(stat, **mstat),
    }

# ---------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[http]", self.address_string(), fmt % args)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        # 允许线上页面（https://...）跨源调用本机服务：
        # 普通 CORS 预检 + Chrome 私有网络访问（Private Network Access）预检都要放行。
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         self.headers.get("Access-Control-Request-Headers") or "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        route = self.path.split("?")[0]
        if route.startswith("/ping"):
            md5 = active_md5()
            tmpl_pages = None
            if md5:
                p = os.path.join(CACHE, "template_%s.json" % md5)
                if os.path.exists(p):
                    tmpl_pages = len(json.load(open(p, encoding="utf-8"))["pages"])
            self._json({"ok": True, "engine": True, "template": md5, "template_pages": tmpl_pages})
            return
        # 托管网页本体：Mac/Safari 直接用 file:// 打开时会限制跨源请求，
        # 改由服务提供 http://127.0.0.1:8791/ ，网页与服务同源，任何浏览器都能用。
        if route in ("/", "/index.html", "/问卷扫描卷识别与核验系统2.html"):
            p = os.path.join(WEB_DIR, "index.html")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"ok": False, "error": "缺少 web/index.html"}, 404)
            return
        if route == "/favicon.ico":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        # 托管 pdf.js 本地副本（网页从 http://127.0.0.1:8791/ 打开时用相对路径引用它们）
        # 线上版：托管 web/ 下的全部静态资源（pdf.js 主库/worker、图标等）
        rel = route.lstrip("/")
        if rel and ".." not in rel and os.path.isfile(os.path.join(WEB_DIR, rel)):
            p = os.path.join(WEB_DIR, rel)
            with open(p, "rb") as f:
                data = f.read()
            ext = os.path.splitext(p)[1].lower()
            ctype = {".js": "application/javascript", ".css": "text/css",
                     ".html": "text/html", ".svg": "image/svg+xml",
                     ".png": "image/png", ".ico": "image/x-icon"}.get(ext, "application/octet-stream")
            if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
                ctype += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=3600")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return
        self._json({"ok": False, "error": "未知路径"}, 404)

    def do_POST(self):
        if self.path.startswith("/api/save"):
            # 线上版禁用写盘接口：导出文件由浏览器直接下载，服务端不落任何用户数据。
            self._json({"ok": False, "error": "在线版不支持写盘"}, 403)
            return
        if False:
            # 网页导出兜底通道：浏览器下载偶发 0 字节文件，改为服务端直接写盘
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > 300 * 1024 * 1024:
                    raise ValueError("请求体为空或超过 300MB")
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                name = str(req.get("name") or "导出文件")
                name = os.path.basename(name.replace("\\", "/")).strip()
                name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
                if not name:
                    raise ValueError("文件名不合法")
                # 扩展名白名单：本服务只用于网页导出的表格/数据文件。
                # 服务开着 CORS *, 任意网页都能 POST /api/save —— 若不限扩展名,
                # 等于允许第三方站点往桌面写 .exe/.bat 等任意文件。
                ALLOWED_EXTS = (".xlsx", ".csv", ".json", ".txt", ".zip")
                if os.path.splitext(name)[1].lower() not in ALLOWED_EXTS:
                    raise ValueError("不支持的文件类型：%s（仅允许 %s）"
                                     % (os.path.splitext(name)[1] or "(无扩展名)", " ".join(ALLOWED_EXTS)))
                data = base64.b64decode(req.get("b64") or "")
                if not data:
                    raise ValueError("文件内容为空")
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                if not os.path.isdir(desktop):
                    desktop = os.path.expanduser("~")
                path = os.path.join(desktop, name)
                stem, ext = os.path.splitext(path)
                k = 1
                while os.path.exists(path):
                    path = "%s(%d)%s" % (stem, k, ext)
                    k += 1
                with open(path, "wb") as f:
                    f.write(data)
                print("[服务] 已直接保存：%s（%d 字节）" % (path, len(data)))
                self._json({"ok": True, "path": path, "size": len(data)})
            except Exception as e:
                print("[服务] 保存失败：", str(e)[:200])
                self._json({"ok": False, "error": str(e)[:300]}, 500)
            return

        if not self.path.startswith("/api/recognize"):
            self._json({"ok": False, "error": "未知路径"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 300 * 1024 * 1024:
                raise ValueError("请求体为空或超过 300MB")
            body = self.rfile.read(n)
            form = parse_multipart(body, self.headers.get("Content-Type", ""))
            scans = form.get("scan", [])
            structure = json.loads(form["structure"][0]["data"].decode("utf-8")) if form.get("structure") else {}
            options = {}
            if form.get("options"):
                try:
                    options = json.loads(form["options"][0]["data"].decode("utf-8"))
                except Exception:
                    options = {}
            if form.get("book"):
                options["_book_bytes"] = form["book"][0]["data"]
            if not scans:
                raise ValueError("缺少扫描卷文件")
            if not structure.get("items"):
                raise ValueError("缺少题本结构（structure）")
            t0 = time.time()
            res = recognize(scans, structure, options)
            res["wall"] = round(time.time() - t0, 1)
            print("[服务] %s 识别完成 %ss，匹配 %d 题，未匹配 %d" %
                  (scans[0].get("filename", "?"), res["wall"], len(res["answers"]), len(res["unmatched"])))
            self._json(res)
        except Exception as e:
            print("[服务] 识别失败：", str(e)[:200])
            self._json({"ok": False, "error": str(e)[:300]}, 500)

def main():
    # 清理上次运行遗留的临时工作目录（设 QSCAN_NO_PRUNE=1 可跳过；失败不影响启动）
    if os.environ.get("QSCAN_NO_PRUNE") != "1":
        try:
            import tempfile, glob
            for d in (glob.glob(os.path.join(tempfile.gettempdir(), "qscan_svc_*"))
                      + glob.glob(os.path.join(CACHE, "req_*"))):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 60)
    print("  问卷扫描卷识别与核验系统 · 在线版 已启动")
    print("  地址: http://%s:%d   （前端与识别 API 同源）" % (HOST, PORT))
    print("  用户直接用浏览器访问该地址即可，无需安装任何本地服务。")
    print("=" * 60)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[服务] 已停止")

if __name__ == "__main__":
    main()
