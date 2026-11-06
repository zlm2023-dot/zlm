# -*- coding: utf-8 -*-
"""
纸质问卷自动识别工具 v2.0 (改进版)
====================
核心引擎改进:
  - 光照均衡 (CLAHE) + 自适应二值化
  - 透视变换 (Homography) 替代纯仿射
  - 多尺度墨迹检测 + 形状特征分析
  - 页匹配置信度校验 (pHash)
  - 检测区域自适应 padding
  - 手写数字模型预处理优化

用法:
  1. 生成模板:   python 问卷识别工具.py build-template --book <题本.pdf> --outdir outputs
  2. 识别扫描件: python 问卷识别工具.py recognize --scan <扫描件.pdf> --outdir outputs
"""
import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import threading

import cv2
import numpy as np
import pymupdf

CIRCLED_RE = re.compile(r'^[\u2460-\u2473\u3251-\u325F\u32B1-\u32BF]')
CIRCLED_VAL = {chr(0x2460 + i): i + 1 for i in range(10)}
UNDERSCORE_RE = re.compile(r'^_{2,}$')
DIGIT_ROW_RE = re.compile(r'^\d{1,2}$')

DPI = 200
SCALE = DPI / 72.0

# ---------------------------------------------------------------- 图像预处理

def preprocess_scan(img):
    """
    对扫描图做光照均衡 + 抗噪声, 配准前调用.
    返回 (预处理后的灰度图, CLAHE增强图用于墨迹检测).
    """
    if img is None or img.size == 0:
        return img, img
    # 轻度高斯模糊去噪 (保留大结构)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # CLAHE 光照均衡: tileGridSize=(8,8) 适合 A4 大小
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    # 用 divide 消除整体亮度不均 (除法矫正)
    gaussian_bg = cv2.GaussianBlur(enhanced, (31, 31), 0)
    div_corrected = cv2.divide(enhanced, gaussian_bg + 1, scale=255.0)
    return blur, div_corrected


def adaptive_binarize(img, block_size=35, C=10):
    """
    自适应二值化: 比固定阈值更能应对光照不均/浅色笔迹.
    返回二值图 (255=墨迹, 0=背景).
    """
    if img is None or img.size == 0:
        return None
    bs = max(3, block_size | 1)  # 必须是奇数
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, bs, C
    )
    # 形态学闭运算填补细笔画断裂
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def phash(img, hash_size=8):
    """
    感知哈希: 用于配准后校验. 返回 64 位二进制字符串.
    两张相似图像的 pHash 汉明距离 < 5.
    """
    if img is None:
        return None
    try:
        h = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        d = h.astype(np.float32)
        diff = d[:, 1:] > d[:, :-1]
        bits = ''.join('1' if v else '0' for v in diff.flatten())
        return bits
    except Exception:
        return None


def phash_dist(h1, h2):
    if h1 is None or h2 is None:
        return 999
    return sum(a != b for a, b in zip(h1, h2))


# ---------------------------------------------------------------- 模板构建

def find_underscore_runs(page, clip=None):
    """字符级查找下划线填空区域, 返回 [{'rect', 'anchor'}]"""
    runs = []
    raw = page.get_text('rawdict', clip=clip)
    chars = []
    for block in raw.get('blocks', []):
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                for ch in span.get('chars', []):
                    chars.append((ch['c'], pymupdf.Rect(ch['bbox'])))
    chars.sort(key=lambda t: (t[1].y0 + t[1].y1) / 2)
    lines = []
    for c, r in chars:
        yc = (r.y0 + r.y1) / 2
        if lines and abs(lines[-1][0] - yc) < 4:
            lines[-1][1].append((r.x0, c, r))
        else:
            lines.append([yc, [(r.x0, c, r)]])
    for _, lcs in lines:
        lcs.sort()
        i = 0
        while i < len(lcs):
            if lcs[i][1] == '_':
                j = i
                while j < len(lcs) and lcs[j][1] == '_':
                    j += 1
                if j - i >= 2:
                    rect = pymupdf.Rect(lcs[i][0], lcs[i][2].y0, lcs[j - 1][2].x1, lcs[j - 1][2].y1)
                    anchor = ''.join(c for _, c, _ in lcs[max(0, i - 12):i])
                    anchor = re.sub(r'[\s_]+', '', anchor)[-8:]
                    runs.append({'rect': rect, 'anchor': anchor})
                i = j
            else:
                i += 1
    return runs


LABEL_COLON_RE = re.compile(r'^([0-9A-Za-z\u4e00-\u9fa5·]{1,6})[：:﹕∶]$')
# 「亲爱的同学：」这类书信导语不是填空槽（高州卷 p1 实测误报），排除
HONORIFIC_RE = re.compile(r'^(亲爱的|尊敬的|敬爱的|致|各位)')


def find_label_blanks(page, exclude_rects=None, min_gap=18, min_blank=30, max_width=300):
    """标签锚定填空槽：行尾是「×××：」且冒号后是空白区 → 生成填空槽。
    与 find_underscore_runs 互补：很多卷的表头（班别：/姓名：/性别：/年龄：）
    后面没有印刷下划线，只有标签和留白，此前完全识别不到。"""
    runs = []
    exclude_rects = exclude_rects or []
    words = [{'text': w[4], 'rect': pymupdf.Rect(w[:4])} for w in page.get_text('words')]
    for ln in cluster_lines(words, tol=4):
        items = sorted(ln['items'], key=lambda w: w['rect'].x0)
        if len(items) < 1:
            continue
        # 行内含圈号/括号题号 → 是题目行，不是表头
        if any(CIRCLED_RE.match(w['text']) for w in items):
            continue
        last = items[-1]
        m = LABEL_COLON_RE.match(re.sub(r'\s+', '', last['text']))
        if not m or HONORIFIC_RE.match(m.group(1)):
            continue
        if any(r.contains(last['rect']) or r.intersects(last['rect'])
               for r in exclude_rects):
            continue
        # 冒号右侧空白带：到同一行下一个词为止；间距太小说明标签后紧跟内容
        nxt = [w['rect'] for w in items if w['rect'].x0 > last['rect'].x1 + 0.5]
        if nxt:
            nx0 = min(r.x0 for r in nxt)
            if nx0 - last['rect'].x1 < min_gap:
                continue
            right = nx0 - 2
        else:
            right = min(last['rect'].x1 + max_width, page.rect.x1 - 36)
        if right - last['rect'].x1 < min_blank:
            continue
        rect = pymupdf.Rect(last['rect'].x1 + 2, last['rect'].y0 - 2, right, last['rect'].y1 + 2)
        runs.append({'rect': rect, 'anchor': m.group(1)})
    return runs


def words_in_rect(page, rect):
    out = []
    for w in page.get_text('words'):
        r = pymupdf.Rect(w[:4])
        if rect.contains(r):
            out.append({'text': w[4], 'rect': r})
    return out


def cluster_lines(items, tol=3.0):
    items = sorted(items, key=lambda w: (w['rect'].y0 + w['rect'].y1) / 2)
    lines = []
    for it in items:
        yc = (it['rect'].y0 + it['rect'].y1) / 2
        placed = False
        for ln in lines:
            if abs(ln['yc'] - yc) < tol:
                ln['items'].append(it)
                ln['yc'] = sum((x['rect'].y0 + x['rect'].y1) / 2 for x in ln['items']) / len(ln['items'])
                placed = True
                break
        if not placed:
            lines.append({'yc': yc, 'items': [it]})
    lines.sort(key=lambda ln: ln['yc'])
    return lines


NUM_TOKEN_RE = re.compile(r'^\s*(\d{1,2})\s*[.、．]?\s*$')
HEADER_STEM = {'条目', '题目', '题号', '项目', '序号', '量表', '维度'}


def _num_options_in_band(band, words, min_opts=2):
    """在一个候选矩形内按「数字 token」切出若干选项。

    兼容两类版式：① 矩阵量表行，数字各自落在独立单元格里；
                  ② 一个单元格里连排「1 完全不会  2 少数几天」。
    刻度值取印刷数字本身：既支持 1..n（常规量表），也支持 0..n-1
    （PHQ-9 式 0 计分量表）。约束保留「连续 +1 递增、起始 ≥0」，
    防止把题干/说明里的散落数字误当选项。
    返回 [(value, label, rect)]，失败返回 []。"""
    ws = [w for w in words
          if w['rect'].x0 >= band[0] - 1 and w['rect'].x1 <= band[2] + 1
          and w['rect'].y0 >= band[1] - 1 and w['rect'].y1 <= band[3] + 1]
    if not ws:
        return []
    nums = []
    for w in ws:
        m = NUM_TOKEN_RE.match(w['text'])
        if m:
            nums.append((int(m.group(1)), w))
    nums.sort(key=lambda t: t[1]['rect'].x0)
    vals = [v for v, _ in nums]
    if (len(nums) < min_opts or vals[0] < 0
            or any(vals[i + 1] - vals[i] != 1 for i in range(len(vals) - 1))):
        return []
    # 分界线必须落在两个数字 token 的「中心之间」, 而不是下一个 token 的左边缘:
    # 后者会让相邻两个选项各覆盖对方一半的格子, 导致两者墨迹几乎相等而被误判多选。
    ctr = [(w['rect'].x0 + w['rect'].x1) / 2 for _, w in nums]
    bnds = []
    step0 = (ctr[1] - ctr[0]) if len(ctr) > 1 else 30.0
    bnds.append(max(band[0], nums[0][1]['rect'].x0 - step0 / 2))
    for i in range(1, len(ctr)):
        bnds.append((ctr[i - 1] + ctr[i]) / 2)
    bnds.append(band[2])
    out = []
    for i, (v, w) in enumerate(nums):
        x0, x1 = bnds[i], bnds[i + 1]
        yc = (w['rect'].y0 + w['rect'].y1) / 2
        lab = ''.join(x['text'] for x in ws
                      if x['rect'].x0 >= w['rect'].x1 - 1 and x['rect'].x1 <= x1 + 1
                      and abs((x['rect'].y0 + x['rect'].y1) / 2 - yc) < 9)
        if NUM_TOKEN_RE.match(lab):
            lab = ''
        out.append({'value': v, 'label': lab.strip(),
                    'rect': [round(x0, 2), round(band[1] + 2, 2),
                             round(x1, 2), round(band[3] - 2, 2)]})
    return out


def _label_cell_options(cells, texts, min_opts=2):
    """单元格里直接写的是文字（是/否、男/女、从不/偶尔…）的矩阵行。
    每个非空短单元格算一个选项，值按列序编号。"""
    opts = []
    v = 0
    for j, c in enumerate(cells or []):
        if c is None or j == 0:
            continue
        t = re.sub(r'\s+', '', texts[j] if j < len(texts) else '' or '')
        if not t or len(t) > 10:
            continue
        v += 1
        opts.append({'value': v, 'label': t,
                     'rect': [round(c[0] + 2, 2), round(c[1] + 2, 2),
                              round(c[2] - 2, 2), round(c[3] - 2, 2)]})
    return opts if v >= min_opts else []


def detect_matrix_tables(page, pno, all_words, two_col_cells, two_col_bboxes):
    """识别「矩阵式量表」表格：每行一道题，选项落在后续各列。

    兼容三种常见变体：
      A 标准量表：表头「条目|从不|偶尔|经常|总是」，数据行是「题干|1|2|3|4」
      B 文字选项：数据行是「题干|是|否」
      C 错位表格：题干一行、选项一行（题干的第二列可能同时装着题干）
    命中的单元格矩形会写进 two_col_cells，供后续浮动检测排除，避免重复出题。
    """
    qs = []
    try:
        tables = page.find_tables().tables
    except Exception:
        return qs
    for tab in tables:
        try:
            rows = tab.rows
            ex = tab.extract()
        except Exception:
            continue
        if len(rows) < 2:
            continue
        texts = []
        for i in range(len(rows)):
            row = ex[i] if i < len(ex) else []
            texts.append([re.sub(r'\s+', '', (c or '')) for c in row])
        ncol = max(len(r.cells) for r in rows)
        if ncol < 3:
            continue
        if any('选项' in t for t in texts[0]):
            continue                      # 交给原有的「两列表」逻辑处理
        # 首行若是刻度表头（「条目|从不|偶尔|经常|总是」），取来当选项文字
        hdr_labels = []
        if texts and (texts[0][0] in HEADER_STEM or len(texts[0][0]) <= 4):
            hdr_labels = [re.sub(r'\s+', '', texts[0][j])
                          for j in range(1, len(texts[0]))
                          if j < len(texts[0]) and texts[0][j].strip()
                          and not NUM_TOKEN_RE.match(texts[0][j])]
        # ---- 按「表格行」一行一题 ----
        # 不能用「首列是否为空」判断归属：题干跨行时（如「4.目前为止…」+「的一些
        # 重要东西。」），首列可能整行为空而文字落在第 2 列，按首列并题会把后面
        # 连续几道题一起吞进上一题 → 模板漏题，网页题号就对不上，答案搬不过去。
        # 改为一行一题；题干以单元格文本为准，再用「该行 y 区间内的左侧文字」重建
        # （能补齐跨到下一视觉行的续写），取两者中更完整的那个。
        def _clean_stem(s):
            return re.sub(r'[\s0-9.、．()（）]', '', s or '')

        items = []
        for i, r in enumerate(rows):
            cells = list(r.cells)
            if not cells or cells[0] is None:
                continue
            ys0 = min(c[1] for c in cells if c)
            ys1 = max(c[3] for c in cells if c)
            xs = min((c[0] for c in cells[1:] if c), default=None)
            base = texts[i][0] if i < len(texts) and texts[i] else ''
            reb = ''
            if xs is not None:
                sw = [w for w in all_words
                      if w['rect'].x1 <= xs + 1 and w['rect'].y0 >= ys0 - 1
                      and w['rect'].y1 <= ys1 + 1
                      and not re.fullmatch(r'\d+', w['text'])]
                sw.sort(key=lambda w: (round(w['rect'].y0, 1), w['rect'].x0))
                reb = re.sub(r'\s+', '', ''.join(w['text'] for w in sw))
            stem = reb if len(_clean_stem(reb)) > len(_clean_stem(base)) else base
            items.append({'rows': [i], 'stem': stem})
        got = []
        for it in items:
            stem = it['stem']
            # 题干只写了题号（如「1 .」）时，题干文字其实在后面的大单元格里
            if len(re.sub(r'[\s0-9.、．()（）]', '', stem)) < 2:
                for ri in it['rows']:
                    for j, c in enumerate(list(rows[ri].cells)[1:], 1):
                        t = texts[ri][j] if j < len(texts[ri]) else ''
                        if len(re.sub(r'\s+', '', t or '')) >= 8:
                            stem = re.sub(r'\s+', '', t)
                            break
                    else:
                        continue
                    break
            # 短题干但带题号（如「2电视.」）是正常数据行，不能丢——
            # 丢一行模板题，网页端同页全错位。
            if stem in HEADER_STEM or (len(stem) <= 4 and not re.search(r'\d', stem)):
                continue                  # 表头/无意义的刻度说明行
            rcs = []
            for ri in it['rows']:
                cells = list(rows[ri].cells)
                for j, c in enumerate(cells):
                    if c is None or j == 0:
                        continue
                    rcs.append(pymupdf.Rect(c[0], c[1], c[2], c[3]))
            if len(rcs) < 2:
                continue
            band = pymupdf.Rect(min(r.x0 for r in rcs), min(r.y0 for r in rcs),
                                max(r.x1 for r in rcs), max(r.y1 for r in rcs))
            opts = _num_options_in_band(band, all_words)
            if not opts:
                li = it['rows'][-1]
                opts = _label_cell_options(list(rows[li].cells), texts[li])
            if not opts:
                continue
            if hdr_labels and len(hdr_labels) == len(opts) and not any(o.get('label') for o in opts):
                for k, o in enumerate(opts):
                    o['label'] = hdr_labels[k]
            for o in opts:
                two_col_cells.append(pymupdf.Rect(o['rect']))
            two_col_cells.append(band)
            got.append({
                'id': '', 'type': 'choice', 'qtext': stem,
                'options': [{'value': o['value'], 'text': o.get('label', ''),
                             'rect': o['rect']} for o in opts]})
        if got:
            two_col_bboxes.append(pymupdf.Rect(tab.bbox))
            for q in got:
                q['id'] = 'p%dm%03d' % (pno, len(qs) + 1)
                qs.append(q)
    return qs


def build_template(book_path, out_json):
    doc = pymupdf.open(book_path)
    pages = []
    for pno, page in enumerate(doc, start=1):
        all_words = [{'text': w[4], 'rect': pymupdf.Rect(w[:4])} for w in page.get_text('words')]
        questions = []
        fills = []
        section = 0
        two_col_cells = []
        two_col_bboxes = []

        for w in all_words:
            if re.fullmatch(r'第[一二三四五六七八九十]+部分', w['text']):
                section += 1

        matrix_qs = detect_matrix_tables(page, pno, all_words, two_col_cells, two_col_bboxes)

        tables = page.find_tables()
        qseq = 0
        for tab in tables.tables:
            nrows = len(tab.rows)
            if nrows < 2:
                continue
            header = [re.sub(r'\s+', '', (c or '')) for c in tab.extract()[0]]
            if '选项' not in ''.join(header):
                continue
            two_col_bboxes.append(pymupdf.Rect(tab.bbox))
            xs1 = [pymupdf.Rect(r.cells[1]) for r in tab.rows if r.cells and r.cells[1] is not None]
            if not xs1:
                continue
            c1x0 = min(r.x0 for r in xs1)
            c1x1 = max(r.x1 for r in xs1)
            qrows = []
            for ri in range(1, nrows):
                cells = tab.rows[ri].cells
                if not cells or cells[0] is None:
                    continue
                t = re.sub(r'\s+', '', tab.extract()[ri][0] or '')
                if not t:
                    continue
                rr = pymupdf.Rect(cells[0])
                qrows.append((rr.y0, rr.y1, t, rr))
            for qi, (ry0, ry1, qtext, qrect) in enumerate(qrows):
                two_col_cells.append(qrect)
                y_end = qrows[qi + 1][0] - 1 if qi < len(qrows) - 1 else tab.bbox[3]
                cell1 = pymupdf.Rect(c1x0, ry0 - 1, c1x1, y_end)
                two_col_cells.append(cell1)
                ws = [w for w in all_words if cell1.contains(w['rect'])]
                ws.sort(key=lambda w: (round(w['rect'].y0, 1), w['rect'].x0))
                opts = []
                cur = None
                for w in ws:
                    if CIRCLED_RE.match(w['text']):
                        qseq += 1
                        cur = {'value': CIRCLED_VAL[w['text'][0]], 'label': w['text'][0],
                               'words': [w], 'rect': list(w['rect'])}
                        opts.append(cur)
                    elif cur is not None:
                        cur['words'].append(w)
                        cur['rect'] = [min(cur['rect'][0], w['rect'].x0), min(cur['rect'][1], w['rect'].y0),
                                       max(cur['rect'][2], w['rect'].x1), max(cur['rect'][3], w['rect'].y1)]
                for run in find_underscore_runs(page, cell1):
                    fills.append({'name': f'p{pno}_fill{len(fills)+1}_{run["anchor"]}',
                                  'rect': [round(v, 2) for v in run['rect']],
                                  'anchor': run['anchor'], 'qtext': qtext})
                if opts:
                    questions.append({
                        'id': f'p{pno}t{qseq:03d}', 'type': 'choice', 'qtext': qtext,
                        'options': [{'value': o['value'],
                                     'text': ''.join(x['text'] for x in o['words'][1:]) or o['label'],
                                     'rect': [round(v, 2) for v in o['rect']]} for o in opts]})

        # ---- 2) 数字打分表 ----
        digit_words = [w for w in all_words
                       if re.fullmatch(r'\d', w['text'])
                       and not any(c.contains(w['rect']) for c in two_col_cells)]
        dgroups = []
        for ln in cluster_lines(digit_words):
            seq = sorted(ln['items'], key=lambda w: w['rect'].x0)
            cur = [seq[0]]
            for a, b in zip(seq, seq[1:]):
                if int(b['text']) == int(cur[-1]['text']) + 1:
                    cur.append(b)
                else:
                    if len(cur) >= 2 and int(cur[0]['text']) in (0, 1):
                        dgroups.append(cur)
                    cur = [b]
            if len(cur) >= 2 and int(cur[0]['text']) in (0, 1):
                dgroups.append(cur)
        dgroups.sort(key=lambda g: g[0]['rect'].y0)

        all_line_clusters = cluster_lines(all_words)
        for gi, g in enumerate(dgroups):
            # 落在矩阵表格 bbox 里的刻度行已由 detect_matrix_tables 整行出题，
            # 这里再出一遍会生成「把多行题干拼在一起」的垃圾题（模板题数虚增、
            # 网页题号错位）。直接跳过。
            gy = (g[0]['rect'].y0 + g[0]['rect'].y1) / 2
            if any(b.y0 - 2 <= gy <= b.y1 + 2 for b in two_col_bboxes):
                continue
            qseq += 1
            gx0 = g[0]['rect'].x0
            y0, y1 = g[0]['rect'].y0, max(x['rect'].y1 for x in g)
            prev_y1 = dgroups[gi-1][-1]['rect'].y1 + 2 if gi > 0 else y0 - 45
            band_words = sorted([w for w in all_words
                                 if w['rect'].x1 < gx0 - 3 and w['rect'].y0 >= prev_y1 - 1
                                 and w['rect'].y1 <= y1 + 1], key=lambda w: (round(w['rect'].y0), w['rect'].x0))
            qtext = re.sub(r'\s+', '', ''.join(w['text'] for w in band_words))
            opts = []
            for d in g:
                label = ''
                best_dist = 1e9
                dcx = (d['rect'].x0 + d['rect'].x1) / 2
                for ln in all_line_clusters:
                    if ln['yc'] >= d['rect'].y0 - 1:
                        continue
                    if any(re.fullmatch(r'\d', w['text']) for w in ln['items']):
                        continue
                    dist = d['rect'].y0 - max(w['rect'].y1 for w in ln['items'])
                    if dist > 55 or dist >= best_dist:
                        continue
                    hits = [w for w in ln['items']
                            if w['rect'].x0 - 2 <= dcx <= w['rect'].x1 + 2]
                    if hits:
                        label = ''.join(w['text'] for w in sorted(hits, key=lambda w: w['rect'].x0))
                        best_dist = dist
                opts.append({'value': int(d['text']), 'text': label,
                             'rect': [round(v, 2) for v in d['rect']]})
            questions.append({'id': f'p{pno}d{qseq:03d}', 'type': 'choice', 'qtext': qtext, 'options': opts})

        # ---- 2.5) 扩展选项检测区域 ----
        all_opts = [o for q in questions for o in q['options']]
        opt_lines = []
        for o in sorted(all_opts, key=lambda o: (o['rect'][1], o['rect'][0])):
            yc = (o['rect'][1] + o['rect'][3]) / 2
            if opt_lines and abs(opt_lines[-1][0] - yc) < 6:
                opt_lines[-1][1].append(o)
                opt_lines[-1][0] = sum((x['rect'][1] + x['rect'][3]) / 2 for x in opt_lines[-1][1]) / len(opt_lines[-1][1])
            else:
                opt_lines.append([yc, [o]])
        line_bounds = []
        for yc, los in opt_lines:
            line_bounds.append((min(o['rect'][1] for o in los), max(o['rect'][3] for o in los)))
        for li, (yc, los) in enumerate(opt_lines):
            los.sort(key=lambda o: o['rect'][0])
            line_top, line_bot = line_bounds[li]
            prev_yc = opt_lines[li - 1][0] if li > 0 else -1e9
            nxt_yc = opt_lines[li + 1][0] if li < len(opt_lines) - 1 else 1e9
            above = [w['rect'].y1 for w in all_words
                     if prev_yc + 10 < (w['rect'].y0 + w['rect'].y1) / 2 < line_top - 2]
            if above:
                top = min(max(above) + 1, line_top - 1)
            elif li > 0:
                top = (line_bounds[li - 1][1] + line_top) / 2 + 1
            else:
                top = line_top - 3
            beloww = [w['rect'].y0 for w in all_words
                      if line_bot + 2 < (w['rect'].y0 + w['rect'].y1) / 2 < nxt_yc - 10]
            if beloww:
                bot = max(min(beloww) - 1, line_bot + 1)
            elif li < len(opt_lines) - 1:
                bot = (line_bot + line_bounds[li + 1][0]) / 2 - 1
            else:
                bot = line_bot + 12
            mids = [(o['rect'][0] + o['rect'][2]) / 2 for o in los]
            for i, o in enumerate(los):
                x0 = (mids[i - 1] + mids[i]) / 2 if i > 0 else o['rect'][0] - 14
                x1 = (mids[i] + mids[i + 1]) / 2 if i < len(los) - 1 else o['rect'][2] + 14
                o['rect'] = [round(x0 + 2, 2), round(top + 2, 2), round(x1 - 2, 2), round(bot - 2, 2)]

        # ---- 2.6) 矩阵式量表（在选项扩张之后并入，保持单元格矩形不被改动）----
        questions.extend(matrix_qs)

        # ---- 3) 行内圈号选项 ----
        raw = page.get_text('rawdict')
        chars = []
        for block in raw.get('blocks', []):
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    for ch in span.get('chars', []):
                        chars.append((ch['c'], pymupdf.Rect(ch['bbox'])))
        chars.sort(key=lambda t: (t[1].y0 + t[1].y1) / 2)
        clines = []
        for c, r in chars:
            ycc = (r.y0 + r.y1) / 2
            if clines and abs(clines[-1][0] - ycc) < 4:
                clines[-1][1].append((r.x0, c, r))
            else:
                clines.append([ycc, [(r.x0, c, r)]])
        circled_opts = []
        for _, lcs in clines:
            lcs.sort()
            for i, (x, c, r) in enumerate(lcs):
                if c in CIRCLED_VAL:
                    rr = pymupdf.Rect(r)
                    if any(bb.contains(rr) for bb in two_col_bboxes):
                        continue
                    txt = ''
                    for x2, c2, r2 in lcs[i + 1:]:
                        if c2 in CIRCLED_VAL:
                            break
                        if (r2.y0 + r2.y1) / 2 - ycc > 6:
                            break
                        txt += c2
                    circled_opts.append({'value': CIRCLED_VAL[c], 'text': txt,
                                         'rect': [round(v, 2) for v in rr]})
        circled_opts.sort(key=lambda o: (o['rect'][1], o['rect'][0]))
        group = []
        for o in circled_opts:
            if o['value'] == 1 and group:
                questions.append({
                    'id': f'p{pno}i{len(questions)+1:03d}', 'type': 'choice',
                    'qtext': '', 'options': group})
                group = []
            group.append(o)
        if group:
            questions.append({
                'id': f'p{pno}i{len(questions)+1:03d}', 'type': 'choice',
                'qtext': '', 'options': group})

        # ---- 3.5) 裸竖排选项题（无表格线: 题干后跟 "1. xx" "2. xx" … 各占一行）----
        # 高州 10 页卷实测: PHQ/GAD 竖排题（题干跨页断表后）引擎此前完全漏检,
        # 页级题数 7 vs 前端 18 → 整卷配对错位、识别值错且无告警。
        consumed = list(two_col_cells) + list(two_col_bboxes)
        vlines = []
        for ln in cluster_lines(all_words, tol=4):
            its = sorted(ln['items'], key=lambda w2: w2['rect'].x0)
            r = pymupdf.Rect(min(w2['rect'].x0 for w2 in its), min(w2['rect'].y0 for w2 in its),
                             max(w2['rect'].x1 for w2 in its), max(w2['rect'].y2 if hasattr(w2['rect'], 'y2') else w2['rect'].y1 for w2 in its))
            vlines.append({'items': its, 'text': ''.join(w2['text'] for w2 in its), 'rect': r})
        vlines.sort(key=lambda L: L['rect'].y0)

        def _consumed(rect):
            cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
            return any(c.contains(pymupdf.Point(cx, cy)) for c in consumed)

        OPT_LINE_RE = re.compile(r'^(\d{1,2})[.、．]\s*(\S.*)$')
        vi = 0
        while vi < len(vlines):
            ln = vlines[vi]
            vi += 1
            if _consumed(ln['rect']):
                continue
            m1 = OPT_LINE_RE.match(ln['text'])
            if not m1 or int(m1.group(1)) != 1:
                continue
            opts, exp, vj = [], 1, vi - 1
            while vj < len(vlines):
                mj = OPT_LINE_RE.match(vlines[vj]['text'])
                if not mj or int(mj.group(1)) != exp:
                    break
                body = mj.group(2).strip()
                if not body or len(body) > 14 or _consumed(vlines[vj]['rect']):
                    break
                opts.append({'value': exp, 'text': body,
                             'rect': [round(v, 2) for v in vlines[vj]['rect']]})
                exp += 1
                vj += 1
            if len(opts) < 3:
                continue
            stem_parts = []
            sk = vi - 2
            while sk >= 0 and len(stem_parts) < 3:
                up = vlines[sk]
                if _consumed(up['rect']):
                    break
                mm = OPT_LINE_RE.match(up['text'])
                if mm:
                    # 数字开头行: 长的是题干(收进来后停), 短的是上一题的选项(停)
                    if len(mm.group(2).strip()) > 14:
                        stem_parts.insert(0, up['text'])
                    break
                if re.match(r'^[一二三四五六七八九十]+\s*[、.]', up['text']):
                    break
                stem_parts.insert(0, up['text'])
                sk -= 1
            if not stem_parts or not re.match(r'^\d{1,2}[.、．]', stem_parts[0]):
                continue
            questions.append({
                'id': f'p{pno}v{len(questions)+1:03d}', 'type': 'choice',
                'qtext': ''.join(stem_parts)[:120],
                'options': opts})
            vi = vj

        # ---- 4) 表格外填空 ----
        for run in find_underscore_runs(page):
            if any(c.contains(run['rect']) for c in two_col_cells):
                continue
            fills.append({'name': f'p{pno}_fill{len(fills)+1}_{run["anchor"]}',
                          'rect': [round(v, 2) for v in run['rect']],
                          'anchor': run['anchor'], 'qtext': ''})

        # ---- 4b) 标签锚定填空（无下划线的表头留白，如 班别：/姓名：/性别：/年龄：）----
        excl = list(two_col_cells) + [pymupdf.Rect(f['rect']) for f in fills]
        for run in find_label_blanks(page, excl):
            fills.append({'name': f'p{pno}_fill{len(fills)+1}_{run["anchor"]}',
                          'rect': [round(v, 2) for v in run['rect']],
                          'anchor': run['anchor'], 'qtext': ''})

        if questions or fills:
            pages.append({'page': pno, 'section': section, 'questions': questions, 'fills': fills})

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'dpi': DPI, 'pages': pages}, f, ensure_ascii=False, indent=1)
    codebook_path = os.path.join(os.path.dirname(out_json), '题目对照表.csv')
    with open(codebook_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['题目ID', '页码', '题干', '选项值', '选项文本'])
        for p in pages:
            for q in p['questions']:
                for o in q['options']:
                    w.writerow([q['id'], p['page'], q['qtext'], o['value'], o['text']])
            for fl in p['fills']:
                w.writerow([fl['name'], p['page'], f'[填空] {fl["anchor"]}', '', ''])
    nq = sum(len(p['questions']) for p in pages)
    nf = sum(len(p['fills']) for p in pages)
    print(f'模板完成: {len(pages)}页, {nq}道选择题, {nf}个填空 -> {out_json}')
    return pages


# ---------------------------------------------------------------- 图像配准与识别

def render_page(page, dpi=DPI):
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return img


def _final_eval(scan, tmpl_img, M):
    """按最终度量评估某个变换: 全分辨率高斯模糊后的内容区归一化相关。"""
    th, tw = tmpl_img.shape
    al = cv2.warpAffine(scan, M, (tw, th), borderValue=255)
    tblur = cv2.GaussianBlur(tmpl_img, (0, 0), 3)
    ablur = cv2.GaussianBlur(al, (0, 0), 3)
    tmask = (tblur < 220)
    if tmask.sum() > 100:
        ta = tblur[tmask].astype(np.float32)
        sa = ablur[tmask].astype(np.float32)
        ta -= ta.mean()
        sa -= sa.mean()
        denom = np.sqrt((ta * ta).sum() * (sa * sa).sum())
        q = float((ta * sa).sum() / denom) if denom > 0 else 0.0
    else:
        q = 0.0
    return al, q


def _refine_pc(scan, tmpl_img, ang, sc, tdx, tdy, down=None, f2=2):
    """策略 A: 相位相关细化 —— 角度×缩放「两尺度两段式」搜索。
    对笔画稀疏的页面更准, 但对周期性表格容易跳到相邻表格行。

    旧版固定 9 角度 × 7 缩放 = 63 次 1/2 尺度 phaseCorrelate (单次 ~21ms),
    单页细化 ~1.3s, 是整份卷最贵的一步 (cProfile: 十页卷 5400 次/64s)。
    实测 1/4 尺度的相位相关峰值位置与 1/2 尺度一致, 只是精度不同, 故改为:
      ① 1/4 尺度上 5 角度 × 3 缩放 = 15 次 (单次 ~5ms), 只用来定位;
      ② 1/2 尺度上最优解邻域 3×3 = 9 次, 给出精确平移。
    最终仍由全分辨率 _final_eval 与 ECC 裁决, 精度不变。
    本函数不再做 ECC —— 由 _refine_best 对最终胜出解统一做一次。"""
    th, tw = tmpl_img.shape
    t2 = _down(tmpl_img, 2).astype(np.float32)
    s2 = _down(scan, 2).astype(np.float32)
    t4 = _down(tmpl_img, 4).astype(np.float32)
    s4 = _down(scan, 4).astype(np.float32)
    m2 = t2 < 230
    m4 = t4 < 230

    def probe(t, s, mask, f, ang_, sc_):
        h, w = t.shape
        mm = cv2.getRotationMatrix2D((w / 2, h / 2), ang_, sc_)
        mm[0, 2] += tdx / f
        mm[1, 2] += tdy / f
        wp = cv2.warpAffine(s, mm, (w, h), borderValue=255)
        try:
            (dx, dy), _ = cv2.phaseCorrelate(t, wp)
        except cv2.error:
            return None
        mm2 = mm.copy()
        mm2[0, 2] -= dx
        mm2[1, 2] -= dy
        a = cv2.warpAffine(s, mm2, (w, h), borderValue=255)
        ta = t[mask]; sa = a[mask]
        taz = ta - ta.mean(); saz = sa - sa.mean()
        den = np.sqrt(float((taz * taz).sum()) * float((saz * saz).sum()))
        q = float((taz * saz).sum() / den) if den > 0 else -1.0
        return q, ang_, sc_, -dx * f, -dy * f

    best4 = None
    for a_ in np.arange(ang - 1, ang + 1.01, 0.5):
        for sc_ in (sc - 0.02, sc, sc + 0.02):
            r = probe(t4, s4, m4, 4, a_, sc_)
            if r is not None and (best4 is None or r[0] > best4[0]):
                best4 = r
    if best4 is None:
        return None, 0.0, None
    b_ang, b_sc = best4[1], best4[2]
    best2 = None
    for a_ in (b_ang - 0.25, b_ang, b_ang + 0.25):
        for sc_ in (b_sc - 0.01, b_sc, b_sc + 0.01):
            r = probe(t2, s2, m2, 2, a_, sc_)
            if r is not None and (best2 is None or r[0] > best2[0]):
                best2 = r
    if best2 is not None:
        b_ang, b_sc, tdx, tdy = best2[1], best2[2], best2[3], best2[4]
    else:
        b_ang, b_sc, tdx, tdy = best4[1], best4[2], best4[3], best4[4]
    M = cv2.getRotationMatrix2D((tw / 2, th / 2), b_ang, b_sc)
    M[0, 2] += tdx
    M[1, 2] += tdy
    al, q = _final_eval(scan, tmpl_img, M)
    # 顺手把「扫描像素 → 模板像素」的仿射变换带出去：网页端画识别框时
    # 用它反算，框才能贴住扫描件上的实际格子（纯显示用途，不影响识别）
    return al, q, M


def _refine_grid(scan, tmpl_img, ang, sc, tdx, tdy, down=None, f2=2):
    """策略 B: 1/2 尺度「角度×缩放 + 平移网格」直接搜索。
    不依赖相位相关, 因此在周期性表格版式上不会跳行, 但网格较粗。

    平移网格改两级: 步长 4 扫 ±16 (81 点) → 最优节点邻域步长 1 扫 ±3 (49 点)。
    步长 4 保证真实极值必落在最优节点的 ±2 内, 第二级即可取到整像素解,
    与旧版 17×17 步长 2 的全量枚举等效, 但只有 130 次 ncc2 (旧版 289 次)。
    本函数不再做 ECC —— 由 _refine_best 对最终胜出解统一做一次。"""
    th, tw = tmpl_img.shape
    t2 = _down(tmpl_img, 2).astype(np.float32)
    s2 = _down(scan, 2).astype(np.float32)
    ch2, cw2 = t2.shape
    tmask2 = t2 < 230
    ta0 = t2[tmask2]
    ta0 = ta0 - ta0.mean()
    denom_t = np.sqrt(float((ta0 * ta0).sum()))

    def ncc2(ang2, sc2, tx, ty):
        M2 = cv2.getRotationMatrix2D((cw2 / 2, ch2 / 2), ang2, sc2)
        M2[0, 2] += tx / 2
        M2[1, 2] += ty / 2
        a2 = cv2.warpAffine(s2, M2, (cw2, ch2), borderValue=255)
        sv = a2[tmask2]
        svz = sv - sv.mean()
        den = denom_t * np.sqrt(float((svz * svz).sum()))
        if den <= 0:
            return -1.0
        return float((ta0 * svz).sum() / den)

    cand1 = []
    for ang2 in (ang - 0.5, ang, ang + 0.5):
        for sc2 in (sc - 0.02, sc, sc + 0.02):
            cand1.append((ncc2(ang2, sc2, tdx, tdy), ang2, sc2))
    cand1.sort(reverse=True)
    bq, b_ang, b_sc = cand1[0]
    b_x, b_y = tdx, tdy
    for step, rng in ((4, 16), (1, 3)):
        cx, cy = b_x, b_y
        for ddy in range(-rng, rng + 1, step):
            for ddx in range(-rng, rng + 1, step):
                q2 = ncc2(b_ang, b_sc, cx + ddx * 2, cy + ddy * 2)
                if q2 > bq:
                    bq, b_x, b_y = q2, cx + ddx * 2, cy + ddy * 2
    M = cv2.getRotationMatrix2D((tw / 2, th / 2), b_ang, b_sc)
    M[0, 2] += b_x
    M[1, 2] += b_y
    al0, q0 = _final_eval(scan, tmpl_img, M)
    return al0, q0, M


def _down(img, f):
    h, w = img.shape
    return cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)


def _coarse_grid(tmpl_img, scan, f0=8):
    """1/8 尺度旋转×缩放粗搜 (每档附带相位相关平移候选)。
    返回 (q, ang, sc, tdx, tdy)；无有效解时 q 为 None。
    独立出来供「两级选页」使用: 先对所有候选模板页粗搜排名, 再只对前几名
    做完整细化 —— 照片件整页 q 常年 0.3~0.5 达不到提前 break 线, 旧流程
    每个扫描页把 5 个候选全部完整对齐, SKM 十页卷实测对齐占总耗时 91%
    (cProfile: 146s/160s, phaseCorrelate 5400 次 + findTransformECC 100 次)。"""
    t8 = _down(tmpl_img, f0).astype(np.float32)
    s8 = _down(scan, f0).astype(np.float32)
    ch, cw = t8.shape
    h8, w8 = s8.shape
    tmask8 = t8 < 230
    ta0z = t8[tmask8] - t8[tmask8].mean()
    na8 = np.sqrt(float((ta0z * ta0z).sum()))

    def q8(Mat):
        a = cv2.warpAffine(s8, Mat, (cw, ch), borderValue=255)
        v = a[tmask8]
        vz = v - v.mean()
        den = na8 * np.sqrt(float((vz * vz).sum()))
        return float((ta0z * vz).sum() / den) if den > 0 else -1.0

    def eval_at(ang, sc):
        M8 = cv2.getRotationMatrix2D((w8 / 2, h8 / 2), ang, sc)
        warped = cv2.warpAffine(s8, M8, (cw, ch), borderValue=255)
        cands_t = [(0.0, 0.0)]
        try:
            (dx, dy), _resp = cv2.phaseCorrelate(t8, warped)
            cands_t.append((-dx, -dy))
        except cv2.error:
            pass
        bc = None
        for tx, ty in cands_t:
            M8t = M8.copy()
            M8t[0, 2] += tx
            M8t[1, 2] += ty
            q = q8(M8t)
            if bc is None or q > bc[0]:
                bc = (q, ang, sc, tx * f0, ty * f0)
        return bc

    # 两段式: 粗 5 角度(步长2°)×3 缩放(步长0.04) → 最优解邻域 3×3 补齐
    # 1°/0.02 网格。级联后覆盖范围与旧版 9×5 全枚举等价, 但只有 23 次
    # phaseCorrelate (旧版 45 次); 细化阶段还会在 ±1°、±0.02 内再搜一遍,
    # 因此粗搜只要落在正确解附近即可。
    best = None
    for ang in (-4, -2, 0, 2, 4):
        for sc in (0.96, 1.0, 1.04):
            r = eval_at(float(ang), sc)
            if r is not None and (best is None or r[0] > best[0]):
                best = r
    if best is None:
        return None, None, None, None, None
    b_ang, b_sc = best[1], best[2]
    for ang in (b_ang - 1, b_ang, b_ang + 1):
        for sc in (b_sc - 0.02, b_sc, b_sc + 0.02):
            ang = float(np.clip(ang, -4.0, 4.0))
            if abs(ang - b_ang) < 1e-9 and abs(sc - b_sc) < 1e-9:
                continue
            r = eval_at(ang, float(sc))
            if r is not None and r[0] > best[0]:
                best = r
    return best


def _ecc_refine(scan, tmpl_img, al, q, M):
    """ECC 仿射精修: 只对最终胜出的解做一次, 且 q 已足够高时直接跳过。

    旧版在 A/B 两条策略里各自无条件跑 200 次迭代 ECC (eps=1e-8), 照片件
    两策略都会执行 → 单候选 2 次 ECC, 十页卷累计 100 次/53s, 占配准 1/3。
    这里改为「谁胜出精修谁」, 迭代上限 120 (旧版 200, 实测 40~60 次收敛),
    精修后 q 没变好就丢弃 (与旧版 B 的取舍一致)。"""
    if al is None or M is None:
        return al, q, M
    fe = 4
    te = _down(tmpl_img, fe).astype(np.float32)
    se = _down(scan, fe).astype(np.float32)
    try:
        Me = np.asarray(M, dtype=np.float32).copy()
        Me[:, 2] /= fe
        Winv = cv2.invertAffineTransform(Me)
        cv2.findTransformECC(te, se, Winv, cv2.MOTION_AFFINE,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-7),
                             None, 3)
        M_ref = cv2.invertAffineTransform(Winv).astype(np.float64)
        M_ref[:, 2] *= fe
        al1, q1 = _final_eval(scan, tmpl_img, M_ref)
        if q1 > q:
            return al1, q1, M_ref
    except cv2.error:
        pass
    return al, q, M


def _refine_best(scan, tmpl_img, ang, sc, tdx, tdy, fast=False):
    """对已定粗搜解做细化: 策略 A(相位相关, 稀疏页更准) 与
    策略 B(平移网格, 周期表格不跳行) 各自细化, 统一用最终度量打分取优,
    最后对胜出解做一次 ECC。fast=True 时跳过策略 B (仅用于第 3 个及以后
    的备选模板页 —— 此时只为打破平局, 不值得再付一整轮网格搜索)。"""
    alA, qA, MA = (None, -1.0, None)
    try:
        alA, qA, MA = _refine_pc(scan, tmpl_img, ang, sc, tdx, tdy)
    except Exception:
        pass
    if alA is not None and qA >= 0.62:
        return _ecc_refine(scan, tmpl_img, alA, qA, MA)
    alB, qB, MB = (None, -1.0, None)
    if not fast:
        try:
            alB, qB, MB = _refine_grid(scan, tmpl_img, ang, sc, tdx, tdy)
        except Exception:
            pass
    if alA is None and alB is None:
        return None, 0.0, None
    if alB is None or qA >= qB:
        return _ecc_refine(scan, tmpl_img, alA, qA, MA)
    return _ecc_refine(scan, tmpl_img, alB, qB, MB)


def align_scan(scan_page, tmpl_img):
    """多分辨率配准: 粗尺度网格搜索(旋转/缩放)+平移估计, 再双策略细化择优。

    策略 A(相位相关)与策略 B(平移网格)各有所长: A 在稀疏印刷页上更准,
    B 在周期性表格版式上不会被相邻表格行吸引。两者统一用最终度量打分, 取更优者,
    因此任一版式都不会退步。"""
    scan = render_page(scan_page)
    q0, ang, sc, tdx, tdy = _coarse_grid(tmpl_img, scan)
    if q0 is None:
        return None, 0.0, None
    return _refine_best(scan, tmpl_img, ang, sc, tdx, tdy)


def _line_score(aligned):
    """结构对齐分: 页面中西部「窄峰」横线 (表格线) 个数。
    表格线对齐正确时在行投影里是 3~6px 窄峰; 旋转 ≥1° 后线在 ~60% 页宽的
    投影窗内散开 15px+, 峰矮于阈值而消失。对平移不敏感, 对旋转极敏感 ——
    用于 align_scan 低质量兜底时的候选裁决。灰底表头是宽块, 不计入。"""
    h, w = aligned.shape
    strip = aligned[:, int(w * 0.25):int(w * 0.85)]
    prof = (strip < 185).astype(np.float32).mean(axis=1)
    prof = cv2.blur(prof.reshape(-1, 1), (1, 3)).ravel()
    cnt, y, H = 0, 0, len(prof)
    while y < H:
        if prof[y] > 0.45:
            y2 = y
            while y2 < H and prof[y2] > 0.30:
                y2 += 1
            if y2 - y <= 9:
                cnt += 1
            y = y2 + 1
        else:
            y += 1
    return float(cnt)


# ---------------------------------------------------------------- 多尺度墨迹检测 + 形态学分析

def extra_ink(aligned, tmpl_dark_dil, rect, pad=5):
    """
    墨迹检测: 固定阈值差分 (基线算法, 实测在真实扫描件上最稳) + 归一化占比辅助.
    返回 (墨迹面积, 占区域百分比)
    """
    x0, y0, x1, y1 = [int(v * SCALE) for v in rect]
    h, w = aligned.shape
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return 0, 0.0
    ps = aligned[y0:y1, x0:x1]
    pt = tmpl_dark_dil[y0:y1, x0:x1] > 0
    extra = (ps < 150) & ~pt
    n, labels, stats, _ = cv2.connectedComponentsWithStats(extra.astype(np.uint8), 8)
    area = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 6:   # 去孤立噪点
            area += int(stats[i, cv2.CC_STAT_AREA])
    region = (x1 - x0) * (y1 - y0)
    return area, (area / max(region, 1)) * 100.0


def ink_shape(aligned, tmpl_dark_dil, rect, pad=5):
    """墨迹的几何形态, 用于区分「打的勾」和「反复涂抹的涂改团」。

    返回 (area, norm, stroke, fill, ncomp, bw, bh)
      stroke = 2*面积/周长 —— 近似笔画平均宽度。实测: 正常勾 ≈1.8~3.4,
               反复涂抹/打圈的涂改团 ≈7~9 (笔画重叠使周长远小于面积)
      fill   = 墨迹占其自身外接矩形的比例。勾 ≈5~14%, 涂改团 ≈26~33%
      ncomp  = 去噪后的连通域个数
      bw,bh  = 墨迹外接宽高 —— 区分「细长删除线」(长宽比>4.5) 与「勾」
    """
    x0, y0, x1, y1 = [int(v * SCALE) for v in rect]
    h, w = aligned.shape
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return 0, 0.0, 0.0, 0.0, 0, 0, 0
    ps = aligned[y0:y1, x0:x1]
    pt = tmpl_dark_dil[y0:y1, x0:x1] > 0
    extra = ((ps < 150) & ~pt).astype(np.uint8)
    # 边线残段剔除: 照片件的表格线与模板线轻微错位/弯曲时, 模板扣除会在
    # 格框边缘留下贴边残段 (SKM p2 m010 实测边框残段 415px 险胜真勾 364px
    # 导致错值)。清环深度自适应: 小格框 (广东卷选项格仅 35px 高) 不能深清,
    # 否则会吃掉压线书写的勾 (gd15 文化程度题实测回归)。
    mg = min(pad + 3, max(3, int(0.12 * min(extra.shape[0], extra.shape[1]))))
    extra[:mg, :] = 0
    extra[extra.shape[0] - mg:, :] = 0
    extra[:, :mg] = 0
    extra[:, extra.shape[1] - mg:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(extra, 8)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 6:
            continue
        bw_ = stats[i, cv2.CC_STAT_WIDTH]
        bh_ = stats[i, cv2.CC_STAT_HEIGHT]
        # 线样组件: 弓曲表格线的残段是「水平细长」或「垂直细长」。
        # 真勾是 V 形 (外接框两个维度都大); 圈选弧段、压线勾的残片两个
        # 维度也都不小, 不会误杀 (gd15 文化程度题实测)
        if (bw_ >= 25 and bh_ <= 10) or (bh_ >= 25 and bw_ <= 10):
            continue
        comps.append(i)
    if not comps:
        return 0, 0.0, 0.0, 0.0, 0, 0, 0
    area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in comps)
    mask = np.zeros_like(extra)
    for i in comps:
        mask[labels == i] = 1
    ys, xs = np.where(mask > 0)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    bbox = bw * bh
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    peri = sum(cv2.arcLength(c, True) for c in cnts)
    region = (x1 - x0) * (y1 - y0)
    return (area, (area / max(region, 1)) * 100.0,
            2 * area / max(peri, 1.0), (area / max(bbox, 1)) * 100.0, len(comps),
            bw, bh)


def decide_choice(q, tmpl_dark_dil, aligned, options_info, min_score=35, margin=40,
                  low_ink_flag=100, blank_ratio=0.35, low_ink_ratio=1.2, multi_ratio=1.25,
                  inset_x=0.12, inset_y=0.22,
                  smear_stroke=4.0, smear_fill=15.0, smear_area=150):
    """
    选项判定 v3: 主判据改为「墨迹占格比」norm(%) —— DPI 无关,
    200/300/400dpi 用同一套阈值, 扫描清晰度变化时不会整体漂移.
    绝对面积参数 (min_score/low_ink_flag/margin) 保留仅为兼容旧调用, 不再参与判定.
      1) best_norm < blank_ratio            → 未检出 (真空白, 阈值≈旧 35px² 在 200dpi 的等价比例。
                                               实测空白格噪声本底 <0.1 (过滤后残余碎屑), 0.35 高于本底
                                               3 倍以上; gd15 卷 p9/p14 实测淡勾 (压印刷字书写、被模板
                                               扣除吃掉大半) 落在 0.28~0.57 —— 阈值 0.6 会把它们静默
                                               丢弃, 降到 0.35 后进入「低墨迹待核」如实挂旗)
      2) 第二名也是真墨迹且与第一名接近      → 多选可疑 (相对差距, 排除噪点)
      3) blank_ratio ≤ best_norm < low_ink_ratio → 低墨迹待核 (边际带)
      4) 其余                               → OK
    返回 (chosen_value, status, best_score, detail)
    """
    scored = []
    for value, text, rect in options_info:
        area, norm, stroke, fill, ncomp, bw, bh = ink_shape(aligned, tmpl_dark_dil, rect)
        scored.append((value, text, area, norm, stroke, fill, ncomp, bw, bh))
    if not scored:
        return '', '未检出', 0, {}
    # 删除线排除: 学生划掉选项时留下一道细长直线(长宽比极端、笔画细),
    # 它的墨迹会跟真正的勾竞争甚至压过勾。细长直线不参与勾选判定。
    def _is_strike(s):
        bw, bh = s[7], s[8]
        if bw <= 0 or bh <= 0 or s[2] <= 0:
            return False
        aspect = max(bw, bh) / max(1, min(bw, bh))
        return aspect >= 4.5 and s[2] <= 3.5 and s[3] <= 12.0
    if len(scored) >= 2:
        kept = [s for s in scored if not _is_strike(s)]
        if kept and len(kept) < len(scored):
            scored = kept
    scored.sort(key=lambda t: -t[2])

    best_val, best_text, best_area, best_norm = scored[0][:4]
    second = scored[1] if len(scored) > 1 else ('', '', 0, 0.0, 0.0, 0.0, 0, 0, 0)
    second_area, second_norm = second[2], second[3]

    # 1) 真空白: 占比极低; 另设 10px² 绝对噪声兜底 (connectedComponents 已去 <6px 孤立点)
    if best_area < 10 or best_norm < blank_ratio:
        return '', '未检出', round(best_norm, 2), {'area': best_area, 'norm': best_norm}

    # 1.5) 涂改识别: 学生改答案时常把原先的勾反复涂抹成一个墨团,
    #      其墨迹面积远大于真正的勾, 会被误当成答案。形态判据——
    #      笔画平均宽度(stroke)与外接矩形填充率(fill)同时超阈 → 判定为涂改团。
    #      此时改选「非涂改」里墨迹最多的那个格子作为真实作答。
    def _is_smear(s):
        return s[2] >= smear_area and s[4] >= smear_stroke and s[5] >= smear_fill

    if _is_smear(scored[0]):
        alt = [s for s in scored[1:] if not _is_smear(s) and s[3] >= low_ink_ratio]
        alt.sort(key=lambda t: -t[2])
        detail = {'area': best_area, 'norm': best_norm,
                  'stroke': round(scored[0][4], 2), 'fill': round(scored[0][5], 1),
                  'smear_val': best_val}
        if alt:
            detail['second_val'] = alt[0][0]
            return str(alt[0][0]), '涂改已跳过', round(alt[0][3], 2), detail
        # 涂了改但没有在别处重新作答 → 不敢猜, 交人工
        return str(best_val), '疑似涂改待核', round(best_norm, 2), detail

    # 2) 多选可疑: 第二名必须也达到「边际墨迹」水平 (≥ low_ink_ratio, 排除格线噪点),
    #    且第一名与第二名差距小于相对阈值 (best < second*multi_ratio)
    import os as _os
    _dbg = _os.environ.get('QDEBUG')

    rect_of = {}
    for value, text, rect in options_info:
        rect_of[value] = rect

    def _ink_connected(va, vb):
        """两格的墨迹是否属于同一笔(同一连通分量跨过格边界)。
        勾的主干在一格、笔尾弧线伸进相邻格时, 两格各算出一份墨迹,
        但在联合区域里是同一个连通分量。"""
        ra, rb = rect_of.get(va), rect_of.get(vb)
        if not ra or not rb:
            return False
        ux0 = min(ra[0], rb[0]) - 2; uy0 = min(ra[1], rb[1]) - 2
        ux1 = max(ra[2], rb[2]) + 2; uy1 = max(ra[3], rb[3]) + 2
        ix0, iy0, ix1, iy1 = [int(v * SCALE) for v in (ux0, uy0, ux1, uy1)]
        h_, w_ = aligned.shape
        ix0, iy0 = max(0, ix0 - 5), max(0, iy0 - 5)
        ix1, iy1 = min(w_, ix1 + 5), min(h_, iy1 + 5)
        if ix1 <= ix0 or iy1 <= iy0:
            return False
        ps_u = aligned[iy0:iy1, ix0:ix1]
        pt_u = tmpl_dark_dil[iy0:iy1, ix0:ix1] > 0
        extra_u = ((ps_u < 150) & ~pt_u).astype(np.uint8)
        nn, ll, ss, _ = cv2.connectedComponentsWithStats(extra_u, 8)
        # 贴边残段不算连通证据 (与 ink_shape 同规则): 否则两格各自的
        # 边框残段会在联合区域里连成一笔, 把两个不相干的格子伪判为同笔
        mg_u = 8
        Wu, Hu = ix1 - ix0, iy1 - iy0
        bad = set()
        for i in range(1, nn):
            if ss[i, cv2.CC_STAT_AREA] < 6:
                continue
            lx = ss[i, cv2.CC_STAT_LEFT]; ly = ss[i, cv2.CC_STAT_TOP]
            lw = ss[i, cv2.CC_STAT_WIDTH]; lh = ss[i, cv2.CC_STAT_HEIGHT]
            if (lx + lw <= mg_u or lx >= Wu - mg_u or
                    ly + lh <= mg_u or ly >= Hu - mg_u):
                bad.add(i)

        def _labels_in(rc):
            gx0, gy0, gx1, gy1 = [int(v * SCALE) for v in rc]
            lx0 = max(ix0, gx0 - 5); ly0 = max(iy0, gy0 - 5)
            lx1 = min(ix1, gx1 + 5); ly1 = min(iy1, gy1 + 5)
            if lx1 <= lx0 or ly1 <= ly0:
                return set()
            sub = ll[ly0 - iy0:ly1 - iy0, lx0 - ix0:lx1 - ix0]
            return {i for i in range(1, nn)
                    if i not in bad and ss[i, cv2.CC_STAT_AREA] >= 6
                    and (sub == i).any()}

        return bool(_labels_in(ra) & _labels_in(rb))

    def _tick_like(s):
        return s[6] <= 2 and 1.2 <= s[4] <= 3.5 and s[5] >= 3.0

    def _noise_like(s):
        return s[6] >= 4

    if (len(scored) > 1 and second_norm >= low_ink_ratio
            and best_area < second_area * multi_ratio):
        if _dbg:
            print('  [QDBG] %s 多选分支 best=%s(%d) second=%s(%d)' %
                  (q.get('id'), best_val, best_area, second[0], second_area))
        # 抢救1: 勾尾跨格 —— 一个勾的主干在一格、笔尾弧线伸进相邻格,
        #   两格的墨迹其实是同一笔(同一连通分量跨过格边界)。
        #   合并判为一勾, 取墨迹多的格; 两格墨迹毫无连通才是真多选。
        if _ink_connected(best_val, second[0]):
            win = scored[0] if best_area >= second_area else second
            st2 = 'OK' if (win[3] >= low_ink_ratio or _tick_like(win)) else '低墨迹待核'
            if _dbg:
                print('  [QDBG]   连通合并 → %s 同一笔' % win[0])
            return str(win[0]), st2, round(win[3], 2), {
                'area': win[2], 'norm': win[3],
                'merged_with': second[0] if win[0] == best_val else best_val}
        # 抢救2: 碎片噪点 vs 紧凑勾 —— 印刷渗透/斑点会在格里碎成多个小连通域,
        #   真勾是 1~2 个紧凑笔画(笔画宽度/填充率都在勾的区间)。
        #   两者竞争时信形态: 若只有一方像勾, 勾的那格就是答案。
        if _noise_like(scored[0]) and _tick_like(second):
            st2 = 'OK' if second[3] >= low_ink_ratio else '低墨迹待核'
            if _dbg:
                print('  [QDBG]   形态择优 → second(%s) 是勾, best 是碎片噪点' % second[0])
            return str(second[0]), st2, round(second[3], 2), {
                'area': second[2], 'norm': second[3], 'noise_val': best_val}
        if _noise_like(second) and _tick_like(scored[0]):
            if _dbg:
                print('  [QDBG]   形态择优 → best(%s) 是勾, second 是碎片噪点' % best_val)
            return str(best_val), 'OK' if best_norm >= low_ink_ratio else '低墨迹待核', \
                round(best_norm, 2), {'area': best_area, 'norm': best_norm,
                                      'noise_val': second[0]}
        # 抢救3: 收边复测防误报: 相邻行的勾迹会从上下边渗进来; 而量表的勾常常画成细长弧线,
        # 横跨左右相邻两格的边界。故四周都收边: 左右各 inset_x、上下各 inset_y。
        inset = []
        for value, text, rect in options_info:
            x0, y0, x1, y1 = rect[0], rect[1], rect[2], rect[3]
            hh = (y1 - y0) * inset_y
            ww = (x1 - x0) * inset_x
            a2, n2 = extra_ink(aligned, tmpl_dark_dil, [x0 + ww, y0 + hh, x1 - ww, y1 - hh])
            inset.append((value, text, a2, n2))
        inset.sort(key=lambda t: -t[2])
        if len(inset) > 1 and inset[1][3] >= blank_ratio * 1.5:
            return str(best_val), '多选可疑', round(best_norm, 2), {
                'second_val': second[0], 'second_area': second_area,
                'norm': best_norm, 'second_norm': second_norm}
        # 收边后第二名只剩噪点水平 → 判定为邻行渗入, 不算多选, 继续按单选处理

    # 3) 边际带: 有墨迹但很弱 → 待核.
    #    (旧绝对阈值 100px² 在 200dpi 中格 ≈1.7%, 恰贴正常勾迹 p5=1.84%, 过于贴边)
    if best_norm < low_ink_ratio:
        if _dbg:
            print('  [QDBG] %s 低墨迹带 all=%s' %
                  (q.get('id'), [(s[0], s[2], round(s[3], 2), round(s[4], 2),
                                  round(s[5], 1), s[6]) for s in scored]))
        # 抢救4: 淡而清楚的勾 —— 形态是勾(笔画宽度/填充率都在勾的区间), 只是下笔轻。
        #   只要没有「另一个像勾的竞争者」(其余格的墨迹也是勾形态 → 真歧义),
        #   就高置信采纳, 不再刷待核。弥散噪点(碎片多/填充率极低)不构成竞争。
        tick_like = (1.2 <= scored[0][4] <= 4.5) and scored[0][5] <= 22.0
        second_tick = (len(scored) > 1 and second_norm >= blank_ratio
                       and (1.2 <= second[4] <= 4.5) and second[5] <= 22.0)
        # 抢救5: 低墨迹版的勾尾跨格 —— 淡铅笔勾主干在一格、笔尾伸进邻格,
        #   邻格的淡墨迹(够不到多选分支的门槛)其实是同一笔 → 合并取主干格。
        if second_tick and _ink_connected(best_val, second[0]):
            win = scored[0] if best_area >= second_area else second
            win_tick = (1.2 <= win[4] <= 4.5) and win[5] <= 22.0
            if _dbg:
                print('  [QDBG]   低墨迹连通合并 → %s 同一笔' % win[0])
            if win_tick:
                return str(win[0]), 'OK', round(win[3], 2), {
                    'area': win[2], 'norm': win[3],
                    'merged_with': second[0] if win[0] == best_val else best_val}
        if tick_like and not second_tick:
            return str(best_val), 'OK', round(best_norm, 2), {
                'area': best_area, 'norm': best_norm, 'faint': 1}
        return str(best_val), '低墨迹待核', round(best_norm, 2), {'area': best_area, 'norm': best_norm}

    return str(best_val), 'OK', round(best_norm, 2), {'area': best_area, 'norm': best_norm}


# ---------------------------------------------------------------- 填空题识别
DIGIT_ANCHOR_RE = re.compile(r'学号|年龄|出生|年月|共有|排行|班|编号|人数|数量|年|答案')
_DIGIT_CLF = None
_DIGIT_CLF_TRIED = False
_DIGIT_LOCK = threading.Lock()


def get_digit_clf():
    """懒加载手写数字模型 (优先 joblib, 缺失时回退 pickle)。

    多线程下两个请求可能同时进入懒加载, 重复解析 5.4MB 模型 (内存翻倍、
    首次识别变慢), 因此加锁; 失败标记 _DIGIT_CLF_TRIED 仍在加载结束后才置位。"""
    global _DIGIT_CLF, _DIGIT_CLF_TRIED
    if _DIGIT_CLF is not None or _DIGIT_CLF_TRIED:
        return _DIGIT_CLF
    with _DIGIT_LOCK:
        if _DIGIT_CLF is not None or _DIGIT_CLF_TRIED:
            return _DIGIT_CLF
        _DIGIT_CLF = _load_digit_clf()
        _DIGIT_CLF_TRIED = True
        return _DIGIT_CLF


def _load_digit_clf():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '手写数字模型.pkl')
    if not os.path.exists(p):
        print('  [填空] 未找到 手写数字模型.pkl, 填空题仅截图供人工录入')
        return None
    clf = None
    try:
        import joblib
        clf = joblib.load(p)
    except ImportError:
        try:
            import pickle
            clf = pickle.load(open(p, 'rb'))
        except Exception as e:
            print('  [填空] 数字模型加载失败(%s), 需安装 scikit-learn: pip install scikit-learn' % type(e).__name__)
            return None
    except Exception as e:
        print('  [填空] 数字模型加载失败(%s)' % type(e).__name__)
        return None
    if clf is not None:
        print('  [填空] 手写数字模型已加载, 数字类填空将自动识别')
    return clf


def _digit_to_canvas_v2(mask_bin, target=28):
    """
    v2.0 手写数字预处理: 更好的归一化 + 居中 + 抗锯齿
    """
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return None
    sub = mask_bin[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255.0
    mh, mw = sub.shape
    # 保持宽高比缩放到 target-4
    side = target - 4
    scale = side / max(mh, mw)
    new_h, new_w = max(1, round(mh * scale)), max(1, round(mw * scale))
    sub = cv2.resize(sub, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 重新二值化 (resize 会产生灰度)
    sub = (sub > 0.35).astype(np.float32)
    # 居中
    canvas = np.zeros((target, target), np.float32)
    yy, xx = (target - new_h) // 2, (target - new_w) // 2
    canvas[yy:yy + new_h, xx:xx + new_w] = sub
    # 轻微模糊平滑笔画抗锯齿
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.4)
    return canvas


DIGIT_LEN_HINT = {'出生日期': (4, 4), '年': (1, 2), '月': (1, 2), '年龄': (1, 3),
                  '班级': (1, 3), '班别': (1, 4), '学号': (8, 12)}

# 形近字符→数字（手写 9 常被 OCR 读成 ) 或 g，7→T 等）
_VISION_SYM2DIG = {'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'U': '0',
                   'I': '1', 'L': '1', 'l': '1', '|': '1', 'T': '7',
                   'Z': '2', 'z': '2', 'S': '5', 's': '5', 'B': '8',
                   'G': '6', 'b': '6', 'g': '9', 'q': '9', 'A': '4', '%': '9', ')': '9'}
_VISION_TRY = [False, True]  # [已试过, 可用]


def vision_read_digits(mask, raw=None):
    """macOS Vision 离线读数字, 返回归一化数字串; 不可用/读不出返回 None。
    raw: 同尺寸的原始灰度裁图 —— 保留真实笔画灰度、只保留笔迹区(印刷抹白),
    实测比二值掩码稳定得多。可选依赖: pip install pyobjc-framework-Vision (仅 macOS)。"""
    if not _VISION_TRY[1] or not mask.size:
        return None
    try:
        import Quartz
        import Vision
        from Foundation import NSData
    except Exception:
        _VISION_TRY[1] = False
        return None
    try:
        if raw is not None and raw.shape == mask.shape:
            fm = cv2.dilate((mask > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            img = np.full(mask.shape, 255, np.uint8)
            img[fm] = raw[fm]
        else:
            g = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.2)
            img = (255 * g / max(1e-6, g.max())).astype(np.uint8)
        h, w = img.shape
        big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 28, 28, 28, 28, cv2.BORDER_CONSTANT, value=255)
        ok, buf = cv2.imencode('.png', big)
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setRecognitionLanguages_(["en-US"])
        req.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        handler.performRequests_error_([req], None)
        texts = [o.topCandidates_(1)[0].string() for o in (req.results() or [])]
        best = None
        for t in texts:
            d = ''.join(_VISION_SYM2DIG.get(ch, ch) for ch in t)
            d = ''.join(ch for ch in d if ch.isdigit())
            if best is None or len(d) > len(best):
                best = d
        return best or None
    except Exception:
        return None


def recognize_digit_mask(mask, expected=None):
    """输入墨迹掩码(uint8, 255=笔迹), 返回 (数字串, 各位置信度)。
    expected=(min,max): 锚点给出的期望位数。
      - 位数偏少且存在宽字符(连笔/贴写) → 按竖直投影谷点拆开重认;
      - 连通域 x 间距超过 0.9×字高视为另一簇(邻行/邻栏泄漏), 只留面积最大簇。
    2026-11-02 修: 合并 bbox 后原来只取单个连通域像素, 贴写的两位数会丢一半笔画。"""
    clf = get_digit_clf()
    if clf is None or not mask.size:
        return '', []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 10 or h < 6:
            continue
        comps.append([x, y, w, h, i])
    comps.sort(key=lambda c: c[0])
    merged = []
    for c in comps:
        if merged and c[0] < merged[-1][0] + merged[-1][2] - 3:
            m = merged[-1]
            m[4].append(c[4])
            nx0, ny0 = min(m[0], c[0]), min(m[1], c[1])
            nx1 = max(m[0] + m[2], c[0] + c[2])
            ny1 = max(m[1] + m[3], c[1] + c[3])
            m[0], m[1], m[2], m[3] = nx0, ny0, nx1 - nx0, ny1 - ny0
        else:
            merged.append([c[0], c[1], c[2], c[3], [c[4]]])
    if not merged:
        return '', []
    H = max(c[3] for c in merged)
    comps = [c for c in merged if c[3] >= H * 0.35]
    if not comps:
        return '', []

    # 聚簇: 只保留面积最大的簇, 去掉远处的泄漏笔迹(邻行勾选、隔壁栏)
    if len(comps) > 1:
        clusters = [[comps[0]]]
        for c in comps[1:]:
            if c[0] - (clusters[-1][-1][0] + clusters[-1][-1][2]) > H * 0.9:
                clusters.append([c])
            else:
                clusters[-1].append(c)
        if len(clusters) > 1:
            comps = max(clusters, key=lambda cl: sum(c[2] * c[3] for c in cl))

    items = []
    for x, y, w, h, labs in comps:
        sub = np.isin(labels[y:y + h, x:x + w], labs).astype(np.uint8) * 255
        items.append({'x': x, 'img': sub})

    def classify(img):
        canvas = _digit_to_canvas_v2(img)
        if canvas is None:
            return '?', 0.0
        try:
            proba = clf.predict_proba(canvas.reshape(1, -1))[0]
            j = int(np.argmax(proba))
            return str(j), float(proba[j])
        except Exception:
            return '?', 0.0

    # 位数不足 → 反复拆最宽的字符(宽高比 ≥1.35)直到位数达标或无可拆
    if expected:
        guard = 0
        while len(items) < expected[0] and guard < 10:
            guard += 1
            cand = max(items, key=lambda t: t['img'].shape[1] / max(1, t['img'].shape[0]))
            w_, h_ = cand['img'].shape[1], cand['img'].shape[0]
            if w_ < h_ * 1.35:
                break
            sub = cand['img']
            col = (sub > 0).sum(axis=0)
            lo, hi = int(w_ * 0.3), int(w_ * 0.7)
            k = lo + int(np.argmin(col[lo:hi]))
            if col[k] > 0.25 * col.max():
                break
            parts = []
            for seg in (sub[:, :k], sub[:, k:]):
                ys, xs = np.where(seg > 0)
                if len(xs) < 8:
                    parts = None
                    break
                parts.append(seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
            if not parts or len(parts) < 2:
                break
            items.remove(cand)
            items.append({'x': cand['x'], 'img': parts[0]})
            items.append({'x': cand['x'] + k, 'img': parts[1]})
            items.sort(key=lambda t: t['x'])

    out, confs = '', []
    for it in items:
        v, p = classify(it['img'])
        out += v
        confs.append(p)
    return out, confs


def fill_value_status(anchor, fill_mask, overflow=False, raw=None):
    """根据锚点类型与墨迹掩码识别填空, 返回 (值, 状态)。
    overflow=True: 笔迹超出下划线矩形（无格约束的手写），数字识别结果不可信。
    raw: 原始灰度裁图（供 macOS Vision 双模型用）。"""
    ink = int((fill_mask > 0).sum())
    if ink <= 30:
        return '', '空白?'
    if not DIGIT_ANCHOR_RE.search(anchor or ''):
        return '', '手写内容,请对照截图录入'
    hint = None
    for k, rng in DIGIT_LEN_HINT.items():
        if k in (anchor or ''):
            hint = rng
            break
    val_m, confs = recognize_digit_mask(fill_mask, expected=hint)
    # 双模型: 本地 MNIST + macOS Vision (离线, 原始灰度)。
    #   一致 → 强证据, 自动采纳;
    #   MNIST 不够硬(min conf<0.98)且 Vision 位数符合先验 → 采纳 Vision 值为建议, 人工核对;
    #   MNIST 高置信(≥0.98)且位数吻合 → 以 MNIST 为准 (Vision 偶发抖动, 不覆盖)。
    val = val_m
    vval = vision_read_digits(fill_mask, raw) if val_m else None
    dual = bool(vval is not None and val_m and vval == val_m)
    if vval is not None and not dual and val_m:
        m_hard = bool(confs and min(confs) >= 0.98 and hint
                      and hint[0] <= len(val_m) <= hint[1])
        v_ok = bool(hint and hint[0] <= len(vval) <= hint[1])
        if v_ok and not m_hard:
            val = vval
    if not val:
        return '', '未识别出,请手工录入'
    len_ok = bool(hint and hint[0] <= len(val) <= hint[1])
    conf_ok = dual or bool(val == val_m and confs and min(confs) >= 0.9)
    # 自动采纳必须同时满足: 笔迹未溢出 + 位数符合锚点先验 + (双模型一致 或 置信 ≥0.9)。
    # 真实扫描实测（广东卷第一页）无格手写数字错误率高（3→5、2009→2、
    # 10 位学号被下划线截断），低门槛会静默采纳错值。
    if overflow or not len_ok or not conf_ok:
        return val, '低置信(自动识别),请核对'
    # 注意: 这里不带「请核对」字样 —— 网页端据此区分「高把握自动识别」(conf 0.82,
    # 不进核验列表) 和「低置信」(conf 0.6, 进核验列表)
    return val, 'OK(自动识别)'


# ---------------------------------------------------------------- 识别主流程

_BOOK_CACHE = {}
_BOOK_LOCK = threading.Lock()
_BOOK_CACHE_MAX = 4  # 每本 ≈ 页数 × (渲染图 4MB + ORB 特征), 不设限会随题本数无限增长


def write_csv_safe(path, header, body):
    """写 CSV; 目标被 Excel/WPS 占用时退到 <path>.bak, 返回实际落盘路径。

    旧实现只捕获 PermissionError 然后静默跳过, 上层却仍按固定文件名读取,
    结果是「识别跑完了, 但服务返回 500 FileNotFoundError」。现在保证一定
    有文件落地, 上层按 name_识别结果*.csv 兜底查找即可。"""
    for p in (path, path + '.bak'):
        try:
            with open(p, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(header)
                for r in body:
                    w.writerow(r)
            if p != path:
                print(f'  提示: {path} 正被占用, 结果已改写到 {os.path.basename(p)}')
            return p
        except OSError as e:
            err = e
    print(f'  警告: 结果 CSV 写入失败({err}), 该文件将由上层按需重新生成')
    return None


def get_book_feats(tmpl, book_path):
    """全局缓存题本页渲染图+ORB特征+pHash。

    服务端是多线程 (ThreadingHTTPServer), 两个请求同时首次加载同一本书时,
    旧实现会在遍历 _BOOK_CACHE 的字典上并发写入 → RuntimeError: dictionary
    changed size during iteration。这里加锁并做二次检查, 同时顺便去掉重复的
    渲染/ORB 计算。"""
    key = os.path.abspath(book_path)
    hit = _BOOK_CACHE.get(key)
    if hit is not None:
        return hit
    with _BOOK_LOCK:
        hit = _BOOK_CACHE.get(key)
        if hit is not None:
            return hit
        orb_pg = cv2.ORB_create(nfeatures=3000)
        bf_pg = cv2.BFMatcher(cv2.NORM_HAMMING)
        tmpl_doc = pymupdf.open(book_path)
        feats = {}
        for pmeta in tmpl['pages']:
            tp = pmeta['page'] - 1
            img1 = render_page(tmpl_doc[tp])
            kp1, d1 = orb_pg.detectAndCompute(img1, None)
            ph = phash(img1)
            feats[pmeta['page']] = (img1, kp1, d1, pmeta, ph)
        _BOOK_CACHE[key] = (feats, orb_pg, bf_pg)
        # 简单 FIFO 上限: 正在使用的一方持有返回值引用, 淘汰只影响后续请求重新构建
        while len(_BOOK_CACHE) > _BOOK_CACHE_MAX:
            _BOOK_CACHE.pop(next(iter(_BOOK_CACHE)))
        return _BOOK_CACHE[key]


def _detect_row_bands(aligned, xs0, xs1, pitch):
    """在选项列 x 范围内检测水平表格线, 返回相邻线对构成的行带中心列表 (pt)。
    照片件的整页仿射常带残余剪切 —— 题干列对齐时选项列可差半行, 题干补丁
    的 dy 不能代表选项行; 而表格横线在选项列 x 范围内直接给出行的真实位置,
    对剪切/缩放/模糊全部免疫。"""
    x0 = max(0, int(xs0 * SCALE))
    x1 = min(aligned.shape[1], int(xs1 * SCALE))
    if x1 - x0 < 40:
        return []
    strip = aligned[:, x0:x1]
    bw = (strip < 185).astype(np.float32)
    prof = bw.mean(axis=1)
    lines = []
    y, H = 0, len(prof)
    thr = 0.45
    while y < H:
        if prof[y] > thr:
            y2 = y
            while y2 < H and prof[y2] > thr * 0.55:
                y2 += 1
            lines.append((y + y2) / 2.0 / SCALE)
            y = y2 + 2
        else:
            y += 1
    # 行带 = 高度接近行距的相邻线对
    bands = []
    for a, b in zip(lines, lines[1:]):
        h = b - a
        if pitch * 0.45 <= h <= pitch * 1.7:
            bands.append((a + b) / 2.0)
    return bands


def _pair_row_dys(tpl_ys, bands, pitch, prior=None):
    """把模板行中心与检测行带做单调整体配对 (DP), 返回 {行序号: dy} 或 None。
    允许: 整体错位、缺行 (照片裁切溢出/裁掉)、跨表格段的偏移突变 —— DP 最小化
    相邻行残差突变, 跳带 ≤3。
    关键1: 模板行也允许「跳过」(罚分 0.5*行距) —— 照片底部裁掉量表尾行时,
    若强制每行都必须配带, DP 会把整个量表前移若干带去吃表头/说明区的带
    (SKM 卷 p10 实测整体错 2 行且残差平滑, 静默无告警)。允许跳行后,
    尾部缺行如实留空, 由「行定位不可靠」旗标兜底。
    关键2: prior = {行序号: 题干NCC dy} 提供绝对位置先验 —— DP 只看残差
    平滑度, 对「整页平移若干行」盲视 (顶部多一行/底部缺几行时, 平移路径
    的跳行罚分可能更低)。对带可靠票的行, 偏离先验超过剪切容差就罚分,
    罚分与跳行罚分同量级, 平移路径不再占优; 题干列与选项列间的剪切残差
    (实测可达 ~13pt) 由 0.35*行距 容差吸收。
    配对后剔除偏离中位数的离盛行, 剩余不足则放弃。"""
    n, m = len(tpl_ys), len(bands)
    if m < 3 or n < 3:
        return None
    SKIP = 0.5 * pitch
    shear_tol = 0.45 * pitch   # 题干列与选项列间的剪切残差 (实测可达 ~13pt)
    cap_pen = 2.0 * pitch

    def _pen(i, j):
        """行 i 配到带 j 的先验罚分 (无票则 0)。
        prior[i] = (题干NCC dy, 权重): 强票(rmx≥0.55)权重1, 弱票(≥0.45)权重0.5。
        p2 实测全部票 rmx≈0.5 但 10 行高度一致 —— 弱而一致的证据足以
        排除「整页平移一行」的路径, 不该被抛弃。"""
        if prior is None or i not in prior:
            return 0.0
        dyv, w = prior[i]
        d = abs(bands[j] - (tpl_ys[i] + dyv)) - shear_tol
        return w * min(max(0.0, d), cap_pen)

    INF = 1e18
    W = m + 1  # 第 m 列 = 虚拟「尚未匹配过任何带」态 (开头连续跳行)
    # dp[i][j]: 前 i+1 行处理完、最后匹配行落在带 j 的最小代价;
    # rl[i][j]: 该状态下最后匹配行的残差 (band - tpl), 未匹配过/跳行时继承
    dp = [[INF] * W for _ in range(n)]
    rl = [[None] * W for _ in range(n)]
    pv = [[None] * W for _ in range(n)]  # (前状态j, 本行是否匹配)
    for j in range(m):
        dp[0][j] = _pen(0, j)
        rl[0][j] = bands[j] - tpl_ys[0]
        pv[0][j] = (j, True)
    dp[0][m] = SKIP
    pv[0][m] = (m, False)
    for i in range(1, n):
        for j in range(m):
            best, bj, bmatch, brl = INF, -1, True, None
            # 匹配本行到带 j: 前一匹配行在 j0 (带序严格递增, 跳带≤3),
            # 或从「未匹配态」首次匹配 (任意带, 无残差基线)
            for j0 in list(range(max(0, j - 4), j)) + [m]:
                if dp[i - 1][j0] >= INF:
                    continue
                r0 = rl[i - 1][j0]
                cost = dp[i - 1][j0] if r0 is None else \
                    dp[i - 1][j0] + abs((bands[j] - tpl_ys[i]) - r0)
                if cost < best:
                    best, bj, brl = cost, j0, bands[j] - tpl_ys[i]
            best += _pen(i, j)
            # 跳过本行 (带不变, 罚分)
            if j < m and dp[i - 1][j] + SKIP < best:
                best, bj, bmatch, brl = dp[i - 1][j] + SKIP, j, False, rl[i - 1][j]
            dp[i][j] = best
            rl[i][j] = brl
            pv[i][j] = (bj, bmatch)
        # 虚拟态: 继续不匹配
        dp[i][m] = dp[i - 1][m] + SKIP
        rl[i][m] = None
        pv[i][m] = (m, False)
    j = min(range(W), key=lambda x: dp[n - 1][x])
    if dp[n - 1][j] >= INF:
        return None
    matched = {}
    matched_band = {}
    for i in range(n - 1, -1, -1):
        pj, was_match = pv[i][j]
        if was_match:
            matched[i] = bands[j] - tpl_ys[i]
            matched_band[i] = j
        j = pj
    if len(matched) < max(3, n - 3):
        return None
    med = float(np.median(list(matched.values())))
    matched = {i: d for i, d in matched.items() if abs(d - med) <= 0.6 * pitch}
    if len(matched) < max(3, n - 3):
        return None
    # 二次补配: DP 的跳带窗口 (≤3) 跨不过矩阵表分段之间的大空白区时
    # (gd15 卷 p7 实测: 段间空白区检出 ~5 个说明文字假带, 段边界两行被
    # 白白跳过, 而空闲带与其行中心残差 <1pt), 对未匹配行做一次保守补配 ——
    # 只在「未被占用的带」里找最近的, 且要求: 残差 ≤0.45*行距、带序与两侧
    # 已配行保持单调。三个条件都不过就宁可留空 (如实挂旗, 不硬配)。
    used = set(matched_band.values())
    for i in range(n):
        if i in matched:
            continue
        exp = tpl_ys[i] + (prior[i][0] if prior and i in prior else med)
        left_b = max((matched_band[k] for k in matched if k < i), default=-1)
        right_b = min((matched_band[k] for k in matched if k > i), default=m)
        cand, cand_d = None, 0.45 * pitch
        for jb in range(left_b + 1, right_b):
            if jb in used:
                continue
            d = abs(bands[jb] - exp)
            if d <= cand_d:
                cand, cand_d = jb, d
        if cand is not None:
            matched[i] = bands[cand] - tpl_ys[i]
            matched_band[i] = cand
            used.add(cand)
    return matched


def _stem_patch(tmpl_img, q, pitch):
    """构造某题的行定位补丁 —— 只用「唯一内容」: 表格式(PBI)题干在同行左侧 →
    取左侧题干条; 竖排题干式(GAD)题干在选项带上方一行 → 取上方题干条。
    千万别把选项strip卷进来 —— 矩阵表每行选项印刷内容完全相同 (1 2 3 4),
    相关面会周期性起峰, 主峰/次峰只差 0.02, 歧义检测拦不住。
    返回 (rt0, rt1, rc0, rc1, rpatch, ys0)。"""
    xs0 = min(o['rect'][0] for o in q['options'])
    ys0 = min(o['rect'][1] for o in q['options'])
    xs1 = max(o['rect'][2] for o in q['options'])
    ys1 = max(o['rect'][3] for o in q['options'])
    stemy0 = max(0, ys0 - max(18.0, pitch * 0.62) if pitch else ys0 - 24)
    stemx0 = max(0, xs0 - min(240.0, max(0.0, xs0 - 20)))
    if xs0 > 150:
        pt0 = max(0, int((ys0 - 4) * SCALE))
        pt1 = int((ys1 + 4) * SCALE)
        pc0 = int(stemx0 * SCALE)
        pc1 = max(pc0 + 4, int((xs0 - 6) * SCALE))
    else:
        pt0 = max(0, int(stemy0 * SCALE))
        pt1 = max(pt0 + 4, int((ys0 - 2) * SCALE))
        pc0 = int(stemx0 * SCALE)
        pc1 = int((xs1 + 10) * SCALE)
    return pt0, pt1, pc0, pc1, tmpl_img[pt0:pt1, pc0:pc1], ys0


def _row_relocate(rt0, rt1, rc0, rc1, aligned, pitch, rpatch, cap=None, full=False,
                  center=None):
    """在配准后的扫描页 aligned 上为模板补丁 rpatch 找实际落点。
    返回 (dy_pt, dx_pt, rmx, ok)。补丁必须含题干 (每行唯一) —— 只用选项格
    区域时矩阵表每行印刷内容相同, 匹配歧义会系统性吸到相邻行 (SKM 卷实测)。
    cap: 允许的 |dy| 上限(pt); None=按行距推断; 负值=不限制 (跨页全页搜索用)。
    full: True 时全页搜索 (页级投票/跨页重试, 无邻域先验)。
    center: 给定页级全局偏移时只在 center±0.6*行距 邻域内搜索 —— 劣质配准
    (照片件) 下全页 NCC 峰值嘈杂, 几何约束是排除噪声峰的关键。"""
    if not rpatch.size or rpatch.std() <= 8:
        return 0.0, 0.0, 0.0, False
    if full:
        sy0, sy1 = 0, aligned.shape[0]
        sx0, sx1 = 0, aligned.shape[1]
    else:
        if center is not None and pitch:
            hw = max(0.6 * pitch, 20.0)
            sy0 = max(0, int((rt0 / SCALE + center - hw) * SCALE))
            sy1 = min(aligned.shape[0], int((rt1 / SCALE + center + hw) * SCALE))
        else:
            # 搜索窗覆盖整页仿射残余; 缩放不一致时逐行累积漂移可接近一个行距
            row_range = max(130, int(pitch * 1.6 * SCALE)) if pitch else 200
            sy0 = max(0, rt0 - row_range)
            sy1 = min(aligned.shape[0], rt1 + row_range)
        sx0 = max(0, rc0 - 14)
        sx1 = min(aligned.shape[1], rc1 + 14)
    sr = aligned[sy0:sy1, sx0:sx1]
    if sr.shape[0] < rpatch.shape[0] or sr.shape[1] < rpatch.shape[1]:
        return 0.0, 0.0, 0.0, False
    try:
        rr = cv2.matchTemplate(sr, rpatch, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return 0.0, 0.0, 0.0, False
    _, rmx, _, rloc = cv2.minMaxLoc(rr)
    # 歧义检测: 屏蔽主峰邻域找次峰, 次峰接近主峰说明补丁不唯一, 不冒险。
    # 屏蔽半径须随补丁尺寸缩放 —— 大补丁的相关面有宽裙边, 固定 ±6px 会把
    # 主峰自己的裙边当成次峰, 永远误判歧义
    rr2 = rr.copy()
    cy, cx = rloc[1], rloc[0]
    er = max(8, int(0.3 * min(rpatch.shape[:2])))
    cv2.rectangle(rr2, (max(0, cx - er), max(0, cy - er)),
                  (min(rr2.shape[1] - 1, cx + er),
                   min(rr2.shape[0] - 1, cy + er)), -1, -1)
    _, mx2, _, _ = cv2.minMaxLoc(rr2)
    dy_pt = (sy0 + rloc[1] - rt0) / SCALE
    dx_pt = (sx0 + rloc[0] - rc0) / SCALE
    if cap is None:
        cap = max(1.6 * pitch, 70.0) if pitch else 70.0
    # rmx 绝对阈值只设 0.30: 劣质扫描 (照片模糊/配准质量 0.3) 上正确行的
    # NCC 也只有 0.3~0.45, 0.55 会把整页全部拒掉; 真正的防错依据是
    # 「次峰差距 ≥0.05」+ 题干补丁唯一 + 行序单调性 (调用方校验)
    ok = rmx > 0.30 and mx2 < rmx - 0.05 and (cap < 0 or abs(dy_pt) <= cap)
    return dy_pt, dx_pt, float(rmx), bool(ok)


def _stem_sim(a, b):
    """题干文本相似度 (字符 bigram Jaccard)。用于跨页重试的平行表防护:
    父亲/母亲量表同号题干只差一个字, NCC 分辨不了, 文本上却能。"""
    a = re.sub(r'\s+', '', str(a or ''))
    b = re.sub(r'\s+', '', str(b or ''))
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def _refine_cells(options_info, tmpl_img, aligned, vy=14):
    """逐格模板匹配微调 (±12pt 补丁), 返回调整后的选项列表。
    vy: 垂直搜索半窗(px)。默认 14 (±7pt); 照片件存在题干列/选项列的剪切残差
    时传入 ±0.55*行距 —— 印刷数字行距 44pt 远大于残差, 「最近数字行」必是
    正确行, 吸附不会串行。"""
    adjusted = []
    for value, text, rect in options_info:
        r = list(rect)
        tx0, ty0 = max(0, int((r[0] - 12) * SCALE)), max(0, int((r[1] - 12) * SCALE))
        tx1, ty1 = int((r[2] + 12) * SCALE), int((r[3] + 12) * SCALE)
        tpatch = tmpl_img[ty0:ty1, tx0:tx1]
        if tpatch.size and tpatch.std() > 8:
            sx0, sy0 = max(0, tx0 - 14), max(0, ty0 - vy)
            sregion = aligned[sy0:ty1 + vy, sx0:tx1 + 14]
            if sregion.size and sregion.shape[0] >= tpatch.shape[0] \
                    and sregion.shape[1] >= tpatch.shape[1]:
                try:
                    res = cv2.matchTemplate(sregion, tpatch, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, ml = cv2.minMaxLoc(res)
                    if mx > 0.4:
                        dx = (ml[0] - (tx0 - sx0)) / SCALE
                        dy = (ml[1] - (ty0 - sy0)) / SCALE
                        r = [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy]
                except cv2.error:
                    pass
        adjusted.append((value, text, r))
    return adjusted


def recognize_scan(scan_path, tmpl, book_path, out_dir, min_score=35, margin=40):
    name = os.path.splitext(os.path.basename(scan_path))[0]
    res_dir = os.path.join(out_dir, name)
    crops_dir = os.path.join(res_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    scan_doc = pymupdf.open(scan_path)

    rows = []
    fill_rows = []
    # 网页端「查看识别框」用：每页的「扫描像素→模板像素」仿射 + 每题实测格子坐标
    # （格子是逐行重定位/逐格模板匹配之后的真实位置，画出来才能贴住扫描件）
    page_reg = {}
    cell_reg = {}

    tmpl_feats, orb_pg, bf_pg = get_book_feats(tmpl, book_path)
    used_pages = set()
    # 各扫描页的配准结果 (供跨页行重试: 扫描件分页与题本不一致时,
    # 某行会落到相邻扫描页上 —— 照片裁切偏移实测存在)
    aligned_pages = {}
    retry_cands = []
    # 上一扫描页的配准结果 (顺序先验快通道: 多页卷相邻页版式/设备一致)
    prev_geo = None

    for si in range(len(scan_doc)):
        simg = render_page(scan_doc[si])
        simg_enh = simg  # 候选页匹配也用原图: 与模板特征口径一致 (CLAHE 会改变纹理导致错页)
        kp2, d2 = orb_pg.detectAndCompute(simg_enh, None)
        cands = []
        if d2 is not None and len(kp2) >= 30:
            for pno, (img1, kp1, d1, pmeta, ph1) in tmpl_feats.items():
                if d1 is None:
                    continue
                matches = bf_pg.knnMatch(d2, d1, k=2)
                good = sum(1 for m, n in matches if m.distance < 0.75 * n.distance)
                cands.append((good, pno))
            cands.sort(reverse=True)
        cands = cands[:5] if cands else [(0, p['page']) for p in tmpl['pages'][:1]]

        # ---- 三级选页 (性能优化; 择优规则与旧版一致) ----
        # 旧版对 5 个候选页逐个「完整对齐」(粗搜+双策略细化+ECC, ~3s/个)。
        # 清晰件第 1 名即达 0.55 提前 break 没问题; 照片件整页 q 常年
        # 0.3~0.5 永远达不到 break 线 → 每页 5 次完整对齐, SKM 十页卷
        # 实测对齐占总耗时 91%。现改为:
        #   ① 顺序先验快通道: 相邻页通常同一设备同一版式扫出, 直接把上一页的
        #      仿射套过来, 一次 warp+打分确认 (q≥0.55 即采纳, ~0.05s);
        #   ② ORB 第 1 名完整对齐 (清晰件在这一步 strong 退出, 与旧版同速);
        #   ③ 仍未达强通过线 → 其余候选先 1/8 粗搜排名 (~0.15s/个), 按
        #      (未用页优先, 粗分降序) 逐个细化, 最多 3 个; 第 3 个起跳过策略 B;
        #      并有两条剪枝: 粗分落后最优 0.12 以上直接放弃; 已得可用解且
        #      粗分领先 0.04 或正好是顺序下一页 → 早停(照片件主要收益在此)。
        # 最终择优规则不变: (未用页优先, 细化后 q 降序), pHash 校验保留。
        def _consider(pno, aligned, quality, align_M):
            nonlocal chosen
            img1, kp1, d1, pmeta, ph1 = tmpl_feats[pno]
            q = quality if aligned is not None else -1
            # pHash 只在中等质量时温和校验: 高质量 (>0.6) 不瞎怀疑,
            # 极低质量 (<0.35) 反正会被丢弃
            if aligned is not None and 0.35 <= q <= 0.6 and ph1 is not None:
                dist = phash_dist(ph1, phash(aligned))
                if dist > 30:
                    q = max(q - 0.1, 0.0)
            unused_bonus = 0 if pno not in used_pages else 1
            key = (unused_bonus, -q)
            if chosen is None or key < chosen[0]:
                tmpl_dark_dil_c = cv2.dilate((img1 < 150).astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
                chosen = (key, pno, pmeta, aligned, q, img1, tmpl_dark_dil_c, align_M)
            return bool(aligned is not None and q >= 0.55 and pno not in used_pages)

        chosen = None
        done = set()
        strong = False
        n_refined = 0
        exp_pno = (prev_geo['pno'] + 1) if prev_geo else None

        # ① 顺序先验快通道
        if prev_geo and prev_geo['M'] is not None and prev_geo['scan_shape'] == simg.shape:
            if (exp_pno in tmpl_feats and exp_pno not in used_pages
                    and tmpl_feats[exp_pno][0].shape == prev_geo['tmpl_shape']):
                al_f, q_f = _final_eval(simg, tmpl_feats[exp_pno][0], prev_geo['M'])
                if q_f >= 0.55:
                    done.add(exp_pno)
                    strong = _consider(exp_pno, al_f, q_f, prev_geo['M'])

        # ② ORB 第 1 名完整对齐
        if not strong and cands:
            pno0 = cands[0][1]
            done.add(pno0)
            cq, ang, sc, tdx, tdy = _coarse_grid(tmpl_feats[pno0][0], simg)
            if cq is not None:
                aligned, quality, align_M = _refine_best(simg, tmpl_feats[pno0][0], ang, sc, tdx, tdy)
                n_refined += 1
                strong = _consider(pno0, aligned, quality, align_M)

        # ③ 其余候选: 粗搜排名 + 剪枝 + 早停
        if not strong:
            rest = []
            for good, pno in cands:
                if pno in done:
                    continue
                cq, ang, sc, tdx, tdy = _coarse_grid(tmpl_feats[pno][0], simg)
                if cq is not None:
                    rest.append((cq, ang, sc, tdx, tdy, pno))
            # 未用页优先, 粗分降序 —— 与旧版 chosen key 的优先级一致
            rest.sort(key=lambda t: (0 if t[5] not in used_pages else 1, -t[0]))
            best_cq = rest[0][0] if rest else -1.0
            for cq, ang, sc, tdx, tdy, pno in rest:
                # 剪枝: 粗分已落后最优 0.12 以上, 细化不可能翻盘 (列表降序)
                if chosen is not None and cq < best_cq - 0.12:
                    break
                aligned, quality, align_M = _refine_best(
                    simg, tmpl_feats[pno][0], ang, sc, tdx, tdy, fast=(n_refined >= 2))
                n_refined += 1
                strong = _consider(pno, aligned, quality, align_M)
                if strong:
                    break
                # 早停: 照片件整页 q 达不到 0.55 强通过线, 但当它已明显优于
                # 其它候选(粗分领先)或正好是顺序上的下一页时, 继续细化剩余
                # 候选只是在等价解里挑噪声。
                if (aligned is not None and quality >= 0.42 and pno not in used_pages
                        and (pno == exp_pno or cq >= best_cq - 0.04)):
                    break
                # 已有可用解 (q≥0.35) 且已细化 3 个 → 剩余候选基本无望更好
                if n_refined >= 3 and chosen is not None and chosen[4] >= 0.35:
                    break
        if chosen is None:
            print(f'  警告: 扫描第{si+1}页无可用题本页匹配, 跳过')
            continue

        _, pno, pmeta, aligned, quality, tmpl_img, tmpl_dark_dil, align_M = chosen
        used_pages.add(pno)
        # 结构兜底: 在「选页完成之后」做, 不能放进 align_scan —— 扫描页试配
        # 错误模板页时旋转大是常态, 兜底改写那些组合的 q 会扰乱页级选优
        # (SKM 卷实测: scan p2/p4/p9 的页映射被整体轮换, 37 题错值)。
        # 心理卷 xl21 p1 实测: 扫描与模板表格线偏差仅 ~2pt 且双双水平, 但
        # q8 对「4° 旋转错位解 (0.295)」与「正确解 (0.25)」失去区分度 (手机
        # 照片光照不均), 选页后得到旋转+错位 ~30pt 的对齐, 行带检测/题干
        # 匹配全部失败, 9 题全部「行定位不可靠」。结构证据裁决: 印刷表格
        # 横线对齐正确时笔直 (行投影 3~6px 窄峰), 旋转 ≥1° 后散开消失
        # (xl21 实测 ang=0 → 9 个窄峰, 4° 错误解 → 0 个)。三重条件触发:
        # q<0.5 且当前解含 ≥1.5° 旋转且窄峰 <3; 零旋转附近小网格中以窄峰
        # 最大者替换 —— 平移误差由行带配对 (整体 dy) 与题干 NCC 窗口吸收。
        if aligned is not None and quality < 0.5 and align_M is not None:
            _ang_cur = float(np.degrees(np.arctan2(align_M[1][0], align_M[0][0])))
            if abs(_ang_cur) >= 1.5 and _line_score(aligned) < 3:
                th0, tw0 = tmpl_img.shape
                # 复用本页已渲染的灰度图 (旧版在此重复 render_page, 200dpi 一页 ~50ms)
                _scan_img = simg
                fb_best, fb_al, fb_M = _line_score(aligned), None, None
                for ang_fb in (0.0, -0.5, 0.5):
                    Mfb = cv2.getRotationMatrix2D((tw0 / 2, th0 / 2), ang_fb, 1.0)
                    alf = cv2.warpAffine(_scan_img, Mfb, (tw0, th0), borderValue=255)
                    sf = _line_score(alf)
                    if sf > fb_best:
                        fb_best, fb_al, fb_M = sf, alf, Mfb
                if fb_al is not None and fb_best > _line_score(aligned) + 2:
                    aligned, align_M = fb_al, fb_M
                    print(f'  提示: 扫描第{si+1}页配准存在 {_ang_cur:.1f}° 旋转错位, '
                          f'已按表格线结构证据校正')
        aligned_pages[si] = {'aligned': aligned, 'tmpl_img': tmpl_img,
                             'dark': tmpl_dark_dil, 'pmeta': pmeta, 'pno': pno}
        # 记录本页几何, 供下一页的「顺序先验快通道」复用
        prev_geo = {'M': align_M, 'pno': pno, 'scan_shape': simg.shape,
                    'tmpl_shape': tmpl_img.shape}
        last_adj_y0 = None  # 行序单调性护栏按页重置
        weak_page = (aligned is None or quality < 0.35)
        if aligned is None:
            print(f'  警告: 扫描第{si+1}页 -> 题本第{pno}页 配准失败, 标记为需人工录入')
            for q in pmeta['questions']:
                rows.append([q['id'], q['qtext'], '', '配准失败', 0, ''])
            for fl in pmeta['fills']:
                fill_rows.append([fl['name'], fl['anchor'], '', '配准失败', ''])
            continue
        if weak_page:
            print(f'  提示: 扫描第{si+1}页整页配准偏低({quality:.2f}), 降级为「逐行重定位」逐题识别')

        # 配准质量 → 自适应 padding
        pad_choice = 12 if quality >= 0.7 else 16 if quality >= 0.5 else 20
        if align_M is not None:
            try:
                page_reg[si + 1] = {
                    'tpl_page': pno,
                    'M': [[round(float(v), 6) for v in r] for r in align_M],
                    'size_pt': [round(float(scan_doc[si].rect.width), 2),
                                round(float(scan_doc[si].rect.height), 2)],
                    'scale': float(SCALE),
                }
            except Exception:
                pass

        # ---- 页级全局偏移投票 (两阶段行定位·阶段一) ----
        # 照片件的整页配准质量可能只有 0.3, 单行 NCC 峰值嘈杂 (真假峰差 <0.05)。
        # 先用各题唯一题干补丁做全页搜索, 直方图投票求「本页内容的全局 dy」,
        # 阶段二把每题搜索限制在全局 dy 邻域 —— 噪声峰被几何约束排除。
        page_ys = sorted(min(o['rect'][1] for o in qq['options'])
                         for qq in pmeta['questions'])
        pitch = 0.0
        if len(page_ys) >= 2:
            gaps = [b - a for a, b in zip(page_ys, page_ys[1:]) if b > a]
            if gaps:
                pitch = float(np.median(gaps))
        global_dy = None
        vote_prior = {}
        prior_raw = {}
        if len(pmeta['questions']) >= 3 and pitch:
            votes = {}
            # 全部题都投 (题干补丁 NCC 全页搜索 ~1ms/题): 既求页级 global_dy,
            # 也为行带 DP 配对提供每题的绝对位置先验 —— DP 只看残差平滑度,
            # 对「整页平移若行」盲视 (SKM p10 顶部多一行/底部缺两行时整体错 1 行),
            # 可靠票 (rmx≥0.55) 是打破平移盲区的关键
            for qiv, qq in enumerate(pmeta['questions']):
                rt0v, rt1v, rc0v, rc1v, rpv, _ysv = _stem_patch(tmpl_img, qq, pitch)
                dyv, _dxv, rmxv, okv = _row_relocate(
                    rt0v, rt1v, rc0v, rc1v, aligned, pitch, rpv, cap=-1, full=True)
                if okv:
                    votes[qiv] = (dyv, rmxv)
                # 先验票不要求 ok: 歧义 (次峰接近) 的主峰位置仍是最佳候选,
                # 降权使用即可。但周期性版式上弱票可能是假锁定 (p9 实测出现
                # dy=+178/-178 的野票) —— 先收集, 下面按共识过滤
                if rmxv >= 0.45:
                    prior_raw[qiv] = (dyv, 1.0 if rmxv >= 0.55 else 0.5)
            # 共识过滤: 6pt 聚类, 只保留 ≥3 票的簇。真票同页高度一致
            # (p2 弱票 10 行聚在 ±0.7pt), 假锁定票散乱无簇 —— 孤立野票
            # 进 DP 会把路径搅乱导致配对整体失败
            _clusters = {}
            for _qi, (_dyv, _w) in prior_raw.items():
                _clusters.setdefault(round(_dyv / 6.0), []).append(_qi)
            for _k, _qis in _clusters.items():
                if len(_qis) >= 3:
                    for _qi in _qis:
                        vote_prior[_qi] = prior_raw[_qi]
            vote_list = [v[0] for v in votes.values()]
            if len(vote_list) >= max(2, len(pmeta['questions']) // 4):
                # 6pt 直方图投票取众数, 再取众数 bin ±6pt 内均值
                bins = {}
                for v in vote_list:
                    k = round(v / 6.0)
                    bins.setdefault(k, []).append(v)
                best_k = max(bins, key=lambda k: len(bins[k]))
                if len(bins[best_k]) >= 2:
                    global_dy = float(np.mean(bins[best_k]))

        # ---- 表格线行带配对 (行定位·首选) ----
        # 选项列 x 范围内的水平表格线直接给出每行的真实位置, 对照片件的
        # 残余剪切/缩放/模糊免疫; 失败再退回题干补丁 NCC 方案
        band_dys = None
        if pitch and len(pmeta['questions']) >= 3:
            xs0p = min(min(o['rect'][0] for o in qq['options'])
                       for qq in pmeta['questions'])
            xs1p = max(max(o['rect'][2] for o in qq['options'])
                       for qq in pmeta['questions'])
            try:
                bands = _detect_row_bands(aligned, xs0p, xs1p, pitch)
                # 行中心锚点 = 选项格矩形中心: 实测模板格框中心与题本真实
                # 行中心逐行精确重合。旧锚点 ys0+9 隐含「格高 18pt」, 遇到
                # 厚格框题本 (807 卷 p2/p5 格高 40pt) 锚点整体偏低 ~10pt,
                # 行带 dy 把本已居中的格子推离行中心, 格内圈进上下表格线
                # 残段 —— 墨迹判定被格线噪声淹没 (SKM p2 m007/m010 错值根因)。
                # 对常规 18pt 格框, 格中心 = ys0+9, 行为不变 (其他卷无影响)。
                tpl_centers = [(min(o['rect'][1] for o in qq['options']) +
                                max(o['rect'][3] for o in qq['options'])) / 2.0
                               for qq in pmeta['questions']]
                if bands:
                    band_dys = _pair_row_dys(tpl_centers, bands, pitch,
                                             prior=vote_prior)
            except Exception:
                band_dys = None

        for qi, q in enumerate(pmeta['questions']):
            options_info = [(o['value'], o['text'], o['rect']) for o in q['options']]

            # 行级重定位 (两阶段·阶段二): 用每行唯一题干补丁在 aligned 图上
            # 重新找本行的垂直位置。有页级 global_dy 时限制搜索邻域。
            row_dy = 0.0
            row_dx = 0.0
            row_mx = 0.0
            row_ok = False
            rt0, rt1, rc0, rc1, rpatch, ys0 = _stem_patch(tmpl_img, q, pitch)
            xs0 = min(o['rect'][0] for o in q['options'])
            if band_dys is not None and qi in band_dys:
                # 行带配对成功: 直接用本行表格线给出的 dy (对剪切免疫)。
                # 不设 |dy| 上限 —— 配对内部已用残差一致性校验, 大 dy 只是
                # 说明 M 仿射误差大, 不是错位 (SKM 卷 p5/p10 实测 dy≈-80 为真)
                row_dy = float(band_dys[qi])
                row_ok = True
                row_mx = 1.0
            elif band_dys is None:
                # 页面级行带检测/配对失败 → 题干补丁 NCC 锚定 (宽搜索窗)
                row_dy, row_dx, row_mx, row_ok = _row_relocate(
                    rt0, rt1, rc0, rc1, aligned, pitch, rpatch)
            # 行带成功但本行未配到 (漏检/溢出邻页): 保持 row_ok=False,
            # 由旗标与跨页重试兜底, 不冒险用噪声 NCC 峰
            # 行序单调性护栏: 重定位后本行 y0 不得反超上一行 (同页内行序固定,
            # 违反说明吸错行了) → 回退为不可靠
            adj_y0 = ys0 + (row_dy if row_ok else 0.0)
            if row_ok and last_adj_y0 is not None and \
                    adj_y0 < last_adj_y0 - 0.25 * pitch:
                row_ok = False
                row_dy = row_dx = 0.0
                adj_y0 = ys0
            last_adj_y0 = adj_y0
            if row_ok and (row_dy or row_dx):
                options_info = [(v, tx, [r[0] + row_dx, r[1] + row_dy,
                                         r[2] + row_dx, r[3] + row_dy])
                                for v, tx, r in options_info]

            # 局部精配准 (模板匹配微调)。NCC 锚定的行用大垂直窗 —— 印刷数字
            # 行距 >> 题干/选项列的剪切残差, 「最近数字行」必是正确行, 吸附
            # 不会串行; 行带 dy 已准, 小窗即可
            if band_dys is not None and qi in band_dys:
                vy = 14
            else:
                vy = int(0.55 * pitch * SCALE) if pitch else 14
            adjusted_opts = _refine_cells(options_info, tmpl_img, aligned, vy=vy)
            cell_reg[q['id']] = [[round(float(v), 2) for v in r] for _, _, r in adjusted_opts]

            # 双组刻度拆分: 同一行内数值标签序列重启(如 1..8, 1..8)是
            # 「上学日一天/周末一天」式双组刻度表 —— 一行其实是两道题。
            # 两组各自独立判定, 否则 16 格里两处勾会被误判成「多选可疑」,
            # 而且第二组的勾会把第一组的答案挤成 9~16 的越界值。
            def _num_val(t):
                # 只认「纯数字」标签为刻度值 —— 题干「2观看短视频…」开头的
                # 题号 2 不是刻度, 混进来会破坏序列重启检测
                t = str(t).strip()
                return int(t) if re.fullmatch(r'\d+', t) else None

            def _consecutive(seq):
                vals = [v for v in seq if v is not None]
                return bool(vals) and vals == list(range(vals[0], vals[0] + len(vals)))

            groups = [adjusted_opts]
            if len(adjusted_opts) >= 4:
                nums = [_num_val(t) for _, t, _ in adjusted_opts]
                first_num = next((v for v in nums if v is not None), None)
                if first_num is not None:
                    for k in range(2, len(adjusted_opts) - 1):
                        if nums[k] == first_num and _consecutive(nums[:k]) \
                                and _consecutive(nums[k:]):
                            groups = [adjusted_opts[:k], adjusted_opts[k:]]
                            break

            def _decide(g):
                return decide_choice(q, tmpl_dark_dil, aligned, g,
                                     min_score=min_score, margin=margin)

            chosen_val, status, score, det_feat = _decide(groups[0])
            # 行位置不可靠时如实标注而非假装 OK: 整页配准差 / 页面级行带配对
            # 成功但本行未配到 (行带漏检或溢出到相邻页) 都算
            if (weak_page or band_dys is not None) and not row_ok:
                status = '行定位不可靠,请核对'

            xs0 = min(o['rect'][0] for o in q['options'])
            ys0 = min(o['rect'][1] for o in q['options'])
            xs1 = max(o['rect'][2] for o in q['options'])
            ys1 = max(o['rect'][3] for o in q['options'])
            x0i, y0i, x1i, y1i = [int(v * SCALE) for v in (xs0, ys0, xs1, ys1)]
            crop = aligned[max(0, y0i - 8):y1i + 8, max(0, x0i - 8):x1i + 8]
            crop_path = ''
            if crop.size:
                crop_path = f'crops_p{pno}_q{qi+1}.png'
                cv2.imwrite(os.path.join(crops_dir, crop_path), crop)
            rows.append([q['id'], q['qtext'], chosen_val, status, score, crop_path])
            # 首轮没定位可靠的题: 记入跨页重试候选。
            # 注意只收「定位失败」的题 —— 定位成功但读不到墨的题是真空白,
            # 拿去别的页重试会抓到平行form(父亲/母亲量表题干相同)的勾。
            if status == '行定位不可靠,请核对' or \
                    (chosen_val in ('', 'null') and not row_ok):
                retry_cands.append({
                    'si': si, 'q': q, 'idx': len(rows) - 1,
                    'rt0': rt0, 'rt1': rt1, 'rc0': rc0, 'rc1': rc1,
                    'rpatch': rpatch.copy(), 'pitch': pitch,
                    'options_info': [(o['value'], o['text'], o['rect'])
                                     for o in q['options']],
                    # 只有页首/页尾边界行才允许跨页重试 (照片裁切只会
                    # 溢出边界行); 中间行缺行带说明检测问题, 不冒险
                    'boundary': qi <= 1 or qi >= len(pmeta['questions']) - 2,
                })
            # 第2组及以后的刻度: 作为附加行写进 CSV(网页题单没有对应题号,
            # 不参与映射、不产生核验项), 值重新从 1 编号。
            for gi, g in enumerate(groups[1:], 2):
                gv = [(k + 1, tx, r) for k, (v, tx, r) in enumerate(g)]
                gval, gst, gsc, _ = _decide(gv)
                rows.append([q['id'] + '#g%d' % gi,
                             (q['qtext'] or '') + '（第%d组刻度）' % gi,
                             gval, gst + '(第%d组)' % gi, gsc, crop_path])

        for fi, fl in enumerate(pmeta['fills']):
            x0, y0, x1, y1 = [int(v * SCALE) for v in fl['rect']]
            ya = max(0, y0 - pad_choice)
            # 手写常超出下划线矩形（如 10 位学号写得又长又大）。裁格向右扩展：
            # 上限 160pt；遇到同一行下一个填空的左缘即停（防止吞并隔壁空的笔迹）。
            # 扩展区里的印刷字由模板墨掩码扣除，不会混进笔迹。
            x_max = min(x1 + int(100 * SCALE), aligned.shape[1] - 2)
            for fl2 in pmeta['fills']:
                if fl2 is fl:
                    continue
                xx0, yy0, _, yy1 = [int(v * SCALE) for v in fl2['rect']]
                if yy0 < y1 + pad_choice and yy1 > ya and xx0 >= x1:
                    x_max = min(x_max, xx0 - 3)
            x1e = max(x1, x_max)
            xa = max(0, x0 - pad_choice)
            crop = aligned[ya:y1 + pad_choice, xa:x1e + pad_choice]
            crop_path = ''
            value, status = '', ''
            if crop.size:
                crop_path = f'fill_p{pno}_{fi+1}.png'
                cv2.imwrite(os.path.join(crops_dir, crop_path), crop)
                # 差分: 扫描有墨而题本无墨
                fm = (crop < 150) & ~(tmpl_dark_dil[ya:y1 + pad_choice, xa:x1e + pad_choice] > 0)
                # 也试试自适应
                fm_adapt = adaptive_binarize(crop, block_size=25, C=10)
                if fm_adapt is not None:
                    # 自适应结果同样必须减掉印刷底稿 —— 否则下划线/印刷字边缘
                    # 会混进笔迹掩码, 数字分割把一道「2」切成「827」这类串
                    fm_adapt = fm_adapt & ~(tmpl_dark_dil[ya:y1 + pad_choice, xa:x1e + pad_choice] > 0)
                    fm = fm.astype(np.uint8) | (fm_adapt > 0).astype(np.uint8)
                else:
                    fm = fm.astype(np.uint8) * 255
                # 笔迹溢出下划线矩形 → 手写不受格约束, 数字识别不可信, 强制人工核对
                ext = fm[:, (x1 + pad_choice - xa):] if (x1 + pad_choice - xa) < fm.shape[1] else None
                overflow = ext is not None and int((ext > 0).sum()) > 60
                value, status = fill_value_status(fl['anchor'], fm * 255, overflow)
            fill_rows.append([fl['name'], fl['anchor'], crop_path, status, value])

    # 跨页行重试: 照片裁切会把页面边界行溢出到相邻扫描页 —— 只对「页首/页尾
    # 边界行」重试, 且相邻页上不得存在近似题干 (父亲/母亲平行form同号题干只差
    # 一个字, NCC 分辨不了, 文本相似度可以), 接受阈值也更高。
    n_pages = len(scan_doc)
    for cand in retry_cands:
        si = cand['si']
        q = cand['q']
        if not cand.get('boundary'):
            continue
        # 只试紧邻页: 照片裁切造成的行溢出只会落到相邻页
        for adj in (si + 1, si - 1):
            if adj < 0 or adj >= n_pages:
                continue
            ap = aligned_pages.get(adj)
            if ap is None or ap.get('aligned') is None:
                continue
            # 平行form防护: 目标页上有近似题干 → NCC 无法区分, 放弃该页
            if any(qq is not q and _stem_sim(q['qtext'], qq['qtext']) > 0.7
                   for qq in ap['pmeta']['questions']):
                continue
            dy, dx, rmx, ok = _row_relocate(cand['rt0'], cand['rt1'],
                                            cand['rc0'], cand['rc1'],
                                            ap['aligned'], cand['pitch'],
                                            cand['rpatch'], cap=-1, full=True)
            if not ok or rmx < 0.60:
                continue
            opts = [(v, t, [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy])
                    for v, t, r in cand['options_info']]
            adjusted = _refine_cells(opts, ap['tmpl_img'], ap['aligned'])
            val, st, sc, _ = decide_choice(q, ap['dark'], ap['aligned'], adjusted,
                                           min_score=min_score, margin=margin)
            if not val or st == '未检出':
                continue  # 这一页上确实没勾, 保留首轮结果
            xs0 = min(r[0] for _, _, r in adjusted)
            ys0 = min(r[1] for _, _, r in adjusted)
            xs1 = max(r[2] for _, _, r in adjusted)
            ys1 = max(r[3] for _, _, r in adjusted)
            x0i, y0i, x1i, y1i = [int(v * SCALE) for v in (xs0, ys0, xs1, ys1)]
            crop = ap['aligned'][max(0, y0i - 8):y1i + 8,
                                 max(0, x0i - 8):x1i + 8]
            crop_path = ''
            if crop.size:
                crop_path = f'crops_xp_p{ap["pno"]}_q{q["id"]}.png'
                cv2.imwrite(os.path.join(crops_dir, crop_path), crop)
            rows[cand['idx']] = [q['id'], q['qtext'], val, st, sc, crop_path]
            cell_reg[q['id']] = [[round(float(v), 2) for v in r]
                                 for _, _, r in adjusted]
            print(f'  跨页重试: {q["id"]} 在扫描第{adj+1}页找到行位 (rmx={rmx:.2f})')
            break

    csv_path = os.path.join(out_dir, f'{name}_识别结果.csv')
    csv2_path = os.path.join(out_dir, f'{name}_填空题.csv')
    # 目标文件被 Excel/WPS 占用时改写到 <名>.bak: 上层按 name_识别结果*.csv
    # 兜底查找, 整次识别不会因为一个 CSV 被占用就失败。
    write_csv_safe(csv_path, ['题目ID', '题干', '识别答案', '状态', '墨迹分数', '校对图'], rows)
    write_csv_safe(csv2_path, ['字段', '锚点文本', '截图', '状态', '识别/录入值'], fill_rows)

    # 页面配准信息（供网页端把识别框贴到扫描件实际格子上）
    try:
        reg_path = os.path.join(out_dir, f'{name}_页面配准.json')
        with open(reg_path, 'w', encoding='utf-8') as f:
            json.dump({'pages': {str(k): v for k, v in page_reg.items()},
                       'cells': cell_reg}, f, ensure_ascii=False)
    except Exception as e:
        print(f'  提示: 页面配准信息写入失败({e})，识别框将按题本坐标显示')

    html_path = os.path.join(out_dir, f'{name}_校对.html')
    write_review_html(html_path, name, rows, fill_rows, res_dir)
    print(f'完成: {csv_path}')
    return csv_path


# ---------------------------------------------------------------- 汇总 + HTML 校对 (保留原版)

def summarize(out_dir):
    import glob
    ref = os.path.join(out_dir, '题目对照表.csv')
    questions = []
    fills_meta = []
    if os.path.exists(ref):
        with open(ref, encoding='utf-8-sig') as f:
            rd = csv.reader(f)
            next(rd, None)
            for r in rd:
                if not r:
                    continue
                if r[2].startswith('[填空]'):
                    fills_meta.append([r[0], r[1], r[2][5:]])
                elif r[0] not in {q[0] for q in questions}:
                    questions.append([r[0], r[1], r[2]])
    scans = []
    for p in sorted(glob.glob(os.path.join(out_dir, '*_识别结果.csv'))):
        name = os.path.basename(p)[:-len('_识别结果.csv')]
        res = {}
        with open(p, encoding='utf-8-sig') as f:
            rd = csv.reader(f)
            next(rd, None)
            for r in rd:
                if r:
                    res[r[0]] = [r[2], r[3], r[5]]
        fvals = {}
        fp = os.path.join(out_dir, f'{name}_填空题.csv')
        if os.path.exists(fp):
            with open(fp, encoding='utf-8-sig') as f:
                rd = csv.reader(f)
                next(rd, None)
                for r in rd:
                    if r:
                        fvals[r[0]] = [r[4] or '', r[3], r[2]]
        n_ok = sum(1 for q in questions if q[0] in res and res[q[0]][1] == 'OK')
        scans.append({'name': name, 'res': res, 'fills': fvals, 'ok': n_ok,
                      'warn': sum(1 for q in questions if q[0] in res and res[q[0]][1] != 'OK')})
    if not scans:
        print('没有找到识别结果, 跳过汇总')
        return
    csv_path = os.path.join(out_dir, '汇总_所有问卷答案.csv')
    warn_path = os.path.join(out_dir, '汇总_可疑题清单.csv')
    html_path = os.path.join(out_dir, '汇总_所有问卷.html')
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['问卷文件'] + [q[0] for q in questions] + [fm[0] for fm in fills_meta])
            w.writerow(['(题干)'] + [q[2] for q in questions] + [f'{fm[2]}[填空]' for fm in fills_meta])
            for s in scans:
                row = [s['name']]
                row += [s['res'].get(q[0], ['', ''])[0] for q in questions]
                row += [s['fills'].get(fm[0], [''])[0] for fm in fills_meta]
                w.writerow(row)
        with open(warn_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['问卷文件', '题目ID', '题干', '识别答案', '状态'])
            for s in scans:
                for q in questions:
                    r = s['res'].get(q[0])
                    if r and r[1] != 'OK':
                        w.writerow([s['name'], q[0], q[2], r[0], r[1]])
        write_summary_html(html_path, scans, questions, fills_meta)
        print(f'汇总: {csv_path}')
    except PermissionError:
        print(f'  提示: 文件被占用, 已跳过写入')


def write_review_html(html_path, name, rows, fill_rows, res_dir):
    opts_js = json.dumps(rows, ensure_ascii=False, default=str)
    fills_js = json.dumps(fill_rows, ensure_ascii=False, default=str)
    n_ok = sum(1 for r in rows if r[3] == 'OK')
    n_warn = sum(1 for r in rows if r[3] != 'OK' and '配准' not in str(r[3]))
    n_bad = sum(1 for r in rows if '配准' in str(r[3]))
    html = '''<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__NAME__ · 识别校对</title>
<style>
:root{--ok:#16a34a;--warn:#d97706;--bad:#dc2626;--line:#e5e7eb;--bg:#f4f6f9}
*{box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;margin:0;background:var(--bg);color:#1f2937;padding-bottom:90px}
header{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;padding:22px 28px 18px}
header h1{margin:0;font-size:20px;font-weight:600}
.chips{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
.chip{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:5px 16px;font-size:13px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card>h3{margin:0;padding:13px 18px;font-size:14px;background:#f9fafb;border-bottom:1px solid var(--line)}
.card .body{padding:16px 18px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{background:#f9fafb;font-size:12px;color:#6b7280}
tr.warn td{background:#fffbeb}
tr.bad td{background:#fef2f2}
img.crop{max-height:46px;border-radius:5px;border:1px solid var(--line);cursor:zoom-in}
.pill{display:inline-block;min-width:32px;text-align:center;padding:3px 12px;border-radius:999px;font-weight:700;background:#ecfdf5;color:var(--ok);font-size:14px}
select,input.fillsel{padding:4px 8px;border:1px solid var(--line);border-radius:6px;font-size:13px;background:#fff}
.btn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 16px;font-size:13px;cursor:pointer}
.btn:hover{border-color:#2563eb;color:#2563eb}
.btn.pri{background:#2563eb;border-color:#2563eb;color:#fff}
#fab{position:fixed;right:28px;bottom:28px;z-index:30;padding:13px 26px;border-radius:999px;border:0;background:#2563eb;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
#lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:99;align-items:center;justify-content:center;cursor:zoom-out}
#lightbox img{max-width:92vw;max-height:90vh;border-radius:8px}
</style></head><body>
<header>
  <h1>📝 __NAME__ — 识别校对</h1>
  <p>黄色行 = 可疑答案（建议对照截图确认），红色行 = 配准失败。</p>
  <div class="chips">
    <span class="chip"><b>__NALL__</b>总题数</span>
    <span class="chip">✅ 自动确认 <b>__NOK__</b></span>
    <span class="chip">⚠️ 待确认 <b>__NWARN__</b></span>
    <span class="chip">❌ 配准失败 <b>__NBAD__</b></span>
  </div>
</header>
<div style="padding:18px 28px">
  <table id="tbl"><thead><tr><th>#</th><th>题干</th><th>截图</th><th>识别答案</th><th>改为</th><th>状态</th><th>分数</th></tr></thead>
<tbody></tbody></table>
  <h2 style="font-size:15px;margin:30px 0 10px">✍️ 填空题（手写内容，请对照截图录入）</h2>
  <table id="ftbl"><thead><tr><th>字段</th><th>截图</th><th>状态</th><th>录入值</th></tr></thead><tbody></tbody></table>
</div>
<button id="fab" onclick="exportCsv()">⬇️ 导出修正后 CSV</button>
<div id="lightbox" onclick="this.style.display='none'"><img id="lbimg"></div>
<script>
const rows = __ROWS__;
const fills = __FILLS__;
const tb = document.querySelector('#tbl tbody');
rows.forEach((r,i)=>{
  const tr=document.createElement('tr');
  const cls = r[3]==='OK'?'ok':(r[3].includes('配准')?'bad':'warn');
  if(cls!=='ok') tr.className=cls;
  const img = r[5]?`<img class="crop" src="__NAME__/crops/${r[5]}" onclick="showImg(this.src)">`:'';
  tr.innerHTML=`<td>${i+1}</td>
   <td>${r[1]||''}<br><small style="color:#6b7280;font-size:11px">${r[0]}</small></td>
   <td>${img}</td>
   <td><span class="pill">${r[2]||'—'}</span></td>
   <td><select data-i="${i}" onchange="onEdit(${i},this.value)"><option value="">(保持)</option>${[1,2,3,4,5,6,7,8].map(v=>`<option ${r[2]==String(v)?'selected':''}>${v}</option>`).join('')}</select></td>
   <td>${r[3]}</td><td>${r[4]}</td>`;
  tb.appendChild(tr);
});
function onEdit(i,v){ if(v!=='') rows[i][6]=true; }
function showImg(src){ document.getElementById('lbimg').src=src; document.getElementById('lightbox').style.display='flex'; }
const fb=document.querySelector('#ftbl tbody');
fills.forEach((r,i)=>{
  const tr=document.createElement('tr');
  const img = r[2]?`<img class="crop" src="__NAME__/crops/${r[2]}" style="max-height:38px" onclick="showImg(this.src)">`:'';
  tr.innerHTML=`<td>${r[0]}<br><small style="color:#6b7280;font-size:11px">${r[1]}</small></td><td>${img}</td><td>${r[3]}</td>
   <td><input class="fillsel" data-i="${i}" value="${(r[4]||'').replace(/"/g,'&quot;')}" placeholder="空白则在此输入"></td>`;
  fb.appendChild(tr);
});
function exportCsv(){
  let out='\\ufeff题目ID,题干,最终答案\\n';
  rows.forEach((r,i)=>{
    const sel=document.querySelector(`select[data-i="${i}"]`);
    const v=sel&&sel.value!==''?sel.value:r[2];
    out+=`"${r[0]}","${(r[1]||'').replace(/"/g,'""')}","${v||''}"\\n`;
  });
  out+='\\n填空字段,人工录入值\\n';
  fills.forEach((r,i)=>{
    const inp=document.querySelector(`input.fillsel[data-i="${i}"]`);
    out+=`"${r[0]}","${inp?inp.value:''}"\\n`;
  });
  const blob=new Blob([out],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='__NAME___最终.csv'; a.click();
}
</script></body></html>'''
    html = (html.replace('__NAME__', name).replace('__ROWS__', opts_js).replace('__FILLS__', fills_js)
                .replace('__NALL__', str(len(rows))).replace('__NOK__', str(n_ok))
                .replace('__NWARN__', str(n_warn)).replace('__NBAD__', str(n_bad)))
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def write_summary_html(html_path, scans, questions, fills_meta):
    n_fill = len(fills_meta)
    cols = [[q[0], q[2] or '', 'choice'] for q in questions]
    cols += [[fm[0], fm[2], 'fill'] for fm in fills_meta]
    data = {}
    for s in scans:
        row = []
        for c in cols:
            if c[2] == 'choice':
                r = s['res'].get(c[0], ['', '', ''])
                row.append([r[0], r[1], r[2]])
            else:
                r = s['fills'].get(c[0], ['', '', ''])
                row.append([r[0], r[1], r[2]])
        data[s['name']] = row
    payload = {
        'scans': [{'name': s['name'], 'ok': s['ok'], 'warn': s['warn'],
                   'review': s['name'] + '_校对.html'} for s in scans],
        'cols': cols, 'data': data,
    }
    html = '''<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>问卷识别结果汇总</title>
<style>
body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;background:#f4f6f9;color:#1f2937}
header{background:linear-gradient(135deg,#065f46,#059669);color:#fff;padding:22px 28px 16px}
header h1{margin:0;font-size:20px;font-weight:600}
.chips{display:flex;gap:10px;margin-top:12px;flex-wrap:wrap}
.chip{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:5px 16px;font-size:13px}
table{border-collapse:separate;border-spacing:0;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
th,td{padding:7px 9px;font-size:13px;border-bottom:1px solid #e5e7eb;text-align:center;white-space:nowrap}
th{background:#1f2937;color:#fff;font-size:12px}
td.ans{font-weight:600;color:#065f46}
td.ans.warn{background:#fffbeb;color:#d97706}
td.ans.bad{background:#fef2f2;color:#dc2626}
.stucol,th.stucol{position:sticky;left:0;background:#fff;z-index:5;text-align:left;min-width:180px}
th.stucol{background:#1f2937;z-index:7}
</style></head><body>
<header>
  <h1>📊 问卷识别结果汇总</h1>
  <div class="chips" id="chips"></div>
</header>
<div style="padding:20px 28px 60px;max-width:1400px;margin:0 auto;overflow:auto">
<table id="tbl"></table>
</div>
<script>
const D = __PAYLOAD__;
let h='<thead><tr><th class="stucol">问卷文件/题号</th>';
D.cols.forEach(c=>{h+=`<th>${c[0]}</th>`});
h+='</tr><tr><th class="stucol">题干</th>';
D.cols.forEach(c=>{h+=`<th style="font-weight:400;font-size:10px">${(c[1]||'').slice(0,20)}</th>`});
h+='</tr></thead><tbody>';
D.scans.forEach(s=>{
  h+=`<tr><td class="stucol"><a href="${s.review}" target="_blank" style="color:#2563eb;text-decoration:none">${s.name}</a><br><span style="font-size:11px;color:#6b7280">✅${s.ok} ⚠️${s.warn}</span></td>`;
  D.data[s.name].forEach(cell=>{
    const [v,st]=cell;
    const cls = st==='OK'||st.includes('自动识别')?'ans':(st.includes('配准')?'ans bad':(st&&!st.includes('OK')?'ans warn':'ans'));
    h+=`<td class="${cls}">${v||'—'}</td>`;
  });
  h+='</tr>';
});
h+='</tbody>';
document.getElementById('tbl').innerHTML=h;
document.getElementById('chips').innerHTML=
`<span class="chip"><b>${D.scans.length}</b>份问卷</span>`+
`<span class="chip"><b>${D.cols.length}</b>列</span>`+
`<span class="chip">✅ 自动识别 ${D.scans.reduce((a,s)=>a+s.ok,0)}</span>`+
`<span class="chip">⚠️ 待确认 ${D.scans.reduce((a,s)=>a+s.warn,0)}</span>`;
</script></body></html>'''
    html = html.replace('__PAYLOAD__', json.dumps(payload, ensure_ascii=False))
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------- 主入口

def main():
    ap = argparse.ArgumentParser(description='纸质问卷自动识别工具 v2.0')
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build-template', help='从题本PDF生成模板')
    b.add_argument('--book', required=True)
    b.add_argument('--outdir', default='outputs')

    r = sub.add_parser('recognize', help='识别扫描件')
    r.add_argument('--scan', required=False)
    r.add_argument('--scan-dir', required=False)
    r.add_argument('--book', required=True)
    r.add_argument('--outdir', default='outputs')
    r.add_argument('--workers', type=int, default=1)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.cmd == 'build-template':
        build_template(args.book, os.path.join(args.outdir, 'template.json'))
    else:
        tmpl_path = os.path.join(args.outdir, 'template.json')
        if not os.path.exists(tmpl_path):
            print('未找到模板, 先自动构建...')
            build_template(args.book, tmpl_path)
        with open(tmpl_path, encoding='utf-8') as f:
            tmpl = json.load(f)
        scans = []
        if args.scan:
            scans.append(args.scan)
        if args.scan_dir:
            scans += [os.path.join(args.scan_dir, x) for x in os.listdir(args.scan_dir)
                      if x.lower().endswith('.pdf') and '题本' not in x]
        for s in scans:
            print(f'识别: {s}')
            recognize_scan(s, tmpl, args.book, args.outdir)
        if scans:
            summarize(args.outdir)


if __name__ == '__main__':
    main()
