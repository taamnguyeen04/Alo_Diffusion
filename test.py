import requests
from bs4 import BeautifulSoup
import time
import re

BASE_URL = "https://600tuvungtoeic.com/index.php?mod=lesson&id={}"
OUTPUT_FILE = "600_toeic_vocab.txt"

def parse_lesson(html):
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    for tuvung in soup.select("div.tuvung"):
        noidung = tuvung.select_one("div.noidung")
        if not noidung:
            continue

        # 1. English word
        span_word = noidung.find("span", style=lambda s: s and "color: blue" in s)
        word = span_word.get_text(strip=True) if span_word else None

        # 2. Meaning: phần ngay sau "Từ loại:"
        meaning = ""
        span_type = noidung.find("span", class_="bold", string=re.compile(r"Từ loại"))
        if span_type and span_type.next_sibling:
            # next_sibling có thể là text hoặc tag, join hết text sau đó đến <br>
            raw = ""
            sib = span_type.next_sibling
            while sib and (not getattr(sib, "name", None) == "br"):
                raw += sib.get_text(strip=True) if hasattr(sib, "get_text") else str(sib).strip()
                sib = sib.next_sibling
            # Loại bỏ phần chỉ định loại từ như "(v):"
            meaning = re.sub(r"^\([^)]+\):\s*", "", raw)

        # 3. Example sentence (tiếng Anh): text ngay sau "Ví dụ:"
        example = ""
        span_vidu = noidung.find("span", class_="bold", string=re.compile(r"Ví dụ"))
        if span_vidu:
            # sibling tiếp theo chứa câu tiếng Anh
            ex_text = span_vidu.next_sibling
            if ex_text:
                example = ex_text.strip()

        if word and meaning and example:
            entries.append((word, meaning, example))

    return entries

def main():
    all_entries = []
    for lesson_id in range(1, 51):
        url = BASE_URL.format(lesson_id)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            entries = parse_lesson(resp.text)
            print(f"Lesson {lesson_id}: found {len(entries)} entries")
            all_entries.extend(entries)
        except Exception as e:
            print(f"Error at lesson {lesson_id}: {e}")
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for word, meaning, ex in all_entries:
            f.write(f"{word}\t{meaning} - Ví dụ: {ex}\n")

    print(f"✔ Đã ghi {len(all_entries)} mục vào `{OUTPUT_FILE}`")

if __name__ == "__main__":
    main()
