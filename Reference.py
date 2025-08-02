import requests
from lxml import etree
import html
import re
import os

GROBID_URL = "http://localhost:8070/api/processCitation"
INPUT_FILE = "References_Grobid_Input.txt"
OUTPUT_FILE = "Tagged_Output.txt"
XML_OUTPUT_FILE = "Grobid_XML_Output.xml"

ns = {'tei': 'http://www.tei-c.org/ns/1.0'}


def normalize(text):
    return re.sub(r'[^\w\s]', '', text).lower()


def normalize_dashes(text):
    return re.sub(r'[–—−]', '-', text)


def decode_entities(ref):
    """Converts HTML/XML entities like &#x2013; to their actual Unicode characters."""
    return html.unescape(ref)


def restore_entities(tagged_output, original_ref):
    # Find all entities in the original reference
    entity_map = {
        html.unescape(m.group(0)): m.group(0)
        for m in re.finditer(r"&#x[0-9A-Fa-f]+;", original_ref)
    }

    # Replace decoded characters back with entities
    for char, entity in entity_map.items():
        tagged_output = tagged_output.replace(char, entity)
    return tagged_output


def wrap_authors(tagged_output):
    pattern = re.compile(r"(?:<gnm>.*?<\/gnm>\s*<snm>.*?<\/snm>|<snm>.*?<\/snm>\s*<gnm>.*?<\/gnm>)")

    def replacer(match):
        return f"{match.group(0)}"

    return pattern.sub(replacer, tagged_output)

import re

def wrap_unstructured_content(xml):
    # First, mark all tag spans so we avoid editing them
    tag_spans = [(m.start(), m.end()) for m in re.finditer(r"<[^>]+>", xml)]

    def is_in_tag(index):
        return any(start <= index < end for start, end in tag_spans)

    # Identify all runs of untagged content
    output = ""
    i = 0
    while i < len(xml):
        if xml[i] == '<':
            # Copy tag content directly
            tag_match = re.match(r"<[^>]+>", xml[i:])
            if tag_match:
                tag = tag_match.group(0)
                output += tag
                i += len(tag)
            else:
                output += xml[i]
                i += 1
        elif xml[i].isspace():
            # Copy whitespace as-is
            output += xml[i]
            i += 1
        else:
            # Start of unstructured text
            j = i
            while j < len(xml) and xml[j] != '<' and not xml[j].isspace():
                j += 1
            unstruct_text = xml[i:j]
            output += f"<unstruct>{unstruct_text}</unstruct>"
            i = j
    return output



def tag_reference(type, original_ref, tagged_parts):
    tagged_ref = original_ref
    tagged_parts = sorted(tagged_parts, key=lambda x: -len(x[0]))  # Longest first
    # print("\n\nTagged parts sorted by length:", tagged_parts)

    def find_tag_ranges(text):
        """Find all ranges (start, end) for existing tags to avoid inner replacements."""
        return [(m.start(), m.end()) for m in re.finditer(r"<[^<>]+>.*?</[^<>]+>", text)]

    def is_inside_tag(pos, tag_ranges):
        """Check if a position falls within any existing tag."""
        for start, end in tag_ranges:
            if start <= pos < end:
                return True
        return False

    for raw, tag in tagged_parts:
        escaped_raw = re.escape(raw)
        pattern = re.compile(rf"(?<![>\w]){escaped_raw}(?![\w<])",flags=re.IGNORECASE)
        tag_ranges = find_tag_ranges(tagged_ref)

        # Find all matches, and only replace the first one not inside a tag
        for match in pattern.finditer(tagged_ref):
            start = match.start()
            end = match.end()
            if not is_inside_tag(start, tag_ranges):
                tagged_ref = tagged_ref[:start] + tag + tagged_ref[end:]
                # print("Taagged ref after replacement:", tagged_ref)
                break  # Only replace once per tag

        # Update tag ranges after each replacement

        tag_ranges = find_tag_ranges(tagged_ref)

    return tagged_ref.strip()


def wrap_with_tei_namespace(xml_fragment):
    return f'''<TEI xmlns="http://www.tei-c.org/ns/1.0">
<text>
    <body>
        {xml_fragment}
    </body>
</text>
</TEI>'''


import re


def remove_style_tags(ref):
    """Removes  and <b> tags and stores content with position to reinsert later."""
    tag_pattern = re.compile(r'<(i|b)>(.*?)</\1>', re.IGNORECASE)
    tags_info = []

    def replacer(match):
        tag = match.group(1).lower()
        content = match.group(2)
        start = match.start()
        tags_info.append({'tag': tag, 'text': content, 'pos': start})
        return content  # Replace with plain content for GROBID

    cleaned_ref = tag_pattern.sub(replacer, ref)
    return cleaned_ref, tags_info


def restore_style_tags(tagged_output, tags_info):
    """Inserts  or <b> tags back into tagged reference output."""
    offset = 0
    for info in sorted(tags_info, key=lambda x: x['pos']):
        tag_open = f"<{info['tag']}>"
        tag_close = f"</{info['tag']}>"
        insertion = f"{tag_open}{info['text']}{tag_close}"
        plain_text = info['text']
        idx = tagged_output.find(plain_text, info['pos'] + offset)
        if idx != -1:
            tagged_output = (
                    tagged_output[:idx] +
                    insertion +
                    tagged_output[idx + len(plain_text):]
            )
            offset += len(tag_open) + len(tag_close)
    return tagged_output


from lxml import etree
from io import StringIO

import re

def extract_author_pairs(xml_root):
    person_tags = ["author", "editor"]
    pairs = []

    for tag in person_tags:
        for pers in xml_root.xpath(f".//tei:{tag}", namespaces=ns):
            gnm_els = pers.xpath(".//tei:forename", namespaces=ns)
            snm_el = pers.find(".//tei:surname", namespaces=ns)

            gnm = " ".join(g.text.strip() for g in gnm_els if g.text) if gnm_els else None
            snm = snm_el.text.strip() if snm_el is not None and snm_el.text else None

            if gnm or snm:
                entry = {}
                if gnm:
                    entry["gnm"] = gnm
                if snm:
                    entry["snm"] = snm
                pairs.append(entry)
    print("pairs : ",pairs)
    # print("Person name pairs extracted:", pairs)
    return pairs

import re

import re
import string

def normalize_text(text):
    return re.sub(rf"[{re.escape(string.punctuation)}\s]", "", text).upper()

import re

def wrap_authors_loose_match(structured, author_pairs):
    def normalize_text(text):
        return re.sub(r"[^\w]", "", text).lower()

    def is_within_au_tag(start, end, au_spans):
        return any(a_start <= start and end <= a_end for a_start, a_end in au_spans)

    def find_au_spans(text):
        return [(m.start(), m.end()) for m in re.finditer(r"<au>.*?</au>", text, re.DOTALL)]

    for author in author_pairs:
        gnm_raw = author.get("gnm", "")
        snm_raw = author.get("snm", "")

        gnm_norm = normalize_text(gnm_raw) if gnm_raw else ""
        snm_norm = normalize_text(snm_raw) if snm_raw else ""

        gnm_matches = [
            (m.start(), m.end(), m.group(0))
            for m in re.finditer(r"<gnm>(.*?)</gnm>", structured)
            if normalize_text(m.group(1)) == gnm_norm
        ]
        snm_matches = [
            (m.start(), m.end(), m.group(0))
            for m in re.finditer(r"<snm>(.*?)</snm>", structured)
            if normalize_text(m.group(1)) == snm_norm
        ]

        au_spans = find_au_spans(structured)

        for g_start, g_end, g_tag in gnm_matches:
            for s_start, s_end, s_tag in snm_matches:
                # Ensure snm and gnm are adjacent (ignore whitespace, comma)
                start = min(g_start, s_start)
                end = max(g_end, s_end)
                in_between = structured[min(g_end, s_end):max(g_start, s_start)]

                if (
                    len(in_between.strip()) <= 3 and
                    re.fullmatch(r"[,\.\s]*", in_between) and
                    not is_within_au_tag(start, end, au_spans)
                ):
                    wrapped = f"<au>{structured[start:end]}</au>"
                    structured = structured[:start] + wrapped + structured[end:]
                    break
            else:
                continue
            break

        # If only snm present, wrap it alone
        if not gnm_matches and snm_matches:
            for s_start, s_end, s_tag in snm_matches:
                if not is_within_au_tag(s_start, s_end, au_spans):
                    wrapped = f"<au>{s_tag}</au>"
                    structured = structured[:s_start] + wrapped + structured[s_end:]
                break

    return structured



def extract_tagged_elements_journal(xml_string, original_ref):
    root = etree.fromstring(xml_string.encode())

    tagged_parts = []

    edition_match = re.search(r"((\d+\.?(st|nd|rd|th)?)?\s*(eds\.|edition))", original_ref, re.IGNORECASE)
    if edition_match:
        edition_text = edition_match.group(0)
        tagged_parts.append((edition_text, f"<eds>{edition_text}</eds>"))

    match = re.match(r"^(?:\[(\d+)\]|(\d+)\.)", original_ref.strip())
    if match:
        label = match.group(0)  # full matched label, e.g., '[10]' or '10.'
        tagged_parts.append((label, f"<lbl>{label}</lbl>"))

    for role in ["author", "editor"]:
        for pers in root.xpath(f".//tei:{role}", namespaces=ns):
            forenames = pers.xpath(".//tei:forename", namespaces=ns)
            surname = pers.find(".//tei:surname", namespaces=ns)

            # Gather initials (e.g., A, W C) → ['A', 'W', 'C']
            gnm_parts = []
            for forename in forenames:
                part = forename.text.strip() if forename.text else ""
                if part:
                    gnm_parts.extend(part.split())  # handle "W C" as ["W", "C"]

            # Handle various dotted/space forms of GNM
            if gnm_parts:
                original_ref_clean = re.sub(r"[^\w]", "", original_ref).lower()  # remove punctuation for loose match
                gnm_clean = "".join(gnm_parts).lower()

                variants = {
                    ".".join(gnm_parts) + ".": "<gnm>{}</gnm>".format(".".join(gnm_parts) + "."),  # A.W.C.
                    " ".join(gnm_parts): "<gnm>{}</gnm>".format(" ".join(gnm_parts)),  # A W C
                    ". ".join(gnm_parts) + ".": "<gnm>{}</gnm>".format(". ".join(gnm_parts) + "."),  # A. W. C.
                    "".join(gnm_parts) + ".": "<gnm>{}</gnm>".format("".join(gnm_parts) + "."),  # AWC.
                    "".join(gnm_parts): "<gnm>{}</gnm>".format("".join(gnm_parts))  # AWC
                }

                found = False
                print("variants : ", variants)
                # Step 1: Exact match check (prioritized)
                for variant, tag in variants.items():
                    if variant in original_ref:
                        print("Matched exact variant:", variant)
                        tagged_parts.append((variant, tag))
                        found = True
                        break

                # Step 2: Regex fallback (if not found yet)
                if not found:
                    for variant, tag in variants.items():
                        if re.search(re.escape(variant) + r"(?=[,\s<])",
                                     original_ref):  # match followed by comma/space or tag end
                            print("Matched regex fallback variant:", variant)
                            tagged_parts.append((variant, tag))
                            found = True
                            break



            # Surname + suffix tagging
            if surname is not None and surname.text:
                snm = surname.text.strip()
                tagged_parts.append((snm, f"<snm>{snm}</snm>"))
    colab = root.xpath(".//tei:orgName[@type='collaboration']", namespaces=ns)
    if colab and colab[0].text:
        Col_text = colab[0].text.strip()
        tagged_parts.append((Col_text, f"<col>{Col_text}</col>"))
    year = root.xpath(".//tei:date[@type='published']", namespaces=ns)
    if year:
        yr_base = year[0].text.strip()

        # Match 1999, 1999a, 1999b, etc.
        year_pattern = re.compile(rf"\b{yr_base}[a-zA-Z]?\b")

        match = year_pattern.search(original_ref)
        if match:
            matched_year = match.group(0)
            tagged_parts.append((matched_year, f"<yr>{matched_year}</yr>"))


    def normalize_dashes(text):
        return re.sub(r'[–—−]', '-', text)

    dash_separators = ["-", "–", "—", "−"]

    title_elem = root.xpath(".//tei:title[@level='a']", namespaces=ns)
    if not title_elem:
        title_elem = root.xpath(".//tei:title[@level='m']", namespaces=ns)

    if title_elem and title_elem[0].text:
        title_text = html.unescape(title_elem[0].text.strip())
        normalized_title = normalize_dashes(title_text)
        normalized_ref = normalize_dashes(original_ref)

        for sep in dash_separators:
            flexible_title = normalized_title.replace("-", sep)
            pattern = re.escape(flexible_title)
            match = re.search(pattern, original_ref)
            if match:
                matched_original = match.group(0)
                tagged_parts.append((matched_original, f"<atl>{matched_original}</atl>"))
                break

    journal_elem = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    if journal_elem and journal_elem[0].text:
        journal_text = journal_elem[0].text.strip()
        normalized_journal = normalize_dashes(journal_text)
        normalized_ref = normalize_dashes(original_ref)

        for sep in dash_separators:
            flexible_journal = normalized_journal.replace("-", sep)
            pattern = re.escape(flexible_journal)
            match = re.search(pattern, original_ref)
            if match:
                matched_original = match.group(0)
                tagged_parts.append((matched_original, f"<jtl>{matched_original}</jtl>"))
                break

    location = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    volume = root.xpath(".//tei:biblScope[@unit='volume']", namespaces=ns)
    if volume and volume[0].text:
        vol = volume[0].text.strip()
        tagged_parts.append((vol, f"<vol>{vol}</vol>"))

    issue = root.xpath(".//tei:biblScope[@unit='issue']", namespaces=ns)
    if not issue:
        issue = root.xpath(".//tei:biblScope[@unit='note']", namespaces=ns)

    if issue and issue[0].text:
        iss = issue[0].text.strip()
        tagged_parts.append((iss, f"<iss>{iss}</iss>"))
    else:
        # Fallback: Try to find common issue patterns in original_ref directly (not in brackets)
        issue_patterns = [
            r'\b(?:e-)?suppl\.?\s*\w+\b',  # "suppl. 3", "e-suppl A"
            r'\b\d+\s+suppl\b',  # "11 Suppl"
            r'\bsuppl\.?\b',  # "suppl" or "suppl."
            r'\b\d+\s+pt\.?\s*\w+\b',  # "5 Pt 1", "7 pt. A"
            r'\bpt\.?\s*\w+\b',  # "Pt 1", "PT A"
              # "special issue"
            r'\bspecial\s+iss\b',  # "special iss"
        ]

        for pattern in issue_patterns:
            match = re.search(pattern, original_ref, re.IGNORECASE)
            if match:
                issue_text = match.group(0).strip()
                tagged_parts.append((issue_text, f"<iss>{issue_text}</iss>"))
                break

    page = root.xpath(".//tei:biblScope[@unit='page']", namespaces=ns)
    if page:
        pg_from = page[0].get("from", "").strip()
        pg_to = page[0].get("to", "").strip()

        if pg_from and pg_to:
            from_digits = re.sub(r"\D", "", pg_from)
            to_digits = re.sub(r"\D", "", pg_to)

            print("Page range found:", pg_from, pg_to)
            print("From digits:", from_digits)
            print("To digits:", to_digits)
            short_to = to_digits[1:]
            print("Short to digits:", short_to)

            for separator in ["-", "–", "—", "--"]:
                # 1. Full match (including prefixes like S71–S75)
                pattern = re.compile(
                    rf"([A-Za-z]{from_digits}\s*?){separator}(\s*?[A-Za-z]{to_digits})", re.UNICODE
                )

                match = pattern.search(original_ref)

                if match:
                    print("Full match found:", match.group(0))
                    tagged_text = match.group(0)
                    tagged_parts.append((tagged_text, f"<pg>{tagged_text}</pg>"))
                    break

                # 2. Pure numeric match (71–75 or 471–475)
                numeric_pattern = f"{from_digits}{separator}{to_digits}"
                if numeric_pattern in original_ref:
                    tagged_parts.append((numeric_pattern, f"<pg>{pg_from}{separator}{pg_to}</pg>"))
                    break

                # 3. Compressed match (e.g., 471—75)

                if len(short_to) < len(from_digits):
                    # short_to = to_digits  # e.g., 75 from 475
                    compressed_pattern = re.compile(
                        rf"([A-Za-z]*{from_digits}\s*){re.escape(separator)}(\s*[A-Za-z]*{short_to}\b)"
                    )
                    print("Compressed pattern:", compressed_pattern.pattern)
                    match = compressed_pattern.search(original_ref)
                    if match:
                        print("YES")
                        matched_text = match.group(0)
                        tagged_parts.append((matched_text, f"<pg>{matched_text}</pg>"))
                        break

                # 4. Flexible spacing (e.g., "71 — 75")
                flexible_pattern = re.compile(
                    rf"([A-Za-z]*{from_digits})\s*{re.escape(separator)}\s*([A-Za-z]*{to_digits})"
                )
                match = flexible_pattern.search(original_ref)
                if match:
                    matched_text = match.group(0)
                    tagged_parts.append((matched_text, f"<pg>{matched_text}</pg>"))
                    break

            else:
                fallback = f"{pg_from}-{pg_to}"
                tagged_parts.append((fallback, f"<pg>{fallback}</pg>"))

        elif page[0].text:
            pg_text = page[0].text.strip()
            if pg_text in original_ref:
                tagged_parts.append((pg_text, f"<pg>{pg_text}</pg>"))
            else:
                alt_pg = pg_text.rstrip(".")
                if alt_pg in original_ref:
                    tagged_parts.append((alt_pg, f"<pg>{alt_pg}</pg>"))

    # Handle <ptr target="...">
    ptr_elem = root.find(".//tei:ptr", namespaces=ns)
    if ptr_elem is not None:
        # print('PTR')
        target_url = ptr_elem.get("target")
        if target_url:
            if "doi.org/" in target_url.lower():
                tagged_parts.append((target_url, f"<doi>{target_url}</doi>"))
            else:
                tagged_parts.append((target_url, f"<uri>{target_url}</uri>"))

    doi_added = False

    # 1. Prefer full DOI from <ptr target="...">
    ptr_elem = root.find(".//tei:ptr", namespaces=ns)
    if ptr_elem is not None:
        target_url = ptr_elem.get("target")
        if target_url and "doi.org/" in target_url.lower():
            tagged_parts.append((target_url, f"<doi>{target_url}</doi>"))
            doi_added = True
        elif target_url:
            tagged_parts.append((target_url, f"<uri>{target_url}</uri>"))

    # 2. Only add idno DOI if <ptr> didn’t already do it
    if not doi_added:
        doi_el = root.xpath(".//tei:idno[@type='DOI']", namespaces=ns)
        if doi_el:
            doi_value = doi_el[0].text.strip() if doi_el[0].text else ""
            if doi_value:
                tagged_parts.append((doi_value, f"<doi>{doi_value}</doi>"))


    loc_elem = root.xpath(".//tei:meeting/tei:address/tei:addrLine", namespaces=ns)
    if loc_elem and loc_elem[0].text:
        location = loc_elem[0].text.strip()
        # print("Location found:", location)
        if location in original_ref:
            tagged_parts.append((location, f"<loc>{location}</loc>"))

    pub_place_elem = root.xpath(".//tei:pubPlace", namespaces=ns)
    if pub_place_elem and pub_place_elem[0].text:
        pub_place = pub_place_elem[0].text.strip()
        if pub_place and pub_place in original_ref:
            tagged_parts.append((pub_place, f"<loc>{pub_place}</loc>"))
    # print(tagged_parts)
    return tagged_parts


def extract_tagged_elements_book(xml_string, original_ref):
    root = etree.fromstring(xml_string.encode())
    tagged_parts = []
    edition_match = re.search(r"((\d+\.?(st|nd|rd|th)?)?\s*(eds\.|edition))", original_ref, re.IGNORECASE)
    if edition_match:
        edition_text = edition_match.group(0)
        tagged_parts.append((edition_text, f"<eds>{edition_text}</eds>"))

    match = re.match(r"^(?:\[(\d+)\]|(\d+)\.)", original_ref.strip())
    if match:
        label = match.group(0)  # full matched label, e.g., '[10]' or '10.'
        tagged_parts.append((label, f"<lbl>{label}</lbl>"))

    for role in ["author", "editor"]:
        for pers in root.xpath(f".//tei:{role}", namespaces=ns):
            forenames = pers.xpath(".//tei:forename", namespaces=ns)
            surname = pers.find(".//tei:surname", namespaces=ns)

            # Gather initials (e.g., A, W C) → ['A', 'W', 'C']
            gnm_parts = []
            for forename in forenames:
                part = forename.text.strip() if forename.text else ""
                if part:
                    gnm_parts.extend(part.split())  # handle "W C" as ["W", "C"]

            # Handle various dotted/space forms of GNM
            if gnm_parts:
                original_ref_clean = re.sub(r"[^\w]", "", original_ref).lower()  # remove punctuation for loose match
                gnm_clean = "".join(gnm_parts).lower()

                variants = {
                    ".".join(gnm_parts) + ".": "<gnm>{}</gnm>".format(".".join(gnm_parts) + "."),  # A.W.C.
                    " ".join(gnm_parts): "<gnm>{}</gnm>".format(" ".join(gnm_parts)),  # A W C
                    ". ".join(gnm_parts) + ".": "<gnm>{}</gnm>".format(". ".join(gnm_parts) + "."),  # A. W. C.
                    "".join(gnm_parts) + ".": "<gnm>{}</gnm>".format("".join(gnm_parts) + "."),  # AWC.
                    "".join(gnm_parts): "<gnm>{}</gnm>".format("".join(gnm_parts))  # AWC
                }

                found = False
                print("variants : ", variants)
                # Step 1: Exact match check (prioritized)
                for variant, tag in variants.items():
                    if variant in original_ref:
                        print("Matched exact variant:", variant)
                        tagged_parts.append((variant, tag))
                        found = True
                        break

                # Step 2: Regex fallback (if not found yet)
                if not found:
                    for variant, tag in variants.items():
                        if re.search(re.escape(variant) + r"(?=[,\s<])",
                                     original_ref):  # match followed by comma/space or tag end
                            print("Matched regex fallback variant:", variant)
                            tagged_parts.append((variant, tag))
                            found = True
                            break

            # Surname + suffix tagging
            if surname is not None and surname.text:
                snm = surname.text.strip()
                tagged_parts.append((snm, f"<snm>{snm}</snm>"))

    year = root.xpath(".//tei:date[@type='published']", namespaces=ns)
    if year:
        yr_base = year[0].text.strip()

        # Match 1999, 1999a, 1999b, etc.
        year_pattern = re.compile(rf"\b{yr_base}[a-zA-Z]?\b")

        match = year_pattern.search(original_ref)
        if match:
            matched_year = match.group(0)
            tagged_parts.append((matched_year, f"<yr>{matched_year}</yr>"))


    publisher = root.xpath('.//tei:publisher', namespaces=ns)
    if publisher:
        pub_text = publisher[0].text.strip()
        if pub_text and pub_text in original_ref:
            tagged_parts.append((pub_text, f"<pub>{pub_text}</pub>"))

    def normalize_dashes(text):
        return re.sub(r'[–—−]', '-', text)

    dash_separators = ["-", "–", "—", "−"]

    title_elem = root.xpath(".//tei:title[@level='a']", namespaces=ns)
    if not title_elem:
        title_elem = root.xpath(".//tei:title[@level='m']", namespaces=ns)
    if not title_elem:
        title_elem = root.xpath(".//tei:title[@level='s']", namespaces=ns)  # <-- Added this line

    if title_elem and title_elem[0].text:
        title_text = html.unescape(title_elem[0].text.strip())
        normalized_title = normalize_dashes(title_text)
        normalized_ref = normalize_dashes(original_ref)

        for sep in dash_separators:
            flexible_title = normalized_title.replace("-", sep)
            pattern = re.escape(flexible_title)
            match = re.search(pattern, original_ref)
            if match:
                matched_original = match.group(0)
                tagged_parts.append((matched_original, f"<btl>{matched_original}</btl>"))
                break

    journal_elem = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    if journal_elem and journal_elem[0].text:
        journal_text = journal_elem[0].text.strip()
        normalized_journal = normalize_dashes(journal_text)
        normalized_ref = normalize_dashes(original_ref)

        for sep in dash_separators:
            flexible_journal = normalized_journal.replace("-", sep)
            pattern = re.escape(flexible_journal)
            match = re.search(pattern, original_ref)
            if match:
                matched_original = match.group(0)
                tagged_parts.append((matched_original, f"<btl>{matched_original}</btl>"))
                break
    volume = root.xpath(".//tei:biblScope[@unit='volume']", namespaces=ns)
    if volume and volume[0].text:
        vol = volume[0].text.strip()
        tagged_parts.append((vol, f"<vol>{vol}</vol>"))
    issue = root.xpath(".//tei:biblScope[@unit='issue']", namespaces=ns)

    if issue and issue[0].text:
        iss = issue[0].text.strip()
        tagged_parts.append((iss, f"<iss>{iss}</iss>"))
    else:
        # Fallback: Try to find common issue patterns in original_ref directly (not in brackets)
        issue_patterns = [
            r'\b(?:e-)?suppl\.?\s*\w+\b',  # "suppl. 3", "e-suppl A"
            r'\b\d+\s+suppl\b',  # "11 Suppl"
            r'\bsuppl\.?\b',  # "suppl" or "suppl."
            r'\b\d+\s+pt\.?\s*\w+\b',  # "5 Pt 1", "7 pt. A"
            r'\bpt\.?\s*\w+\b',  # "Pt 1", "PT A"
            # "special issue"
            r'\bspecial\s+iss\b',  # "special iss"
        ]

        for pattern in issue_patterns:
            match = re.search(pattern, original_ref, re.IGNORECASE)
            if match:
                issue_text = match.group(0).strip()
                tagged_parts.append((issue_text, f"<iss>{issue_text}</iss>"))
                break

    page = root.xpath(".//tei:biblScope[@unit='page']", namespaces=ns)
    if page:
        pg_from = page[0].get("from", "").strip()
        pg_to = page[0].get("to", "").strip()

        if pg_from and pg_to:
            from_digits = re.sub(r"\D", "", pg_from)
            to_digits = re.sub(r"\D", "", pg_to)

            print("Page range found:", pg_from, pg_to)
            print("From digits:", from_digits)
            print("To digits:", to_digits)
            short_to = to_digits[1:]
            print("Short to digits:", short_to)

            for separator in ["-", "–", "—", "--"]:
                # 1. Full match (including prefixes like S71–S75)
                pattern = re.compile(
                    rf"([A-Za-z]{from_digits}\s*?){separator}(\s*?[A-Za-z]{to_digits})", re.UNICODE
                )

                match = pattern.search(original_ref)

                if match:
                    print("Full match found:", match.group(0))
                    tagged_text = match.group(0)
                    tagged_parts.append((tagged_text, f"<pg>{tagged_text}</pg>"))
                    break

                # 2. Pure numeric match (71–75 or 471–475)
                numeric_pattern = f"{from_digits}{separator}{to_digits}"
                if numeric_pattern in original_ref:
                    tagged_parts.append((numeric_pattern, f"<pg>{pg_from}{separator}{pg_to}</pg>"))
                    break

                # 3. Compressed match (e.g., 471—75)

                if len(short_to) < len(from_digits):
                    # short_to = to_digits  # e.g., 75 from 475
                    compressed_pattern = re.compile(
                        rf"{from_digits}\s*{re.escape(separator)}\s*{short_to}\b"
                    )
                    print("Compressed pattern:", compressed_pattern.pattern)
                    match = compressed_pattern.search(original_ref)
                    if match:
                        print("YES")
                        matched_text = match.group(0)
                        tagged_parts.append((matched_text, f"<pg>{pg_from}{separator}{short_to}</pg>"))
                        break

                # 4. Flexible spacing (e.g., "71 — 75")
                flexible_pattern = re.compile(
                    rf"([A-Za-z]*{from_digits})\s*{re.escape(separator)}\s*([A-Za-z]*{to_digits})"
                )
                match = flexible_pattern.search(original_ref)
                if match:
                    matched_text = match.group(0)
                    tagged_parts.append((matched_text, f"<pg>{matched_text}</pg>"))
                    break

            else:
                fallback = f"{pg_from}-{pg_to}"
                tagged_parts.append((fallback, f"<pg>{fallback}</pg>"))

        elif page[0].text:
            pg_text = page[0].text.strip()
            if pg_text in original_ref:
                tagged_parts.append((pg_text, f"<pg>{pg_text}</pg>"))
            else:
                alt_pg = pg_text.rstrip(".")
                if alt_pg in original_ref:
                    tagged_parts.append((alt_pg, f"<pg>{alt_pg}</pg>"))


    doi_added = False

    # 1. Prefer full DOI from <ptr target="...">
    ptr_elem = root.find(".//tei:ptr", namespaces=ns)
    if ptr_elem is not None:
        target_url = ptr_elem.get("target")
        if target_url and "doi.org/" in target_url.lower():
            tagged_parts.append((target_url, f"<doi>{target_url}</doi>"))
            doi_added = True
        elif target_url:
            tagged_parts.append((target_url, f"<uri>{target_url}</uri>"))

    # 2. Only add idno DOI if <ptr> didn’t already do it
    if not doi_added:
        doi_el = root.xpath(".//tei:idno[@type='DOI']", namespaces=ns)
        if doi_el:
            doi_value = doi_el[0].text.strip() if doi_el[0].text else ""
            if doi_value:
                tagged_parts.append((doi_value, f"<doi>{doi_value}</doi>"))

        # 2. Match full DOI URL in original_ref (with <...>)
        # full_doi_url_match = re.search(r"<(https?://dx\.doi\.org/[^>]+)>", original_ref)
        # if full_doi_url_match:
        #     full_url = full_doi_url_match.group(1)
        #     tagged_parts.append((f"<{full_url}>", f"<doi><http://dx.doi.org/{doi_value}</doi></http>"))
        # else:
        #     # fallback to just DOI
        #     tagged_parts.append((doi_value, f"<doi>{doi_value}</doi>"))

    loc_elem = root.xpath(".//tei:meeting/tei:address/tei:addrLine", namespaces=ns)
    if loc_elem and loc_elem[0].text:
        location = loc_elem[0].text.strip()
        # print("Location found:", location)
        if location in original_ref:
            tagged_parts.append((location, f"<loc>{location}</loc>"))

    pub_place_elem = root.xpath(".//tei:pubPlace", namespaces=ns)
    if pub_place_elem and pub_place_elem[0].text:
        pub_place = pub_place_elem[0].text.strip()
        if pub_place and pub_place in original_ref:
            tagged_parts.append((pub_place, f"<loc>{pub_place}</loc>"))
    # print(tagged_parts)
    return tagged_parts


def convert_first_btl_to_ctl(tagged_ref):
    btl_tags = re.findall(r"<btl>.*?</btl>", tagged_ref, re.DOTALL)

    if len(btl_tags) >= 1:
        return re.sub(r"<btl>(.*?)</btl>", r"<ctl>\1</ctl>", tagged_ref, count=1, flags=re.DOTALL)

    return tagged_ref


def extract_tagged_elements_chapter(xml_string, original_ref):
    root = etree.fromstring(xml_string.encode())
    tagged_parts = []

    match = re.match(r"^(?:\[(\d+)\]|(\d+)\.)", original_ref.strip())
    if match:
        label = match.group(0)  # full matched label, e.g., '[10]' or '10.'
        tagged_parts.append((label, f"<lbl>{label}</lbl>"))

    edition_match = re.search(r"((\d+\.?(st|nd|rd|th)?)?\s*(eds\.|edition))", original_ref, re.IGNORECASE)
    if edition_match:
        edition_text = edition_match.group(0)
        tagged_parts.append((edition_text, f"<eds>{edition_text}</eds>"))

    for role in ["author", "editor"]:
        for pers in root.xpath(f".//tei:{role}", namespaces=ns):
            forenames = pers.xpath(".//tei:forename", namespaces=ns)
            surname = pers.find(".//tei:surname", namespaces=ns)

            # Gather initials (e.g., A, W C) → ['A', 'W', 'C']
            gnm_parts = []
            for forename in forenames:
                part = forename.text.strip() if forename.text else ""
                if part:
                    gnm_parts.extend(part.split())  # handle "W C" as ["W", "C"]

            # Handle various dotted/space forms of GNM
            if gnm_parts:
                original_ref_clean = re.sub(r"[^\w]", "", original_ref).lower()  # remove punctuation for loose match
                gnm_clean = "".join(gnm_parts).lower()

                variants = {
                    ".".join(gnm_parts) + ".": "<gnm>{}</gnm>".format(".".join(gnm_parts) + "."),  # A.W.C.
                    " ".join(gnm_parts): "<gnm>{}</gnm>".format(" ".join(gnm_parts)),  # A W C
                    ". ".join(gnm_parts) + ".": "<gnm>{}</gnm>".format(". ".join(gnm_parts)),  # A. W. C.
                    "".join(gnm_parts) + ".": "<gnm>{}</gnm>".format("".join(gnm_parts) + "."),  # AWC.
                    "".join(gnm_parts): "<gnm>{}</gnm>".format("".join(gnm_parts))  # AWC
                }

                found = False
                print("variants : ", variants)
                # Step 1: Exact match check (prioritized)
                for variant, tag in variants.items():
                    if variant in original_ref:
                        print("Matched exact variant:", variant)
                        tagged_parts.append((variant, tag))
                        found = True
                        break

                # Step 2: Regex fallback (if not found yet)
                if not found:
                    for variant, tag in variants.items():
                        if re.search(re.escape(variant) + r"(?=[,\s<])",
                                     original_ref):  # match followed by comma/space or tag end
                            print("Matched regex fallback variant:", variant)
                            tagged_parts.append((variant, tag))
                            found = True
                            break

            # Surname + suffix tagging
            if surname is not None and surname.text:
                snm = surname.text.strip()
                tagged_parts.append((snm, f"<snm>{snm}</snm>"))

    year = root.xpath(".//tei:date[@type='published']", namespaces=ns)
    if year:
        yr_base = year[0].text.strip()

        # Match 1999, 1999a, 1999b, etc.
        year_pattern = re.compile(rf"\b{yr_base}[a-zA-Z]?\b")

        match = year_pattern.search(original_ref)
        if match:
            matched_year = match.group(0)
            tagged_parts.append((matched_year, f"<yr>{matched_year}</yr>"))


    publisher = root.xpath('.//tei:publisher', namespaces=ns)
    if publisher:
        pub_text = publisher[0].text.strip()
        if pub_text and pub_text in original_ref:
            tagged_parts.append((pub_text, f"<pub>{pub_text}</pub>"))

    def normalize_dashes(text):
        return re.sub(r'[–—−]', '-', text)

    dash_separators = ["-", "–", "—", "−"]

    # Capture both chapter and book titles
    title_elems = root.xpath(".//tei:title[@level='a' or @level='m' or @level='s']", namespaces=ns)

    for title_elem in title_elems:
        if title_elem.text:
            title_text = html.unescape(title_elem.text.strip())
            normalized_title = normalize_dashes(title_text)
            normalized_ref = normalize_dashes(original_ref)

            for sep in dash_separators:
                flexible_title = normalized_title.replace("-", sep)
                pattern = re.escape(flexible_title)
                match = re.search(pattern, original_ref)
                if match:
                    matched_original = match.group(0)
                    tagged_parts.append((matched_original, f"<btl>{matched_original}</btl>"))
                    break


    journal_elem = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    if journal_elem and journal_elem.text:
        journal_text = journal_elem.text.strip()
        normalized_journal = normalize_dashes(journal_text)
        normalized_ref = normalize_dashes(original_ref)

        for sep in dash_separators:
            flexible_journal = normalized_journal.replace("-", sep)
            pattern = re.escape(flexible_journal)
            match = re.search(pattern, original_ref)
            if match:
                matched_original = match.group(0)
                tagged_parts.append((matched_original, f"<btl>{matched_original}</btl>"))
                break

    issue = root.xpath(".//tei:biblScope[@unit='issue']", namespaces=ns)
    if not issue:
        issue = root.xpath(".//tei:biblScope[@unit='note']", namespaces=ns)

    if issue and issue[0].text:
        iss = issue[0].text.strip()
        tagged_parts.append((iss, f"<iss>{iss}</iss>"))
    else:
        # Fallback: Try to find common issue patterns in original_ref directly (not in brackets)
        issue_patterns = [
            r'\b(?:e-)?suppl\.?\s*\w+\b',  # "suppl. 3", "e-suppl A"
            r'\b\d+\s+suppl\b',  # "11 Suppl"
            r'\bsuppl\.?\b',  # "suppl" or "suppl."
            r'\b\d+\s+pt\.?\s*\w+\b',  # "5 Pt 1", "7 pt. A"
            r'\bpt\.?\s*\w+\b',  # "Pt 1", "PT A"
            # "special issue"
            r'\bspecial\s+iss\b',  # "special iss"
        ]

        for pattern in issue_patterns:
            match = re.search(pattern, original_ref, re.IGNORECASE)
            if match:
                issue_text = match.group(0).strip()
                tagged_parts.append((issue_text, f"<iss>{issue_text}</iss>"))
                break

    page = root.xpath(".//tei:biblScope[@unit='page']", namespaces=ns)
    if page:
        pg_from = page[0].get("from", "").strip()
        pg_to = page[0].get("to", "").strip()

        if pg_from and pg_to:
            from_digits = re.sub(r"\D", "", pg_from)
            to_digits = re.sub(r"\D", "", pg_to)

            print("Page range found:", pg_from, pg_to)
            print("From digits:", from_digits)
            print("To digits:", to_digits)
            short_to = to_digits[1:]
            print("Short to digits:", short_to)

            for separator in ["-", "–", "—", "--"]:
                # 1. Full match (including prefixes like S71–S75)
                pattern = re.compile(
                    rf"([A-Za-z]{from_digits}\s*?){separator}(\s*?[A-Za-z]{to_digits})", re.UNICODE
                )

                match = pattern.search(original_ref)

                if match:
                    print("Full match found:", match.group(0))
                    tagged_text = match.group(0)
                    tagged_parts.append((tagged_text, f"<pg>{tagged_text}</pg>"))
                    break

                # 2. Pure numeric match (71–75 or 471–475)
                numeric_pattern = f"{from_digits}{separator}{to_digits}"
                if numeric_pattern in original_ref:
                    tagged_parts.append((numeric_pattern, f"<pg>{pg_from}{separator}{pg_to}</pg>"))
                    break

                # 3. Compressed match (e.g., 471—75)

                if len(short_to) < len(from_digits):
                    # short_to = to_digits  # e.g., 75 from 475
                    compressed_pattern = re.compile(
                        rf"{from_digits}\s*{re.escape(separator)}\s*{short_to}\b"
                    )
                    print("Compressed pattern:", compressed_pattern.pattern)
                    match = compressed_pattern.search(original_ref)
                    if match:
                        print("YES")
                        matched_text = match.group(0)
                        tagged_parts.append((matched_text, f"<pg>{pg_from}{separator}{short_to}</pg>"))
                        break

                # 4. Flexible spacing (e.g., "71 — 75")
                flexible_pattern = re.compile(
                    rf"([A-Za-z]*{from_digits})\s*{re.escape(separator)}\s*([A-Za-z]*{to_digits})"
                )
                match = flexible_pattern.search(original_ref)
                if match:
                    matched_text = match.group(0)
                    tagged_parts.append((matched_text, f"<pg>{matched_text}</pg>"))
                    break

            else:
                fallback = f"{pg_from}-{pg_to}"
                tagged_parts.append((fallback, f"<pg>{fallback}</pg>"))

        elif page[0].text:
            pg_text = page[0].text.strip()
            if pg_text in original_ref:
                tagged_parts.append((pg_text, f"<pg>{pg_text}</pg>"))
            else:
                alt_pg = pg_text.rstrip(".")
                if alt_pg in original_ref:
                    tagged_parts.append((alt_pg, f"<pg>{alt_pg}</pg>"))

    doi_added = False

    # 1. Prefer full DOI from <ptr target="...">
    ptr_elem = root.find(".//tei:ptr", namespaces=ns)
    if ptr_elem is not None:
        target_url = ptr_elem.get("target")
        if target_url and "doi.org/" in target_url.lower():
            tagged_parts.append((target_url, f"<doi>{target_url}</doi>"))
            doi_added = True
        elif target_url:
            tagged_parts.append((target_url, f"<uri>{target_url}</uri>"))

    # 2. Only add idno DOI if <ptr> didn’t already do it
    if not doi_added:
        doi_el = root.xpath(".//tei:idno[@type='DOI']", namespaces=ns)
        if doi_el:
            doi_value = doi_el[0].text.strip() if doi_el[0].text else ""
            if doi_value:
                tagged_parts.append((doi_value, f"<doi>{doi_value}</doi>"))

    loc_elem = root.xpath(".//tei:meeting/tei:address/tei:addrLine", namespaces=ns)
    if loc_elem and loc_elem[0].text:
        location = loc_elem[0].text.strip()
        norm_location = normalize(location)
        norm_original = normalize(original_ref)

        # print("Location found:", location)
        if norm_location in norm_original:
            tagged_parts.append((location, f"<loc>{location}</loc>"))
        else:
            print("Location not in original reference:", location)

    pub_place_elem = root.xpath(".//tei:pubPlace", namespaces=ns)
    if pub_place_elem and pub_place_elem[0].text:
        pub_place = pub_place_elem[0].text.strip()
        if pub_place and pub_place in original_ref:
            tagged_parts.append((pub_place, f"<loc>{pub_place}</loc>"))
    # print(tagged_parts)
    return tagged_parts


def tag_suffixes(structured):
    # Define common suffix patterns
    suffix_patterns = [
        r"Jr\.?", r"Sr\.?", r"II", r"III", r"IV", r"VI", r"VII", r"VIII", r"IX", r"X",

       r"Ph\.?D\.?", r"M\.?D\.?", r"Esq\.?", r"DDS", r"DVM"
    ]

    # Create combined suffix pattern (with word boundaries)
    suffix_pattern_str = '|'.join(suffix_patterns)
    suffix_regex = re.compile(rf"\b({suffix_pattern_str})\b", re.IGNORECASE)

    # Step 1: Replace plain text suffixes
    structured = suffix_regex.sub(lambda m: f"<suff>{m.group(1)}</suff>", structured)

    # Step 2: Replace suffixes incorrectly tagged inside <gnm> or <snm>
    for tag in ['gnm', 'snm']:
        tag_pattern = re.compile(rf"(<{tag}><suff>({suffix_pattern_str})</suff></{tag}>)", re.IGNORECASE)
        structured = tag_pattern.sub(r"<suff>\2</suff>", structured)
    return structured


# Apply it to structured ref
import re

import re

def fix_au_closing_before_suffix(structured):
    """
    Moves </au> tag after </suff> if it comes before it, preserving punctuation and whitespace.
    Example: </au>, <suff>Jr</suff> → <suff>Jr</suff>, </au>
    """
    pattern = re.compile(r"(</au>)([\s.,;:]*)(<suff>.*?</suff>)")

    # Reorder: suffix first, then punctuation, then </au>
    return pattern.sub(r"\2\3\1", structured)




from flask import Flask, request, jsonify
from collections import OrderedDict

app = Flask(__name__)


@app.route('/convert_Ref', methods=['POST'])
def process_references():
    original_ref = request.get_data(as_text=True)
    # if not isinstance(input_data, list):
    #     input_data = [input_data]
    print("Received input:", original_ref)
    results = []



    try:

        decoded_ref = decode_entities(original_ref)
        # Detect reference type
        ref_type = 'journal'


        decoded_ref, tags_info = remove_style_tags(decoded_ref)
        # print("After removing style tags:", decoded_ref)
        # Send to GROBID
        response = requests.post(
            GROBID_URL,
            headers={"Accept": "application/xml"},
            data={"citations": decoded_ref}
        )
        if response.status_code != 200:
            raise Exception("GROBID request failed")

        xml_string = response.text
        # print("Received XML from GROBID:", xml_string)
        with open(XML_OUTPUT_FILE, "w", encoding="utf-8") as xml_out:
            xml_out.write(f"Original: {decoded_ref}\n")
            xml_out.write(xml_string.strip() + "\n\n")

        wrapped_xml = wrap_with_tei_namespace(xml_string)

        # Extract tags by type
        if ref_type == "journal":
            tagged_elements = extract_tagged_elements_journal(wrapped_xml, decoded_ref)
        elif ref_type == "book":
            tagged_elements = extract_tagged_elements_book(wrapped_xml, decoded_ref)
        elif ref_type == "chapter":
            tagged_elements = extract_tagged_elements_chapter(wrapped_xml, decoded_ref)
        else:
            tagged_elements = []
        # print(f"[{process_id}] Tagged elements: {tagged_elements}")
        # Compose structured tagged output
        print(tagged_elements)
        structured = tag_reference(ref_type, decoded_ref, tagged_elements)

        # Restore style tags
        structured = restore_style_tags(structured, tags_info)
        structured = restore_entities(structured, original_ref)
        root = etree.fromstring(wrapped_xml.encode())
        author_pairs = extract_author_pairs(root)
        structured = wrap_authors_loose_match(structured, author_pairs)
        structured = tag_suffixes(structured)
        # Wrap "et al." in <etal> if present
        structured = re.sub(r'\bet al\.', r'<etal>et al.</etal>', structured, flags=re.IGNORECASE)
        structured=fix_au_closing_before_suffix(structured)
        # structured = wrap_authors(structured)

        # Adjust chapter structure
        if ref_type == "chapter":
            structured = convert_first_btl_to_ctl(structured)
        # print(f"[{process_id}] Structured output: {structured}")
        # Final XML wrap
        structured_ref = f"""<mixed-citation publication-type="{ref_type}">{structured}</mixed-citation>"""
        # print(f"\n\n[{process_id}] Processed successfully: {structured_ref}\n\n")
    except Exception as e:
        print("Error processing reference:", str(e))

        structured_ref = None

    results.append(OrderedDict([
        # ("process_id", process_id),
        # ("element_outer_html", html_ref),
        ("structuredXML", structured_ref)
    ]))
    # print("Results:", results)
    return results


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=6000, debug=True)