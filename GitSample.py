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





def analyze_type(ref):

    prompt = f"""

You are a reference classification expert. Your job is to classify each reference into one of the following types:

- journal – if the reference is from a journal article (typically includes volume, issue, pages, journal name)
- book – if the reference is a complete book (has publisher, location, no "In:" or editors)
- chapter – if the reference is a chapter in a book (usually contains "In:", editors, book title, and publisher)

Only return one of these three tags: journal, book, or chapter. Do not explain.

Reference:
{ref}


Your response:
"""


    response = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,  
        temperature=0
    )

    type = response['choices'][0]['message']['content'].strip()



    return type



def normalize(text):
    return re.sub(r'[^\w\s]', '', text).lower()

def normalize_dashes(text):
    return re.sub(r'[–—−]', '-', text)



def add_unstruct_tag(tagged_output, original_ref):
    """
    Wrap any untagged plain-text chunks in <unstruct>, but do not wrap inside existing tags.
    """
    from xml.etree.ElementTree import fromstring, tostring, Element
    import re

    # Regex to match <tag>...</tag> or <tag/> or <tag attr="...">...</tag>
    xml_tag_pattern = re.compile(r"(</?\w+[^>]*>)")

    # Split by XML tags to isolate raw text portions
    segments = xml_tag_pattern.split(tagged_output)

    new_segments = []
    for segment in segments:
        if xml_tag_pattern.match(segment):
            new_segments.append(segment)
        else:
            # Clean segment
            clean = segment.strip()
            if clean:
                new_segments.append(f"<unstruct>{segment}</unstruct>")
            else:
                new_segments.append(segment)
    return ''.join(new_segments)




def tag_reference(type,original_ref, tagged_parts):
    tagged_ref = original_ref
    tagged_parts = sorted(tagged_parts, key=lambda x: -len(x[0]))
    # print("Tagged parts sorted by length:", tagged_parts)
    # for raw, tag in tagged_parts:
        # normalized_raw = normalize_dashes(raw)

        # normalized_ref = normalize_dashes(tagged_ref)
        # match_pos = normalized_ref.find(normalized_raw)

        # if match_pos != -1:
        #     original_match = tagged_ref[match_pos:match_pos + len(raw)]

        #     tag_match = re.match(rf"(?<![>\w]){pattern}(?![\w<])", tag)
        #     if tag_match:
        #         tag_name = tag_match.group("tag")
        #         replacement = f"<{tag_name}>{original_match}</{tag_name}>"

        #         tagged_ref = (tagged_ref[:match_pos] + replacement + tagged_ref[match_pos + len(original_match):])
    for raw, tag in tagged_parts:
        pattern = re.escape(raw)
        regex = rf"(?<![>\w]){pattern}(?![\w<])"
        tagged_ref = re.sub(regex, tag, tagged_ref, count=1)
        normalized_raw = normalize_dashes(raw)

        normalized_ref = normalize_dashes(tagged_ref)
        match_pos = normalized_ref.find(normalized_raw)

        if match_pos != -1:
            original_match = tagged_ref[match_pos:match_pos + len(raw)]

            tag_match = re.match(rf"(?<![>\w]){pattern}(?![\w<])", tag)
            if tag_match:
                tag_name = tag_match.group("tag")
                replacement = f"<{tag_name}>{original_match}</{tag_name}>"

                tagged_ref = (tagged_ref[:match_pos] + replacement + tagged_ref[match_pos + len(original_match):])

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
    """Removes <i> and <b> tags and stores content with position to reinsert later."""
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
    """Inserts <i> or <b> tags back into tagged reference output."""
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




def extract_tagged_elements_journal(xml_string, original_ref):

    root = etree.fromstring(xml_string.encode())
  
    tagged_parts = []

    match = re.match(r"^\[(\d+)\]", original_ref.strip())

    if match:
        label = match.group(0)
        print("Label found:", label)
        tagged_parts.append((label, f"<lbl>{label}</lbl>"))

    for pers in root.xpath(".//tei:author", namespaces=ns):
        forenames = pers.xpath(".//tei:forename", namespaces=ns)
        surnames = pers.xpath(".//tei:surname", namespaces=ns)

        for forename in forenames:
            gname = forename.text.strip() if forename.text else ""
            if not gname:
                continue
            gname_with_dot = f"{gname}."
            if gname_with_dot in original_ref:
                tagged_parts.append((gname_with_dot, f"<au><gnm>{gname_with_dot}</gnm></au>"))
            elif gname in original_ref:
                tagged_parts.append((gname, f"<au><gnm>{gname}</gnm></au>"))

        if surnames and surnames[0].text:
            snm = surnames[0].text.strip()
            if snm.lower() in ["jr", "jr."]:
                tagged_parts.append((snm, f"<suff>{snm}</suff>"))
            else:
                tagged_parts.append((snm, f"<au><snm>{snm}</snm></au>"))

    year = root.xpath(".//tei:date[@type='published']", namespaces=ns)
    if not year:
        year = root.xpath(".//tei:date", namespaces=ns)
    if year and year[0].text:
        yr = year[0].text.strip()
        tagged_parts.append((yr, f"<yr>{yr}</yr>"))

    title_elem = root.xpath(".//tei:title[@level='a']", namespaces=ns)
    if not title_elem:
        title_elem = root.xpath(".//tei:title[@level='m']", namespaces=ns)
    if title_elem and title_elem[0].text:
        title_text = html.unescape(title_elem[0].text.strip())
        tagged_parts.append((title_text, f"<atl>{title_text}</atl>"))

    journal_elem = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    if journal_elem and journal_elem[0].text:
        journal_text = journal_elem[0].text.strip()
        if f"<i>{journal_text}</i>" in original_ref:
            tagged_parts.append((f"<i>{journal_text}</i>", f"<jtl><i>{journal_text}</i></jtl>"))
        else:
            tagged_parts.append((journal_text, f"<jtl>{journal_text}</jtl>"))
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

    page = root.xpath(".//tei:biblScope[@unit='page']", namespaces=ns)
    if page:
        pg_from = page[0].get("from", "").strip()
        pg_to = page[0].get("to", "").strip()
        if pg_from and pg_to:
            # Try matching range with various separators
            for separator in ["-", "–" ,"—","-"]:
                pg_pattern1 = f"{pg_from}{separator}{pg_to}"
                pg_to2 = pg_to[1:]
                pg_pattern2 = f"{pg_from}{separator}{pg_to2}"
                if pg_pattern1 in original_ref:
                    tagged_parts.append((pg_pattern1, f"<pg>{pg_from}{separator}{pg_to}</pg>"))
                    break
                elif pg_pattern2 in original_ref:
                    tagged_parts.append((pg_pattern2, f"<pg>{pg_from}{separator}{pg_to2}</pg>"))
                    break
            else:
                print(f"{pg_from}-{pg_to}", f"<pg>{pg_from}-{pg_to}</pg>")
                tagged_parts.append((f"{pg_from}-{pg_to}", f"<pg>{pg_from}-{pg_to}</pg>"))

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


    doi_elem = root.xpath(".//tei:idno[@type='DOI']", namespaces=ns)
    if doi_elem and doi_elem[0].text:
        doi = doi_elem[0].text.strip()
        if doi in original_ref:
            tagged_parts.append((doi, f"<doi>{doi}</doi>"))

    loc_elem = root.xpath(".//tei:meeting/tei:address/tei:addrLine", namespaces=ns)
    if loc_elem and loc_elem[0].text:
        location = loc_elem[0].text.strip()
        # print("Location found:", location)
        if location in original_ref:
            tagged_parts.append((location, f"<loc>{location}</loc>"))
    # print(tagged_parts)
    return tagged_parts



def extract_tagged_elements_book(xml_string, original_ref):
    root = etree.fromstring(xml_string.encode())
    tagged_parts = []

    match = re.match(r"^\[(\d+)\]", original_ref.strip())
    if match:
        label = match.group(0)
        tagged_parts.append((label, f"<lbl>{label}</lbl>"))

    edition_match = re.search(r"(\d+\.?(st|nd|rd|th)?\s*(ed\.|edition))", original_ref, re.IGNORECASE)
    if edition_match:
        edition_text = edition_match.group(0)
        tagged_parts.append((edition_text, f"<edn>{edition_text}</edn>"))
    for role in ["author", "editor"]:
        for pers in root.xpath(f".//tei:{role}", namespaces=ns):
            forenames = pers.xpath(".//tei:forename", namespaces=ns)
            surname = pers.find(".//tei:surname", namespaces=ns)

            # Gather initials with dots (B.K., P.G., etc.)
            gnm_parts = []
            for forename in forenames:
                part = forename.text.strip() if forename.text else ""
                if part:
                    gnm_parts.append(part)

            if gnm_parts:
                gnm_spaced = " ".join(gnm_parts)                     # "N K"
                gnm_dotted = ".".join(gnm_parts) + "."               # "N.K."
                gnm_dotted_nospace = "".join(gnm_parts) + "."        # "NK."
                gnm_dot_space = ". ".join(gnm_parts) + "."
                gnm_without_space="".join(gnm_parts)           # "N. K." ← NEW

                if gnm_dot_space in original_ref:
                    tagged_parts.append((gnm_dot_space, f"<au><gnm>{gnm_dot_space}</gnm></au>"))
                elif gnm_dotted in original_ref:
                    tagged_parts.append((gnm_dotted, f"<au><gnm>{gnm_dotted}</gnm></au>"))
                elif gnm_spaced in original_ref:
                    tagged_parts.append((gnm_spaced, f"<au><gnm>{gnm_spaced}</gnm></au>"))
                elif gnm_dotted_nospace in original_ref:
                    tagged_parts.append((gnm_dotted_nospace, f"<au><gnm>{gnm_dotted_nospace}</gnm></au>"))
                elif gnm_without_space in original_ref:
                    tagged_parts.append((gnm_without_space, f"<au><gnm>{gnm_without_space}</gnm></au>"))
                else:
                    # fallback
                    gnm_joined = " ".join(gnm_parts)
                    tagged_parts.append((gnm_joined, f"<au><gnm>{gnm_joined}</gnm></au>"))


            # Surname handling
            if surname is not None and surname.text:
                snm = surname.text.strip()
                if snm.lower() in ["jr", "jr."]:
                    tagged_parts.append((snm, f"<suff>{snm}</suff>"))
                else:
                    tagged_parts.append((snm, f"<snm>{snm}</snm>"))



    year = root.xpath(".//tei:date[@type='published']", namespaces=ns)
    if not year:
        year = root.xpath(".//tei:date", namespaces=ns)
    if year and year[0].text:
        yr = year[0].text.strip()
        tagged_parts.append((yr, f"<yr>{yr}</yr>"))

    publisher = root.xpath('.//tei:publisher', namespaces=ns)
    if publisher:
        pub_text = publisher[0].text.strip()
        if pub_text and pub_text in original_ref:
            tagged_parts.append((pub_text, f"<pub>{pub_text}</pub>"))

    title_elem = root.xpath(".//tei:title[@level='a']", namespaces=ns)
    if not title_elem:
        title_elem = root.xpath(".//tei:title[@level='m']", namespaces=ns)
    if not title_elem:
        title_elem = root.xpath(".//tei:title[@level='s']", namespaces=ns)  # <-- Added this line

    if title_elem and title_elem[0].text:
        title_text = html.unescape(title_elem[0].text.strip())
        tagged_parts.append((title_text, f"<btl>{title_text}</btl>"))



    journal_elem = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    if journal_elem and journal_elem[0].text:
        journal_text = journal_elem[0].text.strip()
        if f"<i>{journal_text}</i>" in original_ref:
            tagged_parts.append((f"<i>{journal_text}</i>", f"<btl><i>{journal_text}</i></btl>"))
        else:
            tagged_parts.append((journal_text, f"<btl>{journal_text}</btl>"))
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

    page = root.xpath(".//tei:biblScope[@unit='page']", namespaces=ns)
    if page:
        # Case 1: Page range using attributes (from–to)
        pg_from = page[0].get("from", "").strip()
        pg_to = page[0].get("to", "").strip()

        if pg_from and pg_to:
            # Try matching range with various separators
            for separator in ["-", "–" ,"—","-"]:
                pg_pattern1 = f"{pg_from}{separator}{pg_to}"
                pg_to2 = pg_to[1:]
                pg_pattern2 = f"{pg_from}{separator}{pg_to2}"
                if pg_pattern1 in original_ref:
                    tagged_parts.append((pg_pattern1, f"<pg>{pg_from}{separator}{pg_to}</pg>"))
                    break
                elif pg_pattern2 in original_ref:
                    tagged_parts.append((pg_pattern2, f"<pg>{pg_from}{separator}{pg_to2}</pg>"))
                    break
            else:
                print(f"{pg_from}-{pg_to}", f"<pg>{pg_from}-{pg_to}</pg>")
                tagged_parts.append((f"{pg_from}-{pg_to}", f"<pg>{pg_from}-{pg_to}</pg>"))

        # Case 2: Only text content like <biblScope unit="page">17</biblScope>
        elif page[0].text:
            pg_text = page[0].text.strip()
            if pg_text in original_ref:
                tagged_parts.append((pg_text, f"<pg>{pg_text}</pg>"))


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



 






def convert_first_btl_to_ctl(tagged_ref):
    pattern = re.compile(r"<btl>(.*?)</btl>", re.DOTALL)
    return pattern.sub(r"<ctl>\1</ctl>", tagged_ref, count=1)


def extract_tagged_elements_chapter(xml_string, original_ref):
    root = etree.fromstring(xml_string.encode())
    tagged_parts = []

    match = re.match(r"^\[(\d+)\]", original_ref.strip())
    if match:
        label = match.group(0)
        tagged_parts.append((label, f"<lbl>{label}</lbl>"))

    edition_match = re.search(r"(\d+\.?(st|nd|rd|th)?\s*(ed\.|edition))", original_ref, re.IGNORECASE)
    if edition_match:
        edition_text = edition_match.group(0)
        tagged_parts.append((edition_text, f"<edn>{edition_text}</edn>"))
    for role in ["author", "editor"]:
        for pers in root.xpath(f".//tei:{role}", namespaces=ns):
            forenames = pers.xpath(".//tei:forename", namespaces=ns)
            surname = pers.find(".//tei:surname", namespaces=ns)

            # Gather initials with dots (B.K., P.G., etc.)
            gnm_parts = []
            for forename in forenames:
                part = forename.text.strip() if forename.text else ""
                if part:
                    gnm_parts.append(part)

            if gnm_parts:
                gnm_spaced = " ".join(gnm_parts)                     # "N K"
                gnm_dotted = ".".join(gnm_parts) + "."               # "N.K."
                gnm_dotted_nospace = "".join(gnm_parts) + "."        # "NK."
                gnm_dot_space = ". ".join(gnm_parts) + "."
                gnm_without_space="".join(gnm_parts)           # "N. K." ← NEW

                if gnm_dot_space in original_ref:
                    tagged_parts.append((gnm_dot_space, f"<au><gnm>{gnm_dot_space}</gnm></au>"))
                elif gnm_dotted in original_ref:
                    tagged_parts.append((gnm_dotted, f"<au><gnm>{gnm_dotted}</gnm></au>"))
                elif gnm_spaced in original_ref:
                    tagged_parts.append((gnm_spaced, f"<au><gnm>{gnm_spaced}</gnm></au>"))
                elif gnm_dotted_nospace in original_ref:
                    tagged_parts.append((gnm_dotted_nospace, f"<au><gnm>{gnm_dotted_nospace}</gnm></au>"))
                elif gnm_without_space in original_ref:
                    tagged_parts.append((gnm_without_space, f"<au><gnm>{gnm_without_space}</gnm></au>"))
                else:
                    # fallback
                    gnm_joined = " ".join(gnm_parts)
                    tagged_parts.append((gnm_joined, f"<au><gnm>{gnm_joined}</gnm></au>"))


            # Surname handling
            if surname is not None and surname.text:
                snm = surname.text.strip()
                if snm.lower() in ["jr", "jr."]:
                    tagged_parts.append((snm, f"<suff>{snm}</suff>"))
                else:
                    tagged_parts.append((snm, f"<au><snm>{snm}</snm></au>"))



    year = root.xpath(".//tei:date[@type='published']", namespaces=ns)
    if not year:
        year = root.xpath(".//tei:date", namespaces=ns)
    if year and year[0].text:
        yr = year[0].text.strip()
        tagged_parts.append((yr, f"<yr>{yr}</yr>"))

    publisher = root.xpath('.//tei:publisher', namespaces=ns)
    if publisher:
        pub_text = publisher[0].text.strip()
        if pub_text and pub_text in original_ref:
            tagged_parts.append((pub_text, f"<pub>{pub_text}</pub>"))

    # Capture both chapter and book titles
    title_elems = root.xpath(".//tei:title[@level='a' or @level='m' or @level='s']", namespaces=ns)

    for title_elem in title_elems:
        if title_elem.text:
            title_text = html.unescape(title_elem.text.strip())
            if title_text in original_ref:
                tagged_parts.append((title_text, f"<btl>{title_text}</btl>"))
            else:
                print(f"⚠️ Title '{title_text}' not found in original ref: {original_ref}")




    journal_elem = root.xpath(".//tei:title[@level='j']", namespaces=ns)
    if journal_elem and journal_elem[0].text:
        journal_text = journal_elem[0].text.strip()
        
        tagged_parts.append((journal_text, f"<btl>{journal_text}</btl>"))
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

    page = root.xpath(".//tei:biblScope[@unit='page']", namespaces=ns)
    if page:
        # Case 1: Page range using attributes (from–to)
        pg_from = page[0].get("from", "").strip()
        pg_to = page[0].get("to", "").strip()

        if pg_from and pg_to:
            # Try matching range with various separators
            for separator in ["-", "–" ,"—","-"]:
                pg_pattern1 = f"{pg_from}{separator}{pg_to}"
                pg_to2 = pg_to[1:]
                pg_pattern2 = f"{pg_from}{separator}{pg_to2}"
                if pg_pattern1 in original_ref:
                    tagged_parts.append((pg_pattern1, f"<pg>{pg_from}{separator}{pg_to}</pg>"))
                    break
                elif pg_pattern2 in original_ref:
                    tagged_parts.append((pg_pattern2, f"<pg>{pg_from}{separator}{pg_to2}</pg>"))
                    break
            else:
                print(f"{pg_from}-{pg_to}", f"<pg>{pg_from}-{pg_to}</pg>")
                tagged_parts.append((f"{pg_from}-{pg_to}", f"<pg>{pg_from}-{pg_to}</pg>"))

        # Case 2: Only text content like <biblScope unit="page">17</biblScope>
        elif page[0].text:
            pg_text = page[0].text.strip()
            if pg_text in original_ref:
                tagged_parts.append((pg_text, f"<pg>{pg_text}</pg>"))

    
    ptr_elem = root.find(".//tei:ptr", namespaces=ns)
    if ptr_elem is not None:
        # print('PTR')
        target_url = ptr_elem.get("target")
        if target_url:
            if "doi.org/" in target_url.lower():
                tagged_parts.append((target_url, f"<doi>{target_url}</doi>"))
            else:
                tagged_parts.append((target_url, f"<uri>{target_url}</uri>"))


    doi_el = root.xpath(".//tei:idno[@type='DOI']", namespaces=ns)
    if doi_el:
        doi_value = doi_el[0].text.strip() if doi_el[0].text else ""

        # 2. Match full DOI URL in original_ref (with <...>)
        full_doi_url_match = re.search(r"<(https?://dx\.doi\.org/[^>]+)>", original_ref)
        if full_doi_url_match:
            full_url = full_doi_url_match.group(1)
            tagged_parts.append((f"<{full_url}>", f"<doi><http://dx.doi.org/{doi_value}</doi></http>"))
        else:
            # fallback to just DOI
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




# def process_references():
#     if not os.path.exists(INPUT_FILE):
#         print(f"Input file '{INPUT_FILE}' not found.")
#         return

#     with open(INPUT_FILE, "r", encoding="utf-8") as f:
#         references = [line.strip() for line in f if line.strip()]

#     with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
#         for i, ref in enumerate(references, 1):
#             try:

#                 type = analyze_type(ref)
#                 # Remove <i> and <b> tags, store formatting info
#                 cleaned_ref, tags_info = remove_style_tags(ref)

#                 # Send to GROBID
#                 response = requests.post(
#                     GROBID_URL,
#                     headers={"Accept": "application/xml"},
#                     data={"citations": cleaned_ref}
#                 )
#                 if response.status_code != 200:
#                     print(f"[{i}] Failed to process reference: {ref}")
#                     continue

#                 xml_string = response.text
                # with open(XML_OUTPUT_FILE, "a", encoding="utf-8") as xml_out:
                #     xml_out.write(f"===== Reference {i:03} =====\n")
                #     xml_out.write(f"Original: {ref}\n")
                #     xml_out.write(xml_string.strip() + "\n\n")

#                 wrapped_xml = wrap_with_tei_namespace(xml_string)
#                 if type== "<journal>":
#                     tagged_elements = extract_tagged_elements_journal(wrapped_xml, cleaned_ref)

#                 elif type == "<book>":
#                     tagged_elements = extract_tagged_elements_book(wrapped_xml, cleaned_ref)

#                 elif type == "<chapter>":
#                     tagged_elements = extract_tagged_elements_chapter(wrapped_xml, cleaned_ref)

#                 tagged_output = tag_reference(type,cleaned_ref, tagged_elements)

#                 # Restore <i> and <b> tags to tagged_output
#                 tagged_output = restore_style_tags(tagged_output, tags_info)
#                 # print("Tagged output after restoring styles:", tagged_output)
#                 # 



#                 if type == "<chapter>":
#                     tagged_output = convert_first_btl_to_ctl(tagged_output)

#                 out_f.write(f"===== Reference {i:03} =====\n")
#                 out_f.write(tagged_output + "\n\n")
#                 # print("Type : ",type)
#                 print(f"[{i}] Processed successfully.")

#             except Exception as e:
#                 print(f"[{i}] Error processing reference: {ref}\n{e}")

from flask import Flask, request, jsonify
from collections import OrderedDict


app = Flask(__name__)



@app.route('/convert_Ref', methods=['POST'])


def process_references():
    input_data = request.get_json()
    if not isinstance(input_data, list):
        input_data = [input_data]

    results = []

    for entry in input_data:
        process_id = entry.get("process_id")
        html_ref = entry.get("element_outer_html", "")
        structured_ref = None

        try:
            # Extract reference inside <ref>...</ref>
            match = re.search(r"<ref>(.*?)</ref>", html_ref, re.DOTALL)
            if not match:
                raise ValueError("No <ref> tag found in input")
            original_ref = match.group(1).strip()

            # Detect reference type
            # ref_type = analyze_type(original_ref)
            ref_type='journal'

            # Remove <i> and <b> tags for processing, retain tag info
            cleaned_ref, tags_info = remove_style_tags(original_ref)

            # Send to GROBID
            response = requests.post(
                GROBID_URL,
                headers={"Accept": "application/xml"},
                data={"citations": cleaned_ref}
            )
            if response.status_code != 200:
                raise Exception("GROBID request failed")

            xml_string = response.text
            with open(XML_OUTPUT_FILE, "a", encoding="utf-8") as xml_out:
                xml_out.write(f"Original: {cleaned_ref}\n")
                xml_out.write(xml_string.strip() + "\n\n")

            wrapped_xml = wrap_with_tei_namespace(xml_string)

            # Extract tags by type
            if ref_type == "journal":
                tagged_elements = extract_tagged_elements_journal(wrapped_xml, cleaned_ref)
            elif ref_type == "book":
                tagged_elements = extract_tagged_elements_book(wrapped_xml, cleaned_ref)
            elif ref_type == "chapter":
                tagged_elements = extract_tagged_elements_chapter(wrapped_xml, cleaned_ref)
            else:
                tagged_elements = []

            # Compose structured tagged output
            structured = tag_reference(ref_type, cleaned_ref, tagged_elements)

            # Restore style tags
            structured = restore_style_tags(structured, tags_info)

            # Adjust chapter structure
            if ref_type == "chapter":
                structured = convert_first_btl_to_ctl(structured)

            # Final XML wrap
            structured_ref = f"""<ref><mixed-citation type="{ref_type}">{structured}</mixed-citation></ref>"""

        except Exception as e:
            print(f"[{process_id}] Error: {e}")
            structured_ref = None

        results.append(OrderedDict([
            ("process_id", process_id),
            ("element_outer_html", html_ref),
            ("structured_ref", structured_ref)
        ]))

    return (results)


if __name__ == "__main__":
 
    app.run(host="IS-S3315",port=5000,debug=True)