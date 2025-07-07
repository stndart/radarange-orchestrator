from functools import cmp_to_key
from typing import Any, Optional

from docling_core.types.doc import DoclingDocument

from .models import BaseElement, BBox, ElementType, GroupElement, ProvItem, Ref, RefName


class DoclingDocumentProcessor:
    doc: dict[str, Any]
    element_map: dict[RefName, tuple[ElementType, int]]
    group_counter: int

    def __init__(self, document: dict[str, Any] | DoclingDocument):
        if isinstance(document, DoclingDocument):
            self.doc = document.export_to_dict()
        else:
            self.doc = document

        self.element_map = self._build_element_map()
        self.group_counter = len(self.doc.get('groups', []))

    def _build_element_map(self) -> dict[RefName, tuple[ElementType, int]]:
        """Map self_ref to (element_type, index)"""
        element_map = {}
        for element_type in ['texts', 'pictures', 'tables']:
            if element_type in self.doc:
                for idx, element in enumerate(self.doc[element_type]):
                    element_map[element['self_ref']] = (element_type, idx)
        return element_map

    def _get_element(self, ref: RefName) -> BaseElement:
        """Get element by reference"""
        if ref == '#/body':
            return self.doc['body']
        element_type, idx = self.element_map[ref]
        return self.doc[element_type][idx]

    def _update_element(self, ref: RefName, element: dict[str, Any]):
        """Update element in document"""
        element_type, idx = self.element_map[ref]
        self.doc[element_type][idx] = element

    def _unchild_element(self, ref_parent: RefName, ref_child: RefName) -> None:
        parent = self._get_element(ref_parent)
        parent['children'].remove({'$ref': ref_child})
        child = self._get_element(ref_child)
        child['parent']['$ref'] = ''

    def _calculate_union_bbox(self, elements: list[BaseElement]) -> list[ProvItem]:
        """Calculate union bounding box for multiple elements"""
        all_provs: list[ProvItem] = []
        for element in elements:
            all_provs.extend(element['prov'])

        # Group by page number
        page_provs: dict[int, list[BBox]] = dict()
        for prov in all_provs:
            page_no = prov['page_no']
            if page_no not in page_provs:
                page_provs[page_no] = []
            page_provs[page_no].append(prov['bbox'])

        # Calculate union per page
        union_provs = []
        for page_no, bboxes in page_provs.items():
            L = min(bbox['l'] for bbox in bboxes)
            R = max(bbox['r'] for bbox in bboxes)
            T = max(bbox['t'] for bbox in bboxes)
            B = min(bbox['b'] for bbox in bboxes)
            coord_origin = bboxes[0]['coord_origin']
            assert all([bbox['coord_origin'] == coord_origin for bbox in bboxes])

            union_provs.append(
                ProvItem(
                    page_no=page_no,
                    bbox=BBox(l=L, t=T, r=R, b=B, coord_origin=coord_origin),
                )
            )

        return union_provs

    def _is_picture_ref(self, ref: RefName) -> bool:
        """True if ref identifies a picture element in element_map."""
        entry = self.element_map.get(ref)
        return bool(entry and entry[0] == 'pictures')

    def _is_caption_ref(self, ref: RefName) -> bool:
        """True if ref identifies a caption element in element_map."""
        entry = self.element_map.get(ref)
        return bool(entry and entry[0] == 'caption')

    def _extract_picture_block(
        self,
        children: list[Ref],
        start_index: int,
    ) -> tuple[list[RefName], Optional[RefName], int] | None:
        """
        If children[start_index:] begins with ≥2 pictures,
        collect all consecutive picture refs, then (optionally)
        capture one following caption element.

        Returns:
            (picture_refs, caption_ref_or_None, next_index)
        or
            None  — if fewer than 2 pictures in a row.
        """
        pics: list[RefName] = []
        i = start_index
        # collect pictures
        while i < len(children):
            ref = children[i]['$ref']
            if self._is_picture_ref(ref):
                pics.append(children[i]['$ref'])
                i += 1
            else:
                break

        # need at least two pictures to group
        if len(pics) < 2:
            return None

        # check for caption immediately after
        caption = None
        if i < len(children):
            maybe_caption_ref = children[i]
            if self._is_caption_ref(maybe_caption_ref['$ref']):
                caption = maybe_caption_ref['$ref']
                i += 1

        return pics, caption, i

    def _create_picture_caption_group(
        self, picture_refs: list[RefName], caption_ref: Optional[RefName]
    ) -> RefName:
        """
        Given a list of picture refs and an optional caption ref,
        (1) collects all element objects,
        (2) re‐parents them into a new GroupElement,
        (3) updates the element store,
        (4) appends the group to self.doc['groups'],
        (5) returns the new group’s self_ref string.
        """
        # 1. allocate group ID
        group_id = self.group_counter
        self.group_counter += 1
        group_ref: RefName = f'#/groups/{group_id}'

        # 2. fetch elements by ref
        all_elements = [self._get_element(ref) for ref in picture_refs]
        if caption_ref:
            all_elements.append(self._get_element(caption_ref))

        # 3. reparent & update each element
        for el in all_elements:
            self._unchild_element(el['parent']['$ref'], el['self_ref'])
            el['parent'] = {'$ref': group_ref}
            self._update_element(el['self_ref'], el)

        # 4. build the new group element
        children_refs = [Ref({'$ref': el['self_ref']}) for el in all_elements]
        # group doesn't have prov field
        # union_bbox = self._calculate_union_bbox(all_elements)
        group_el = GroupElement(
            self_ref=group_ref,
            parent={'$ref': '#/body'},
            children=children_refs,
            content_layer='body',
        )

        # 5. record it
        self.doc.setdefault('groups', []).append(group_el)
        self.element_map[group_ref] = ('groups', len(self.doc['groups']) - 1)

        return group_ref

    def _sorted_by_y(self, elements: list[Ref]) -> list[Ref]:
        """
        Return elements sorted by descending top‑coordinate (prov[0]['bbox']['t']).
        """

        def compare(ref1: Ref, ref2: Ref):
            p1 = self._get_element(ref1['$ref'])['prov'][0]
            p2 = self._get_element(ref2['$ref'])['prov'][0]
            return p2['bbox']['t'] - p1['bbox']['t']

        return sorted(elements, key=cmp_to_key(compare))

    def _merge_picture_blocks(self) -> None:
        """
        Find runs of >=2 pictures (plus optional caption) in the body,
        wrap each run in a new GroupElement, and insert its ref back into the doc.
        """

        body_children = self._sorted_by_y(self.doc['body']['children'])
        i = 0

        while i < len(body_children):
            # Try to extract a block of pictures (and optional caption) at i
            block = self._extract_picture_block(body_children, start_index=i)

            if block:
                pics, caption, next_index = block
                first_body_occurence: int = self.doc['body']['children'].index(
                    {'$ref': pics[0]}
                )
                print(f'Found group! {pics}/{caption}')
                group_ref = self._create_picture_caption_group(pics, caption)
                self.doc['body']['children'].insert(
                    first_body_occurence, {'$ref': group_ref}
                )
                i = next_index
            else:
                # No grouping here—just copy the element
                i += 1

    def _is_caption_candidate(
        self,
        text_el: BaseElement,
        picture_el: BaseElement,
        vertical_tolerance: float = 0.02,
        horizontal_tolerance: float = 0.1,
        check_allignment: bool = False,
    ) -> bool:
        """
        Determine if a text element is likely a caption for a picture.

        Conditions:
        1. Text must be below the picture (within vertical tolerance)
        2. Text width must be ≤ picture width (within horizontal tolerance)
        3. Horizontal alignment must match (centered or left-aligned)

        Tolerances are fractions of the picture's dimensions.
        """
        # Get bounding boxes (using first prov item for simplicity)
        pic_bbox = picture_el['prov'][0]['bbox']
        text_bbox = text_el['prov'][0]['bbox']

        # Calculate dimensions
        pic_height = pic_bbox['t'] - pic_bbox['b']
        pic_width = pic_bbox['r'] - pic_bbox['l']
        text_width = text_bbox['r'] - text_bbox['l']

        # Check if text is below the picture
        if text_bbox['t'] > pic_bbox['b']:
            return False  # Text is above the picture

        # Calculate vertical gap (picture bottom to text top)
        vertical_gap = pic_bbox['b'] - text_bbox['t']

        # Check vertical proximity
        if vertical_gap > vertical_tolerance * pic_height:
            return False

        # Check width relationship
        if text_width > (1 + horizontal_tolerance) * pic_width:
            return False

        if not check_allignment:
            return True

        # Check horizontal alignment
        text_center = (text_bbox['l'] + text_bbox['r']) / 2
        pic_center = (pic_bbox['l'] + pic_bbox['r']) / 2
        center_diff = abs(text_center - pic_center)

        # Allow either centered or left-aligned
        is_centered = center_diff < horizontal_tolerance * pic_width
        is_left_aligned = (
            abs(text_bbox['l'] - pic_bbox['l']) < horizontal_tolerance * pic_width
        )

        return is_centered or is_left_aligned

    def _find_caption_for_picture(
        self,
        picture_ref: RefName,
        candidate_texts: list[BaseElement],
        vertical_tolerance: float = 0.02,
        horizontal_tolerance: float = 0.1,
    ) -> Optional[RefName]:
        """
        Find the best caption candidate for a picture from a list of text elements.
        Returns the self_ref of the best candidate if found, otherwise None.
        """
        picture_el = self._get_element(picture_ref)
        best_candidate = None
        best_score = float('inf')

        for text_el in candidate_texts:
            if not self._is_caption_candidate(
                text_el, picture_el, vertical_tolerance, horizontal_tolerance
            ):
                continue

            # Calculate distance score (closer is better)
            pic_bbox = picture_el['prov'][0]['bbox']
            text_bbox = text_el['prov'][0]['bbox']
            vertical_dist = (
                pic_bbox['b'] - text_bbox['t']
            )  # Positive when text is below picture
            horizontal_dist = abs(
                (pic_bbox['l'] + pic_bbox['r']) / 2
                - (text_bbox['l'] + text_bbox['r']) / 2
            )

            # Weight vertical distance more heavily
            score = 2 * vertical_dist + horizontal_dist

            if score < best_score:
                best_score = score
                best_candidate = text_el

        return best_candidate['self_ref'] if best_candidate else None

    def _merge_captions_into_pictures(
        self, vertical_tolerance: float = 0.02, horizontal_tolerance: float = 0.1
    ) -> None:
        """
        Main method to associate text elements as captions for nearby pictures.
        Operates directly on the document structure.
        """
        # Collect all text elements that might be captions
        candidate_texts = []
        for text in self.doc.get('texts', []):
            if text['label'] == 'text':  # Only consider unclassified text
                candidate_texts.append(text)

        # Process each picture
        for picture in self.doc.get('pictures', []):
            picture: BaseElement
            picture_ref = picture['self_ref']
            caption_ref = self._find_caption_for_picture(
                picture_ref, candidate_texts, vertical_tolerance, horizontal_tolerance
            )

            if caption_ref:
                # Remove caption from candidates list
                candidate_texts = [
                    t for t in candidate_texts if t['self_ref'] != caption_ref
                ]

                # Get the caption element
                caption_el = self._get_element(caption_ref)

                # Reparent caption to picture
                self._unchild_element(caption_el['parent']['$ref'], caption_ref)
                caption_el['parent'] = {'$ref': picture_ref}

                # Add to picture's children
                picture['children'].append({'$ref': caption_ref})

                # Expand picture's bbox
                union_bbox = self._calculate_union_bbox([picture, caption_el])
                picture['prov'][0]['bbox'] = union_bbox[0]['bbox']

                self._update_element(picture_ref, picture)
                self._update_element(caption_ref, caption_el)

    def _free_captions(self) -> None:
        """
        Unparents caption texts from pictures
        """
        # Process each picture
        for picture in self.doc.get('pictures', []):
            picture: BaseElement
            for caption_fref in picture['children']:
                caption_ref = caption_fref['$ref']
                caption = self._get_element(caption_ref)
                if caption['label'] == 'caption':
                    # Reparent caption from picture
                    self._unchild_element(caption['parent']['$ref'], caption_ref)
                    caption['parent'] = {'$ref': '#/body'}

                    # Add to body children
                    idx = self.doc['body']['children'].index(
                        {'$ref': picture['self_ref']}
                    )
                    self.doc['body']['children'].insert(idx + 1, caption_fref)
                    self._update_element(caption_ref, caption)

    def process_document(self) -> None:
        """
        Execute all processing steps on self.doc.
        """

        self._free_captions()
        self._merge_captions_into_pictures()
        self._merge_picture_blocks()
