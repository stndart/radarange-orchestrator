from typing import Literal, Optional, TypedDict

RefName = str
Ref = TypedDict('Ref', {'$ref': RefName})

ElementType = Literal[
    'text', 'picture', 'caption', 'table', 'listitem', 'formula', 'page_footer'
]


class BBox(TypedDict):
    l: float  # noqa: E741
    t: float
    r: float
    b: float
    coord_origin: str = 'BOTTOMLEFT'


class ProvItem(TypedDict):
    page_no: int
    bbox: BBox
    charspan: tuple[int, int] = (0, 0)


class BaseElement(TypedDict):
    self_ref: RefName
    parent: Ref
    children: list[Ref]
    content_layer: str
    label: ElementType
    prov: list[ProvItem]


class TextElement(BaseElement):
    orig: Optional[str] = None
    text: Optional[str] = None


class PictureElement(BaseElement):
    captions: list = []
    references: list = []
    footnotes: list = []
    annotations: list = []


class FormulaElement(BaseElement):
    pass


class GroupElement(BaseElement):
    pass
