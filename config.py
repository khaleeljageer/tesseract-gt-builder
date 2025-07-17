from typing import Protocol


class FontConfigProtocol(Protocol):
    """Protocol defining the interface for font configurations"""

    PADDING: int
    FONT_SIZE: int
    A4_WIDTH_MM: int
    A4_HEIGHT_MM: int
    MM_PER_INCH: float
    DPI: int
    LINE_SPACING: int
    LINES_PER_PAGE: int

    @property
    def A4_WIDTH(self) -> int:
        ...

    @property
    def A4_HEIGHT(self) -> int:
        ...


class AnekTamilConfig:
    """Configuration for AnekTamil font"""

    PADDING = 8
    FONT_SIZE = 40
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297
    MM_PER_INCH = 25.4
    DPI = 300
    LINE_SPACING = 20
    LINES_PER_PAGE = 50

    @property
    def A4_WIDTH(self) -> int:
        return int(self.A4_WIDTH_MM * self.DPI / self.MM_PER_INCH)

    @property
    def A4_HEIGHT(self) -> int:
        return int(self.A4_HEIGHT_MM * self.DPI / self.MM_PER_INCH)


class ArialConfig:
    """Configuration for Arial font"""

    PADDING = 10
    FONT_SIZE = 36
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297
    MM_PER_INCH = 25.4
    DPI = 300
    LINE_SPACING = 18
    LINES_PER_PAGE = 55

    @property
    def A4_WIDTH(self) -> int:
        return int(self.A4_WIDTH_MM * self.DPI / self.MM_PER_INCH)

    @property
    def A4_HEIGHT(self) -> int:
        return int(self.A4_HEIGHT_MM * self.DPI / self.MM_PER_INCH)
